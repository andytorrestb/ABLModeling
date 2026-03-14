import pystencils as ps
from lbmpy.session import (
    LBStencil, Stencil, LBMConfig, Method, create_lb_method,
    create_lb_update_rule, LBMOptimisation, LatticeBoltzmannBoundaryHandling,
    UBB, NoSlip, ExtrapolationOutflow, FreeSlip, slice_from_direction
)

from lbmpy.relaxationrates import relaxation_rate_from_lattice_viscosity
from lbmpy.macroscopic_value_kernels import pdf_initialization_assignments
import numpy as np
import os
import cv2
import time
import json
import logging
logger = logging.getLogger(__name__)

from config_sim import SimulationConfig
from geometry import GeometryFactory

# We only import the function we actually use now
from postprocessing import save_velocity_slices_npz, compute_slice_indices

# ==================================================================================================
#                               SCIENTIFIC DOCUMENTATION: UNITS & SCALING
# ==================================================================================================
# This simulation operates in LATTICE UNITS (LU), not Physical Units (SI).
#
# scaling_concept: "Acoustic Scaling"
#   - delta_x (lattice spacing) = 1 [LU]
#   - delta_t (time step)       = 1 [TS]
#
# Primary Conversions:
#   1. Reynolds Number (Re) is non-dimensional and identical in both systems.
#      Re = (U_phys * L_phys) / nu_phys  ==  (U_lattice * L_lattice) / nu_lattice
#
#   2. Velocity (U):
#      U_lattice = U_phys * (delta_t / delta_x)
#      *CRITICAL*: To ensure compressibility errors are negligible (Ma < 0.1),
#      the characteristic lattice velocity (maximal_velocity) should generally be < 0.1.
#
#   3. Viscosity (nu):
#      nu_lattice = (U_lattice * L_lattice) / Re
#      Relaxation time (tau) corresponds to: nu_lattice = (2*tau - 1) / 6
#      Ideally, tau should be kept in the range [0.51, ~2.0] for stability.
#
#   4. Length (L):
#      L_lattice = L_phys / delta_x
#      This is the resolution of the simulation (e.g., "reference_length" in config).
#
# For data analysis, convert results back to physical units using the inverse factors:
#   U_phys = U_lattice * (dx_phys / dt_phys)
#   L_phys = L_lattice * dx_phys
#   T_phys = T_lattice * dt_phys
# ==================================================================================================

# -------------------------
# Memory probing helper
# -------------------------
try:
    import psutil
    import tracemalloc
except Exception:
    psutil = None
    tracemalloc = None

class MemProbe:
    """Context manager to log RSS and a small tracemalloc diff around a code region."""
    def __init__(self, label: str, top_n: int = 5):
        self.label = label
        self.top_n = top_n
        self.log = logging.getLogger(__name__)

    def __enter__(self):
        self.proc = psutil.Process() if psutil else None
        if tracemalloc:
            tracemalloc.start()
            self.snap_before = tracemalloc.take_snapshot()
        self.t0 = time.time()
        self.rss_before = self.proc.memory_info().rss if self.proc else None
        if self.rss_before is not None:
            self.log.debug("%s: RSS before %.2f MB", self.label, self.rss_before / 1024**2)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.proc:
            rss_after = self.proc.memory_info().rss
            rss_delta = (rss_after - (self.rss_before or 0)) / 1024**2
            elapsed = time.time() - self.t0
            self.log.info("%s: RSS after %.2f MB (Δ %.2f MB) elapsed %.3fs", self.label, rss_after / 1024**2, rss_delta, elapsed)
        if tracemalloc:
            snap_after = tracemalloc.take_snapshot()
            stats = snap_after.compare_to(self.snap_before, 'lineno')
            for s in stats[: self.top_n]:
                self.log.debug("%s: tracemalloc %s", self.label, s)
            tracemalloc.stop()

# memory_profiler decorator fallback
try:
    from memory_profiler import profile
except Exception:
    def profile(func):  # no-op
        return func


def set_geometry_mask(x, y, z, config):
    """
    Delegates geometry creation to the GeometryFactory.
    This replaces the old 'set_ship' hardcoded function.
    """
    geometry_handler = GeometryFactory.get_geometry(config)
    return geometry_handler(x, y, z)


def save_video(output_dir, video_filename="output.mp4", fps=10):
    os.makedirs(output_dir, exist_ok=True)
    # Get all image files in the directory (e.g., png, jpg)
    images = [img for img in sorted(os.listdir(output_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found in the directory.")
        return

    # Read the first image to get frame size
    first_frame = cv2.imread(os.path.join(output_dir, images[0]))
    height, width, layers = first_frame.shape

    # Define the video writer
    video_path = os.path.join(output_dir, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write each image as a frame
    for img_name in images:
        frame = cv2.imread(os.path.join(output_dir, img_name))
        video.write(frame)

    video.release()
    # print(f"Video saved as {video_path}")



class LBMSolver:
    """
    Main Lattice Boltzmann Method solver class.

    SCIENTIFIC METHODS:
    - Method: Cumulant LBM (Geier et al., 2015).
        * Optimizes Galilean invariance.
        * Eliminates 'ghost modes' common in MRT.
    - Stencil: D3Q27 (3 Dimensions, 27 Discrete Velocities).
    """
    def __init__(self, config):
        self.config = config
        self.stencil = LBStencil(Stencil.D3Q27)
        self.dh = None
        self.method = None
        self.kernel_init = None
        self.kernel_update = None
        self.bh = None
        
        self._setup_data_handling()
        self._setup_method()
        self._compile_kernels()
        self._setup_boundaries()
        self._run_initialization()

    def _setup_data_handling(self):
        with MemProbe("create_data_handling"):
            # Choose data handling default target/device based on config
            dh_target = ps.Target.CPU
            if getattr(self.config, 'kernel_target', None) and str(self.config.kernel_target).lower() == 'gpu':
                dh_target = ps.Target.GPU
            self.dh = ps.create_data_handling(
                domain_size=self.config.domain_size,
                periodicity=(False, False),
                default_target=dh_target,
                device_number=getattr(self.config, 'gpu_device', None)
            )

        with MemProbe("add_arrays_and_fill"):
            self.src = self.dh.add_array('src', values_per_cell=len(self.stencil), alignment=True)
            self.dh.fill('src', 0.0, ghost_layers=True)
            self.dst = self.dh.add_array('dst', values_per_cell=len(self.stencil), alignment=True)
            self.dh.fill('dst', 0.0, ghost_layers=True)
            self.velField = self.dh.add_array('velField', values_per_cell=self.dh.dim, alignment=True)
            self.dh.fill('velField', 0.0, ghost_layers=True)

        try:
            logger.debug("Approx src nvytes: %s", getattr(self.src, 'nbytes', 'n/a'))
        except Exception:
            pass

    def _setup_method(self):
        with MemProbe("create_lb_model"):
            # Relaxation parameter 'omega' is related to kinematic viscosity:
            # omega = 1 / (3 * nu_lattice + 0.5)
            omega = relaxation_rate_from_lattice_viscosity(self.config.kinematic_viscosity)
            self.lbm_config = LBMConfig(stencil=Stencil.D3Q27, method=Method.CUMULANT, relaxation_rate=omega,
                                compressible=True,
                                output={'velocity': self.velField}, kernel_type='stream_pull_collide')

            self.method = create_lb_method(lbm_config=self.lbm_config)

    def _get_backend_and_target(self):
        # Allow overriding target via config (simulation.kernel_target)
        target_override = None
        try:
            if getattr(self.config, 'kernel_target', None):
                kt = str(self.config.kernel_target).lower()
                if kt == 'cpu':
                    target_override = ps.Target.CPU
                elif kt == 'gpu':
                    target_override = ps.Target.GPU
        except Exception:
            target_override = None

        target = target_override or self.dh.default_target
        backend = ps.Backend.CUDA if target == ps.Target.GPU else ps.Backend.C
        return target, backend

    def _compile_kernels(self):
        target, backend = self._get_backend_and_target()

        # Initialization Kernel
        init = pdf_initialization_assignments(self.method, 1.0, self.config.initial_velocity, self.src.center_vector)
        with MemProbe("kernel_init_compile"):
            ast_init = ps.create_kernel(init, target=target, backend=backend)
            self.kernel_init = ast_init.compile()

        # Update Kernel
        lbm_optimisation = LBMOptimisation(symbolic_field=self.src, symbolic_temporary_field=self.dst)
        update = create_lb_update_rule(lb_method=self.method,
                                    lbm_config=self.lbm_config,
                                    lbm_optimisation=lbm_optimisation)

        with MemProbe("kernel_update_compile"):
            ast_kernel = ps.create_kernel(
                update,
                target=target,
                backend=backend,
                cpu_openmp=getattr(self.config, 'cpu_openmp', False)
            )
            self.kernel_update = ast_kernel.compile()

    def _setup_boundaries(self):
        self.bh = LatticeBoltzmannBoundaryHandling(self.method, self.dh, 'src', name="bh")

        wall = NoSlip("wall")
        inflow = UBB(self.config.initial_velocity)
        
        # stencil[4] corresponds to the standard "East" / Positive-X direction in D3Q27
        # This aligns with the domain outflow boundary.
        outflow = ExtrapolationOutflow(self.stencil[4], self.method)

        def get_boundary_obj(b_type, direction_label):
            b_type = b_type.lower()
            if b_type == 'noslip':
                return wall
            elif b_type == 'inflow':
                return inflow
            elif b_type == 'outflow':
                return outflow
            elif b_type == 'freeslip':
                normals = {
                    'N': (0, 1, 0), 'S': (0, -1, 0),
                    'T': (0, 0, 1), 'B': (0, 0, -1),
                    'E': (1, 0, 0), 'W': (-1, 0, 0)
                }
                if direction_label in normals:
                    return FreeSlip(self.method.stencil, normal_direction=normals[direction_label], name=f"freeSlip_{direction_label}")
                else:
                    logger.warning("Unknown direction %s for FreeSlip, defaulting to NoSlip", direction_label)
                    return wall
            else:
                logger.warning("Unknown boundary type %s, defaulting to NoSlip", b_type)
                return wall

        # Apply BCs from configuration
        dim = self.dh.dim
        for direction_label, b_type in self.config.boundary_conditions.items():
            boundary_obj = get_boundary_obj(b_type, direction_label)
            self.bh.set_boundary(boundary_obj, slice_from_direction(direction_label, dim))

        # Apply no-slip to obstacle
        self.bh.set_boundary(NoSlip("obstacle"), mask_callback=lambda x, y, z, *_: set_geometry_mask(x, y, z, self.config))

    def _run_initialization(self):
         self.dh.run_kernel(self.kernel_init)

    def step(self):
        self.bh()
        self.dh.run_kernel(self.kernel_update)
        self.dh.swap("src", "dst")


def run_simulation(reynolds_number, ref_length, cfg):
    # Step 1) Create output directory for specific configuration
    outdir = f"output_Re_{int(reynolds_number):d}_L_{ref_length:d}"
    config = SimulationConfig(reynolds_number, ref_length, outdir, cfg)

    logger.info("Running simulation with Re=%.0e, L=%d, nu=%.2e, U=%s", 
                reynolds_number, ref_length, config.kinematic_viscosity, config.maximal_velocity)

    # Step 2) Defensive memory check
    domain_size = config.domain_size
    n_cells = domain_size[0] * domain_size[1] * domain_size[2]
    n_arrays = 3  # src, dst, velField
    bytes_per_cell = 8  # float64
    estimated_mem_gb = n_cells * n_arrays * bytes_per_cell / (1024**3)
    max_mem_gb = 1  # Set your max allowed RAM (GB)
    if estimated_mem_gb > max_mem_gb:
        logger.warning("Skipping simulation for L=%d, Re=%.0e: Estimated memory %.2f GB exceeds limit of %d GB.", 
                       ref_length, reynolds_number, estimated_mem_gb, max_mem_gb)
        return
    else:
        logger.info("Memory check passed for L=%d, Re=%.0e: %.2f GB used.", 
                    ref_length, reynolds_number, estimated_mem_gb)

    # Step 3) Initialize Solver
    solver = LBMSolver(config)

    # Step 4) Save config used
    with open(os.path.join(config.outdir, "config_used.json"), 'w') as f:
        # Reconstruct a dict for JSON dumping
        cfg_dump = {
            "simulation": {
                "n_time_steps": config.n_time_steps,
                "output_interval": config.output_interval,
                "reynolds_number": config.reynolds_number,
                "reference_length": config.reference_length,
                "kernel_target": getattr(config, 'kernel_target', 'default'),
                "gpu_device": getattr(config, 'gpu_device', None)
            },
            "domain": {
                "domain_size": config.domain_size,
                "maximal_velocity": config.maximal_velocity,
                "initial_velocity": config.initial_velocity
            },
            "derived": {
                "kinematic_viscosity": config.kinematic_viscosity
            }
        }
        json.dump(cfg_dump, f, indent=2)

    # Timing counters
    solver_time = 0.0
    export_time = 0.0
    
    t0 = time.time() # Start whole timer

    with MemProbe("main_simulation_loop"):
        for time_step in range(1, config.n_time_steps + 1):
            
            t_solver_start = time.perf_counter()
            solver.step()
            solver_time += (time.perf_counter() - t_solver_start)
            
            if time_step % config.output_interval == 0:
                t_export_start = time.perf_counter()
                
                # --- MEMORY-EFFICIENT SLICING ---
                try:
                    dh = solver.dh # Access data handling from solver
                    
                    # Calculate indices
                    y_idx, z_idx = compute_slice_indices(config.domain_size)
                    
                    # Assume single-block (standard for this repo)
                    gl = dh.ghost_layers if hasattr(dh, 'ghost_layers') and dh.ghost_layers is not None else 1
                    sl_res = slice(gl, -gl) if gl > 0 else slice(None)

                    is_gpu = (dh.default_target == ps.Target.GPU)

                    
                    z_slice = None
                    y_slice = None

                    if is_gpu:
                        # --- GPU Path ---
                        try:
                            import cupy as cp
                        except ImportError:
                            logger.error("GPU target active but cupy not found for slicing.")
                            raise

                        # Access first block's GPU array
                        if not dh.gpu_arrays:
                             raise RuntimeError("dh.gpu_arrays is empty but target is GPU")
                        gpu_arr = next(iter(dh.gpu_arrays.values()))
                        
                        # Slicing on GPU
                        # array shape: (nx+2gl, ny+2gl, nz+2gl, components)
                        z_mem = z_idx + gl
                        y_mem = y_idx + gl
                        
                        # z-cut: xy plane
                        z_cut_gpu = gpu_arr[sl_res, sl_res, z_mem, :2] 
                        z_slice = cp.asnumpy(z_cut_gpu)
                        
                        # y-cut: xz plane
                        y_cut_gpu = gpu_arr[sl_res, y_mem, sl_res, :]
                        y_slice = cp.asnumpy(y_cut_gpu[..., [0, 2]])
                        
                    else:
                        # --- CPU Path ---
                        if not dh.cpu_arrays:
                             raise RuntimeError("dh.cpu_arrays is empty")
                        cpu_arr = next(iter(dh.cpu_arrays.values()))
                        
                        z_mem = z_idx + gl
                        y_mem = y_idx + gl
                        
                        z_slice = cpu_arr[sl_res, sl_res, z_mem, :2].copy()
                        y_cut = cpu_arr[sl_res, y_mem, sl_res, :]
                        y_slice = y_cut[..., [0, 2]].copy()

                    # Save compressed slices
                    save_velocity_slices_npz(z_slice, y_slice, y_idx, z_idx, time_step, config)
                    logger.debug("Saved slices for time_step %d", time_step)

                except Exception as e:
                    logger.warning("Slice save failed at time_step %s: %s", time_step, e)
                    import traceback
                    logger.debug(traceback.format_exc())
                    
                # Optional: Log scalar metrics to a simple CSV (append mode)
                try:
                    stats_csv = os.path.join(config.outdir, "run_stats.csv")
                    # Check if header needs writing
                    write_header = not os.path.exists(stats_csv)
                    with open(stats_csv, "a") as f:
                        if write_header:
                            f.write("time_step,time_elapsed,max_vel_mag\n")
                        f.write(f"{time_step},{time.time()-t0:.4f},0.0\n")
                except Exception:
                    pass
                
                export_time += (time.perf_counter() - t_export_start)

    total_time = time.time() - t0
    logger.info("Simulation completed in %.2fs (Solver: %.2fs, Export: %.2fs)", total_time, solver_time, export_time)
    
    # Save timing stats for post-processing summary
    with open(os.path.join(config.outdir, "timing_sim.json"), "w") as f:
        json.dump({
            "total_simulation_time_sec": total_time,
            "solver_time_sec": solver_time,
            "data_export_time_sec": export_time,
            "initialization_time_sec": max(0, total_time - solver_time - export_time)
        }, f, indent=2)
        
    return config

if __name__ == "__main__":
    CONFIG_FILE = "config.json"
    cfg = load_config(CONFIG_FILE)
    sim = cfg['simulation']
    phys = cfg['physical']

    reynolds_list = sim['reynolds_list']
    ref_length_list = sim['ref_length_list']
    L_phy = phys['L_phy']
    nu_phy = phys['nu_phy']
    
    print("Simulating the following cases")
    for reynolds_number in reynolds_list:
        u_phy = (reynolds_number * nu_phy) / L_phy
        print(f"Physical velocity scale (u_phy) for Re={reynolds_number:.0e}: {u_phy:.2f} m/s")
    
    for reynolds_number in reynolds_list:
        for ref_length in ref_length_list:
            sim_config = run_simulation(reynolds_number, ref_length, cfg)
            # save_video(sim_config.outdir + '/vel_magnitude_output' , video_filename="vel_magnitude.mp4", fps=sim_config.vel_video_fps)
            # save_video(sim_config.outdir + '/vorticity_output' , video_filename="vorticity.mp4", fps=sim_config.vort_video_fps)

