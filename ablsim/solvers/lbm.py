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

from ablsim.core.config import SimulationConfig
from ablsim.geometry.factory import GeometryFactory
from ablsim.post.processing import save_velocity_slices_npz, compute_slice_indices

# ==================================================================================================
#                               SCIENTIFIC DOCUMENTATION: UNITS & SCALING
# ==================================================================================================
# ... (Same as original)
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
            try:
                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                self.snap_before = tracemalloc.take_snapshot()
            except Exception:
                self.snap_before = None
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
        if tracemalloc and self.snap_before:
            try:
                snap_after = tracemalloc.take_snapshot()
                stats = snap_after.compare_to(self.snap_before, 'lineno')
                for s in stats[: self.top_n]:
                    self.log.debug("%s: tracemalloc %s", self.label, s)
                # tracemalloc.stop() # Do not stop globally
            except Exception:
                pass

def set_geometry_mask(x, y, z, config):
    """
    Delegates geometry creation to the GeometryFactory.
    """
    geometry_handler = GeometryFactory.get_geometry(config)
    return geometry_handler(x, y, z)

class LBMSolver:
    """
    Main Lattice Boltzmann Method solver class.
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

    def _setup_method(self):
        with MemProbe("create_lb_model"):
            omega = relaxation_rate_from_lattice_viscosity(self.config.kinematic_viscosity)
            self.lbm_config = LBMConfig(stencil=Stencil.D3Q27, method=Method.CUMULANT, relaxation_rate=omega,
                                compressible=True,
                                output={'velocity': self.velField}, kernel_type='stream_pull_collide')

            self.method = create_lb_method(lbm_config=self.lbm_config)

    def _get_backend_and_target(self):
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
                    return wall
            else:
                return wall

        # Apply BCs
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
    """
    Main entry point for running a case.
    """
    # 1. Prepare Config
    outdir = f"output_Re_{int(reynolds_number):d}_L_{ref_length:d}"
    # If outdir is an absolute path or relative, ensure we respect if cfg overrides it? 
    # For now, stick to legacy logic: folder named by params.
    # The 'cfg' passed here is a dict.
    
    config = SimulationConfig(reynolds_number, ref_length, outdir, cfg)

    # Logging derived params
    nu_lb = config.kinematic_viscosity
    omega = 1.0 / (3.0 * nu_lb + 0.5)
    tau = 1.0 / omega
    u_lb = config.maximal_velocity

    logger.info("Running simulation Re=%.0e, L=%d", reynolds_number, ref_length)
    logger.info("  nu_lb: %.6f, u_lb: %.4f, tau: %.4f", nu_lb, u_lb, tau)

    if u_lb > 0.1:
        logger.warning("  !! WARNING: u_lb > 0.1")
    if tau < 0.51:
        logger.warning("  !! WARNING: tau < 0.51")

    # 2. Memory Check
    domain_size = config.domain_size
    n_cells = np.prod(domain_size)
    estimated_mem_gb = n_cells * 3 * 8 / (1024**3)
    max_mem_gb = 32 # Default cap
    if estimated_mem_gb > max_mem_gb:
        logger.warning("Skipping sim: Estimated %.2f GB > %.2f GB", estimated_mem_gb, max_mem_gb)
        return config

    # 3. Init Solver
    solver = LBMSolver(config)

    # 4. Save Config Used
    with open(os.path.join(config.outdir, "config_used.json"), 'w') as f:
        cfg_dump = {
            "simulation": {
                "n_time_steps": config.n_time_steps,
                "output_interval": config.output_interval,
                "reynolds_number": config.reynolds_number,
                "reference_length": config.reference_length,
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

    # 5. Loop
    solver_time = 0.0
    export_time = 0.0
    t0 = time.time()

    with MemProbe("main_simulation_loop"):
        for time_step in range(1, config.n_time_steps + 1):
            
            t_solver_start = time.perf_counter()
            solver.step()
            solver_time += (time.perf_counter() - t_solver_start)
            
            if time_step % config.output_interval == 0:
                t_export_start = time.perf_counter()
                try:
                    dh = solver.dh
                    y_idx, z_idx = compute_slice_indices(config.domain_size)
                    gl = getattr(dh, 'ghost_layers', 1) 
                    if gl is None: gl = 1
                    sl_res = slice(gl, -gl) if gl > 0 else slice(None)
                    is_gpu = (dh.default_target == ps.Target.GPU)

                    z_slice = None
                    y_slice = None

                    if is_gpu:
                        import cupy as cp
                        if not dh.gpu_arrays: raise RuntimeError("No GPU arrays")
                        gpu_arr = next(iter(dh.gpu_arrays.values())) # 'src' or 'dst' depending on step?
                        # Note: dh.gpu_arrays might be keyed by field name.
                        # Wait, lbmpy DH stores current active field?
                        # We usually want 'src' or 'dst' that has valid data. 
                        # After step(), self.dh.swap("src", "dst") was called.
                        # So 'src' is the fresh one.
                        # But we should access `solver.src` specifically?
                        # solver.src is a field descriptor.
                        # dh.gather(solver.src) -> brings to CPU.
                        # But we want to slice on GPU if possible.
                        # Assuming 'src' is the valid one (as per loop: step runs update src->dst then swaps, so dst became src).
                        
                        # Let's use `dh.cpu_arrays[self.src.name]` etc logic.
                        # The code loop above did `solver.dh` access directly.
                        # We should robustly access the current source array.
                        # In the loop: step() -> update src->dst, then swap src,dst.
                        # So 'src' holds the new state.
                        
                        # Direct access:
                        arr_name = solver.src.name
                        if is_gpu:
                            gpu_arr = dh.gpu_arrays[arr_name]
                            z_mem = z_idx + gl
                            y_mem = y_idx + gl
                            z_cut_gpu = gpu_arr[sl_res, sl_res, z_mem, :2]
                            z_slice = cp.asnumpy(z_cut_gpu)
                            y_cut_gpu = gpu_arr[sl_res, y_mem, sl_res, :]
                            y_slice = cp.asnumpy(y_cut_gpu[..., [0, 2]])
                        else:
                            cpu_arr = dh.cpu_arrays[arr_name]
                            z_mem = z_idx + gl
                            y_mem = y_idx + gl
                            z_slice = cpu_arr[sl_res, sl_res, z_mem, :2].copy()
                            y_cut = cpu_arr[sl_res, y_mem, sl_res, :]
                            y_slice = y_cut[..., [0, 2]].copy()
                            
                    else:
                        # CPU Path
                        arr_name = solver.src.name
                        cpu_arr = dh.cpu_arrays[arr_name]
                        z_mem = z_idx + gl
                        y_mem = y_idx + gl
                        z_slice = cpu_arr[sl_res, sl_res, z_mem, :2].copy()
                        y_cut = cpu_arr[sl_res, y_mem, sl_res, :]
                        y_slice = y_cut[..., [0, 2]].copy()

                    save_velocity_slices_npz(z_slice, y_slice, y_idx, z_idx, time_step, config)

                except Exception as e:
                    logger.warning("Slice export failed: %s", e)
                    import traceback
                    logger.debug(traceback.format_exc())
                
                # Stats CSV
                try:
                    stats_csv = os.path.join(config.outdir, "run_stats.csv")
                    write_header = not os.path.exists(stats_csv)
                    with open(stats_csv, "a") as f:
                        if write_header: f.write("time_step,time_elapsed\n")
                        f.write(f"{time_step},{time.time()-t0:.4f}\n")
                except: pass
                
                export_time += (time.perf_counter() - t_export_start)

    total_time = time.time() - t0
    logger.info("Complete. Total: %.2fs (Solver: %.2fs)", total_time, solver_time)
    
    with open(os.path.join(config.outdir, "timing_sim.json"), "w") as f:
        json.dump({
            "total_simulation_time_sec": total_time,
            "solver_time_sec": solver_time,
            "export_time_sec": export_time
        }, f, indent=2)
        
    return config
