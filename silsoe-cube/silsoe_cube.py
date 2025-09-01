# Load configuration
# import json
# from logging import config
# from venv import logger
from lbmpy.session import *
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

from plotting import *

# def load_config(path):
#     """Load and validate required keys from JSON config file."""
#     try:
#         with open(path, 'r', encoding='utf-8') as f:
#             cfg = json.load(f)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Configuration file '{path}' not found. Please create it.")
#     except json.JSONDecodeError as e:
#         raise ValueError(f"Invalid JSON in '{path}': {e}")

#     # Minimal required sections and keys - extend as needed
#     required_sections = {
#         'simulation': ['n_time_steps', 'output_interval', 'vel_video_fps', 'vort_video_fps', 'reynolds_list', 'ref_length_list'],
#         'domain': ['maximal_velocity', 'initial_velocity', 'mult'],
#         'physical': ['L_phy', 'nu_phy']
#     }

#     for section, keys in required_sections.items():
#         if section not in cfg:
#             raise KeyError(f"Config missing required section '{section}'.")
#         for k in keys:
#             if k not in cfg[section]:
#                 raise KeyError(f"Config section '{section}' missing required key '{k}'.")

#     # Basic type checks (lightweight)
#     if not isinstance(cfg['domain']['initial_velocity'], (list, tuple)) or len(cfg['domain']['initial_velocity']) != 3:
#         raise TypeError("domain.initial_velocity must be a list/tuple of length 3.")

#     return cfg

# class SimulationConfig:
#     def __init__(self, reynolds_number, reference_length, outdir, cfg):
#         # cfg is the config dict loaded by load_config
#         sim = cfg['simulation']
#         dom = cfg['domain']

#         self.reference_length = reference_length
#         self.maximal_velocity = dom['maximal_velocity']
#         self.reynolds_number = reynolds_number
#         self.kinematic_viscosity = (self.reference_length * self.maximal_velocity) / self.reynolds_number
#         self.initial_velocity = tuple(dom['initial_velocity'])

#         self.mult = dom['mult']
#         self.domain_size = (
#             int(self.reference_length * 5 * self.mult),
#             int(self.reference_length * self.mult),
#             int(self.reference_length * 0.5 * self.mult)
#         )
#         self.dim = len(self.domain_size)

#         self.n_time_steps = sim['n_time_steps']
#         self.output_interval = sim['output_interval']
#         self.vel_video_fps = sim['vel_video_fps']
#         self.vort_video_fps = sim['vort_video_fps']

#         self.outdir = outdir
#         os.makedirs(self.outdir, exist_ok=True)

# Update set_ship to accept config
# # filepath: c:\Users\andy9\Dev\LBM-BEM\ship-wake\shipwake_3D.py
# def set_ship(x, y, z, config, *_):
#     geo = config_data['geometry']
#     mid = (int(0.15 * config.domain_size[0]), config.domain_size[1] // 2, 0)
#     half_size = (config.reference_length // 2, config.reference_length // 2, config.reference_length // 2)
    
#     # Main hull of ship
#     ship_hull_x = (mid[0] <= x) & (x < mid[0] + geo['hull_multiplier'] * config.reference_length)
# #     # ... (similarly update other sections with geo values)
    
#     return ship_hull | ship_back | ship_nose | box

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


def set_ship(x, y, z, config, *_):
    mid = (int(0.15 * config.domain_size[0]), config.domain_size[1] // 2, 0)
    half_size = (config.reference_length // 2, config.reference_length // 2, config.reference_length // 2)

    # Main hull of ship.
    box_x = (mid[0] - half_size[0] <= x) & (x < mid[0] + half_size[0])
    box_y = (mid[1] - half_size[1] <= y) & (y < mid[1] + half_size[1])
    box_z = (0 <= z) & (z < config.reference_length)
    box = box_x & box_y & box_z

    return box


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


def timeloop(timeSteps, bh, dh, kernel):
    for i in range(timeSteps):
        bh()
        dh.run_kernel(kernel)
        dh.swap("src", "dst")

# @profile
def run_simulation(reynolds_number, ref_length, cfg):

    # Step 1) Create output directory for specific configuration
    outdir = f"output_Re_{int(reynolds_number):d}_L_{ref_length:d}"
    config = SimulationConfig(reynolds_number, ref_length, outdir, cfg)
    print(f"Running simulation with Re={reynolds_number:.0e}, L={ref_length}, nu={config.kinematic_viscosity:.2e}, U={config.maximal_velocity}")

    # Step 2) Defensive memory check
    # Estimate memory: domain_size * arrays * dtype (float64=8 bytes)
    domain_size = config.domain_size
    n_cells = domain_size[0] * domain_size[1] * domain_size[2]
    n_arrays = 3  # src, dst, velField
    bytes_per_cell = 8  # float64
    estimated_mem_gb = n_cells * n_arrays * bytes_per_cell / (1024**3)
    max_mem_gb = 1  # Set your max allowed RAM (GB)
    if estimated_mem_gb > max_mem_gb:
        print(f"Skipping simulation for L={ref_length}, Re={reynolds_number:.0e}: Estimated memory {estimated_mem_gb:.2f} GB exceeds limit of {max_mem_gb} GB.")
        return
    else:
        print(f"Memory check passed for L={ref_length}, Re={reynolds_number:.0e}: {estimated_mem_gb:.2f} GB used.")

    # Step 3) Define LBM lattice structure to use
    stencil = LBStencil(Stencil.D3Q27)
    
    
    # Step 4) : Initialize data array for the simulation.
    with MemProbe("create_data_handling"):
        dh = ps.create_data_handling(domain_size=domain_size, periodicity=(False, False))

    with MemProbe("add_arrays_and_fill"):
        src = dh.add_array('src', values_per_cell=len(stencil), alignment=True)
        dh.fill('src', 0.0, ghost_layers=True)
        dst = dh.add_array('dst', values_per_cell=len(stencil), alignment=True)
        dh.fill('dst', 0.0, ghost_layers=True)
        velField = dh.add_array('velField', values_per_cell=dh.dim, alignment=True)
        dh.fill('velField', 0.0, ghost_layers=True)

    try:
        logger.debug("Approx src nvytes: %s", getattr(src, 'nbytes', 'n/a'))
    except Exception:
        pass

    # Step 5) Define LBM models and parameters.
    with MemProbe("create_lb_model"):
        omega = relaxation_rate_from_lattice_viscosity(config.kinematic_viscosity)
        dim = config.dim
        lbm_config = LBMConfig(stencil=Stencil.D3Q27, method=Method.CUMULANT, relaxation_rate=omega,
                            compressible=True,
                            output={'velocity': velField}, kernel_type='stream_pull_collide')

        method = create_lb_method(lbm_config=lbm_config)
    # print(method)  

    init = pdf_initialization_assignments(method, 1.0, config.initial_velocity, src.center_vector)

    with MemProbe("kernel_init_compile_and_run"):
        ast_init = ps.create_kernel(init, target=dh.default_target)
        kernel_init = ast_init.compile()
        dh.run_kernel(kernel_init)

    lbm_optimisation = LBMOptimisation(symbolic_field=src, symbolic_temporary_field=dst)
    update = create_lb_update_rule(lb_method=method,
                                lbm_config=lbm_config,
                                lbm_optimisation=lbm_optimisation)

    with MemProbe("kernel_update_compile_and_run"):
        ast_kernel = ps.create_kernel(update, target=dh.default_target, cpu_openmp=True)
        kernel = ast_kernel.compile()

    # Step 6) Set Boundary Conditions
    bh = LatticeBoltzmannBoundaryHandling(method, dh, 'src', name="bh")

    inflow = UBB(config.initial_velocity)
    outflow = ExtrapolationOutflow(stencil[4], method)
    wall = NoSlip("wall")

    bh.set_boundary(inflow, slice_from_direction('W', dim))
    bh.set_boundary(outflow, slice_from_direction('E', dim))
    for direction in ('N', 'S'):
        bh.set_boundary(wall, slice_from_direction(direction, dim))

    bh.set_boundary(NoSlip("obstacle"), mask_callback=lambda x, y, z, *_: set_ship(x, y, z, config))

    # Step 7) Run the simulation
    # print(f'Running simulation for Re={reynolds_number:.0e}, output to {config.outdir}')
    t0 = time.time()
    logger.info("Running simulation for Re=%s, output to %s", reynolds_number, config.outdir)
    with MemProbe("main_simulation_loop"):
        for step in range(1, config.n_time_steps + 1):
            timeloop(1, bh, dh, kernel)
            if step % config.output_interval == 0:
                # print(f'Timestep {step}')
                plot_velocity(dh.gather_array('velField'), domain_size, step, config)
                plot_vorticity_frame(dh.gather_array('velField'), domain_size, step, config)

    total_time = time.time() - t0
    logger.info("Simulation completed in %.2fs", total_time)
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
            save_video(sim_config.outdir + '/vel_magnitude_output' , video_filename="vel_magnitude.mp4", fps=sim_config.vel_video_fps)
            save_video(sim_config.outdir + '/vorticity_output' , video_filename="vorticity.mp4", fps=sim_config.vort_video_fps)

