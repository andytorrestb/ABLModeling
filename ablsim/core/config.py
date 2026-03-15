import json
import os
from dataclasses import dataclass

def load_config(path):
    """Load and validate required keys from JSON config file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{path}' not found. Please create it.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in '{path}': {e}")

    required_sections = {
        'simulation': ['n_time_steps', 'output_interval', 'vel_video_fps', 'vort_video_fps'],
        'domain': ['maximal_velocity', 'initial_velocity', 'mult'],
        'physical': ['L_phy', 'nu_phy'],
        'geometry': [],
        'boundary_conditions': ['W', 'E', 'N', 'S', 'T', 'B'] 
    }

    for section, keys in required_sections.items():
        if section not in cfg:
            # Tolerant read for now, but log warning in future
            continue 
        for k in keys:
            if k not in cfg[section]:
                raise KeyError(f"Config section '{section}' missing required key '{k}'.")

    if 'domain' in cfg and 'initial_velocity' in cfg['domain']:
        if not isinstance(cfg['domain']['initial_velocity'], (list, tuple)) or len(cfg['domain']['initial_velocity']) != 3:
            raise TypeError("domain.initial_velocity must be a list/tuple of length 3.")

    return cfg


def deep_update(base_dict, update_dict):
    """
    Recursively update a dictionary.
    """
    import copy
    merged = copy.deepcopy(base_dict)
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in merged and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


class SimulationConfig:
    def __init__(self, reynolds_number, reference_length, outdir, cfg):
        sim = cfg.get('simulation', {})
        dom = cfg.get('domain', {})
        geo = cfg.get('geometry', {})
        wt = cfg.get('wind_tunnel_scenario', None)

        self.config_dict = cfg # Store full config for geometry/other access

        # ------------------------------------------------------------------------------------------
        # CONFIGURATION PATH A: WIND TUNNEL BENCHMARK (Physical explicit sizing)
        # ------------------------------------------------------------------------------------------
        if wt:
            # In this mode, 'reference_length' is interpreted as "Cells per Cube Height (N_H)"
            self.N_H = int(reference_length)
            
            # Physical dimensions
            self.cube_height_m = wt['cube_height_mm'] / 1000.0
            L_tun, W_tun, H_tun = wt['tunnel_dims_m']
            
            # Calculate Domain in Lattice Units
            # Aspect ratios from physical dimensions * N_H
            nx = int((L_tun / self.cube_height_m) * self.N_H)
            ny = int((W_tun / self.cube_height_m) * self.N_H)
            nz = int((H_tun / self.cube_height_m) * self.N_H)
            
            self.domain_size = (nx, ny, nz)
            self.dim = 3
            self.reference_length = self.N_H  # For LBM code, ref length is characteristic length (cube height)
            
            # Velocity / Viscosity Scaling
            # We fix the Lattice Velocity (u_lb) usually to a small compressible limit (e.g. 0.05 - 0.1)
            # Re = (u_lb * N_H) / nu_lb  =>  nu_lb = (u_lb * N_H) / Re
            
            # Using specific cube-height velocity if specified, or generic max
            physical_u_ref = wt.get('cube_height_velocity_ms', 5.0)
            
            # Determine lattice maximal velocity (u_lb)
            # If specified in domain, use it, else default to 0.05
            self.maximal_velocity = dom.get('maximal_velocity', 0.05)
            self.initial_velocity = (self.maximal_velocity, 0.0, 0.0) # Start with uniform flow approx
            
            self.reynolds_number = reynolds_number
            self.kinematic_viscosity = (self.maximal_velocity * self.N_H) / self.reynolds_number
            
            # Store physical scaling factors
            self.dx_phy = self.cube_height_m / self.N_H  # meters per grid cell
            z0_mm = wt.get('roughness_z0_mm', 0.05)
            self.z0_lattice = (z0_mm / 1000.0) / self.dx_phy
            
            self.mult = 1 # Not used in this mode but kept for compat
            
        # ------------------------------------------------------------------------------------------
        # CONFIGURATION PATH B: GENERIC DOMAIN (Abstract multiplier sizing)
        # ------------------------------------------------------------------------------------------
        else:
            # Basic physical and domain parameters
            self.reference_length = reference_length  # L_lattice (resolution)
            self.maximal_velocity = dom.get('maximal_velocity', 0.1)  # U_lattice
            
            self.reynolds_number = reynolds_number
            # nu_lattice = (U_lattice * L_lattice) / Re
            self.kinematic_viscosity = (self.reference_length * self.maximal_velocity) / self.reynolds_number
            
            init_vel = dom.get('initial_velocity', [0.1, 0.0, 0.0])
            self.initial_velocity = tuple(init_vel)  # [ux, uy, uz] in Lattice Units

            self.mult = dom.get('mult', 5)
            self.domain_size = (
                int(self.reference_length * 5 * self.mult),
                int(self.reference_length * self.mult),
                int(self.reference_length * self.mult)
            )
            self.dim = len(self.domain_size)

        self.geometry = geo
        self.boundary_conditions = cfg.get('boundary_conditions', {})


        # Simulation controls
        self.n_time_steps = sim.get('n_time_steps', 100)
        self.output_interval = sim.get('output_interval', 10)
        self.vel_video_fps = sim.get('vel_video_fps', 20)
        self.vort_video_fps = sim.get('vort_video_fps', 20)
        
        # Post-processing settings
        self.run_post_processing = sim.get('run_post_processing', False)
        self.generate_video = sim.get('generate_video', True)

        # Optional compute settings
        self.kernel_target = sim.get('kernel_target')
        self.cpu_openmp = bool(sim.get('cpu_openmp', False))
        self.gpu_device = sim.get('gpu_device', None)

        # Output directory
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
