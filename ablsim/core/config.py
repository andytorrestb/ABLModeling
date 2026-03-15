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

        # Basic physical and domain parameters
        self.reference_length = reference_length  # L_lattice (resolution)
        self.maximal_velocity = dom.get('maximal_velocity', 0.1)  # U_lattice
        self.geometry = geo
        self.boundary_conditions = cfg.get('boundary_conditions', {})

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
