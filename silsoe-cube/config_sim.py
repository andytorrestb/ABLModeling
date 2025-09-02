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
        'simulation': ['n_time_steps', 'output_interval', 'vel_video_fps', 'vort_video_fps', 'reynolds_list', 'ref_length_list'],
        'domain': ['maximal_velocity', 'initial_velocity', 'mult'],
        'physical': ['L_phy', 'nu_phy']
    }

    for section, keys in required_sections.items():
        if section not in cfg:
            raise KeyError(f"Config missing required section '{section}'.")
        for k in keys:
            if k not in cfg[section]:
                raise KeyError(f"Config section '{section}' missing required key '{k}'.")

    if not isinstance(cfg['domain']['initial_velocity'], (list, tuple)) or len(cfg['domain']['initial_velocity']) != 3:
        raise TypeError("domain.initial_velocity must be a list/tuple of length 3.")

    return cfg

# @dataclass
# class SimulationConfig:
#     reynolds_number: float
#     reference_length: int
#     outdir: str
#     cfg: dict

#     @classmethod
#     def from_cfg(cls, re, L, cfg, outdir):
#         outdir = outdir or f"output_Re_{int(re)}_L_{L}"
#         os.makedirs(outdir, exist_ok=True)
#         return cls(re, L, outdir, cfg)
    

class SimulationConfig:
    def __init__(self, reynolds_number, reference_length, outdir, cfg):
        sim = cfg['simulation']
        dom = cfg['domain']

        # Basic physical and domain parameters
        self.reference_length = reference_length
        self.maximal_velocity = dom['maximal_velocity']
        self.reynolds_number = reynolds_number
        self.kinematic_viscosity = (self.reference_length * self.maximal_velocity) / self.reynolds_number
        self.initial_velocity = tuple(dom['initial_velocity'])

        self.mult = dom['mult']
        self.domain_size = (
            int(self.reference_length * 5 * self.mult),
            int(self.reference_length * self.mult),
            int(self.reference_length * self.mult)
        )
        self.dim = len(self.domain_size)

        # Simulation controls
        self.n_time_steps = sim['n_time_steps']
        self.output_interval = sim['output_interval']
        self.vel_video_fps = sim['vel_video_fps']
        self.vort_video_fps = sim['vort_video_fps']

        # Optional compute settings (all optional)
        # simulation.kernel_target: "cpu" | "gpu" | "default"
        # simulation.cpu_openmp: boolean (CPU only)
        # simulation.gpu_device: integer device index
        self.kernel_target = sim.get('kernel_target')
        self.cpu_openmp = bool(sim.get('cpu_openmp', False))
        self.gpu_device = sim.get('gpu_device', None)

        # Output directory
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)