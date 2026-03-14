from pathlib import Path
from .config import load_config

class Case:
    def __init__(self, case_dir):
        self.path = Path(case_dir).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Case directory not found: {self.path}")
        
        self.name = self.path.name
        self.config_path = self.path / "case.json"
        
        # Load config logic
        self.config = load_config(str(self.config_path))
    
    def get_output_dir(self, re, L):
        """
        Standardized output directory structure:
        results/Re_{re}_L_{L}
        """
        return self.path / "results" / f"Re_{int(re)}_L_{int(L)}"

    def get_overridden_config(self, overrides=None):
        """
        Return config dict with overrides applied.
        """
        import copy
        cfg = copy.deepcopy(self.config)
        
        if overrides:
            for k, v in overrides.items():
                # Support dot notation for nested keys? e.g. "simulation.n_time_steps"
                keys = k.split('.')
                d = cfg
                for key in keys[:-1]:
                    d = d.setdefault(key, {})
                d[keys[-1]] = v
        return cfg
