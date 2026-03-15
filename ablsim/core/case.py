import json
from pathlib import Path
from .config import load_config

def deep_update(base, override):
    """
    Recursively update a dictionary.
    """
    import copy
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in merged and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged

class Case:
    """
    Represents a simulation case with configuration, assets, and results.
    """
    def __init__(self, case_dir):
        self.path = Path(case_dir).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Case directory not found: {self.path}")
        
        self.name = self.path.name
        self.config_path = self.path / "case.json"
        
        # Load base configuration
        if not self.config_path.exists():
             raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            self.base_config = load_config(str(self.config_path))
        except Exception as e:
            # We allow partial loading for listing purposes but validation will catch it
            # For now, let's just fail if main config is bad
            raise ValueError(f"Failed to load case configuration for '{self.name}': {e}")

        # Profiles defined in the case.json or separate profiles.json (future extension)
        self.profiles = self.base_config.get('profiles', {})
        
        # Current active config starts as base
        self.config = self.base_config
        self.active_profile_name = None

    def list_profiles(self):
        """Return a list of available execution profiles."""
        return list(self.profiles.keys())

    def apply_profile(self, profile_name):
        """
        Apply a named profile to the current configuration.
        """
        if profile_name not in self.profiles:
             # Basic handling if profile doesn't exist? raise for now
             if profile_name == "default":
                 self.config = self.base_config
                 self.active_profile_name = None
                 return self.config
             raise ValueError(f"Profile '{profile_name}' not found in case '{self.name}'. Available: {list(self.profiles.keys())}")
        
        override = self.profiles[profile_name]
        self.config = deep_update(self.base_config, override)
        self.active_profile_name = profile_name
        return self.config

    def get_output_dir(self, re, L):
        """
        Standardized output directory structure:
        results/[profile_name]/Re_{re}_L_{L}
        or
        results/Re_{re}_L_{L} (if no profile or default)
        """
        results_dir = self.path / "results"
        if self.active_profile_name:
            # Create a localized profile subdirectory
            results_dir = results_dir / self.active_profile_name
            
        return results_dir / f"Re_{int(re)}_L_{int(L)}"

    def validate(self):
        """
        Check if the case structure and config are valid.
        Returns (valid: bool, issues: list)
        """
        issues = []
        if not self.config_path.exists():
            issues.append("Missing case.json")
        
        if not self.base_config:
             issues.append("Config failed to load")
             return False, issues

        # Check for geometry assets if defined
        geo = self.base_config.get('geometry', {})
        if 'file' in geo:
            geo_file = self.path / geo['file']
            if not geo_file.exists():
                issues.append(f"Missing geometry file: {geo['file']}")

        # Basic config check
        try:
            load_config(str(self.config_path))
        except Exception as e:
            issues.append(f"Config validation failed: {str(e)}")

        return len(issues) == 0, issues

    @staticmethod
    def find_cases(root_dir):
        """
        Recursively find all valid case directories under root_dir.
        A case directory must contain a 'case.json'.
        """
        root = Path(root_dir)
        cases = []
        # Searching specifically for case.json ensures we match our format
        for path in root.rglob("case.json"):
            # Check if parent is already a case (avoid nested matches if needed or just use parent)
            # Assuming simple structure
            try:
                c = Case(path.parent)
                cases.append(c)
            except Exception:
                pass 
        cases.sort(key=lambda c: c.name)
        return cases

    def describe(self):
        """Return a string description of the case."""
        desc = [f"Case: {self.name}"]
        desc.append(f"Path: {self.path}")
        desc.append(f"Profiles: {', '.join(self.list_profiles())}")
        re_list = self.base_config.get('simulation', {}).get('reynolds_list', [])
        desc.append(f"Default Re: {re_list}")
        return "\n".join(desc)

