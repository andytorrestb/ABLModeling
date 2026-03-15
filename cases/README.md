# Simulation Cases

This directory contains simulation cases. Each subdirectory represents a self-contained case.

## Structure of a Case Folder

A valid case folder (e.g., `cases/my_new_case/`) must contain:
1.  **`case.json`**: The main configuration file.
2.  (Optional) **Geometry Files**: e.g., `.stl` or custom Python geometry definitions.
3.  (Optional) **Readme**: Describes the physics/setup.

## `case.json` Format

The `case.json` file is the source of truth. It supports:
- **Base Configuration**: Defines physical parameters, domain size, and default simulation settings.
- **Profiles**: Named overrides for specific run modes (e.g., `smoke`, `production`).

Example:
```json
{
  "simulation": {
    "reynolds_list": [100000],
    "ref_length_list": [11],
    "n_time_steps": 1000
  },
  "profiles": {
    "smoke": {
      "simulation": {
        "n_time_steps": 10
      }
    }
  },
  "physical": { ... },
  "domain": { ... }
}
```

## Running Cases

Use the CLI tool `scripts/run_case.py`:

```bash
# List all available cases
python scripts/run_case.py --list

# Run a specific case (default profile)
python scripts/run_case.py my_new_case

# Run with a specific profile (e.g., smoke test)
python scripts/run_case.py my_new_case --profile smoke

# Inspect case configuration
python scripts/run_case.py my_new_case --info
```
