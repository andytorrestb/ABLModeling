# Silsoe Cube Case

This directory contains the configuration and setup for the Silsoe Cube validation case.

## Usage

To run this case using the ABLSim framework:

```bash
# From project root
python scripts/run_case.py cases/silsoe_cube --re 50000 --dry-run
```

## Structure
- `case.json`: Main configuration file defining physics, domain, and solver settings.
- `results/`: Output directory (gitignored).

## Validation Data
This case is designed to validate against experimental data for flow over a cube.
