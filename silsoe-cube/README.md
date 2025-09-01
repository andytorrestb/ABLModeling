# ship-wake/run.py — Quick README

Purpose
- Run orchestrator for ship-wake LBM simulations. It reads a JSON config and executes combinations of Reynolds numbers (Re) and reference lengths (L).

Quick start (Windows)
- From the project root run:
  - cd ship-wake
  - python3 run.py --config config.json
- Examples:
  - Run a single Re from CLI: python ship-wake\run.py --config config.json --re 10000
  - Run a single reference length from CLI: python ship-wake\run.py --config config.json --ref 5
  - Dry-run (validate plan without executing): python ship-wake\run.py --config config.json --dry-run

Notes about CLI arguments
- The CLI arguments in this tool are lightweight placeholders:
  - They can override lists provided in the config (e.g., --re or --ref).
  - Primary case definitions should be placed in the config file (see below).
  - Use --dry-run to inspect planned cases without running simulations.

Config file (config.json)
- The script expects a JSON config by default (default path: `config.json`).
- Key sections used by run.py:
  - "simulation": holds simulation-specific lists and parameters (e.g., `reynolds_list`, `ref_length_list`).
  - "physical": physical parameters used to compute derived values (example keys: `L_phy`, `nu_phy`).
  - "domain": domain/grid settings for simulations.
- Minimal example structure:
```json
{
  "simulation": {
    "reynolds_list": [1000, 5000],
    "ref_length_list": [2, 4]
  },
  "physical": {
    "L_phy": 1.0,
    "nu_phy": 1e-6
  },
  "domain": {
    "nx": 200,
    "ny": 100,
    "nz": 50
  }
}
```

Output & logs
- For each case the script creates an output directory named like:
  - `output_Re_<Re>_L_<L>` (e.g. `output_Re_1000_L_2`)
- A per-case file logger is attached so logs for each run are written in that case's output directory.

Troubleshooting
- If the config path is not found, run.py exits with an error: ensure `--config` points to an existing JSON file.
- If you see missing or empty `reynolds_list` / `ref_length_list`, add them to the "simulation" block or pass `--re` / `--ref` on the CLI.

If further details are needed (example config, expected outputs, or how logging is configured), provide the config.json you plan to


## Memory profiling (mem_profile)
There are two simple ways to inspect memory use while running run.py:

1) Time-series (process-level) with mprof
- Install (Windows):
  - pip install memory_profiler psutil
- Record a run and view peak/plot:
  - mprof run --python python run.py --config config.json
  - mprof peak        # show peak memory from the last run
  - mprof plot        # open a plot of memory over time

2) Line-by-line with memory_profiler
- Install:
  - pip install memory_profiler
- Mark functions you want profiled with the @profile decorator, then run:
  - python -m memory_profiler run.py --config config.json
- Output shows per-line memory usage for decorated functions.

Notes
- On Windows, installing psutil improves accuracy for mprof.
- Use the time-series approach to find when peaks occur, then add @profile to hot functions to inspect line-level usage.