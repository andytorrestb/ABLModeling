# Silsoe Cube Benchmark Case

This case replicates the Silsoe 6m cube experiment in a wind tunnel setup, focusing on the developed boundary layer flow around a surface-mounted cube.

## Benchmark Parameters

The current configuration is set up as a **Wind Tunnel Validation** case:
- **Cube Size**: 150 mm (model scale), representing the 6m full-scale cube.
- **Tunnel Dimensions**: 1.1 m (H) x 1.85 m (W) x 7.5 m (L)
- **Fluid**: Air (incompressible approximation via LBM)
- **Reference Velocity**: ~6.4 m/s at cube height (H)
- **Inflow**: Log-law boundary layer profile with $z_0 \approx 0.42$ mm.

## Running the Case

To run the standard validation profile:
```bash
python scripts/run_case.py cases/silsoe_cube --profile validation
```

To run a quick debug/smoke test:
```bash
python scripts/run_case.py cases/silsoe_cube --profile smoke
```

## Profiles

| Profile          | N_H    | Memory (Est.) | Description                     |
|------------------|--------|---------------|---------------------------------|
| `smoke`          | 8      | ~1 GB         | Fast debug check                |
| `validation`     | 24     | ~26 GB        | Desktop validation (Standard)   |
| `hpc_validation` | 48     | ~200 GB       | High-fidelity matching exp.     |
| default          | 20     | ~16 GB        | Balanced setup                  |

## Outputs

Results are stored in `results/[profile]/Re_50000_L_[N_H]/`.

Key outputs include:
- `slice_data/*.npz`: Instantaneous velocity slices.
- `validation_metrics.json`: Time-averaged vertical profiles at key locations (Inlet, Upstream, Cube Center, Wake).
- `vel_magnitude.mp4`: Visualization of the flow field.

## Setup Details

The simulation uses:
- **LBM Solver**: Cumulant collision operator (D3Q27).
- **BCs**: No-slip floor, log-law inlet, extrapolated outlet.
- **Scaling**: Dimensionless lattice units derived strictly from the physical wind tunnel dimensions and the target resolution ($N_H$).

