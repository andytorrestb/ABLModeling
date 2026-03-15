import sys
import os
sys.path.append(os.getcwd())

from ablsim.core.case import Case
from ablsim.core.config import SimulationConfig

def test_silsoe_config():
    case_dir = "cases/silsoe_cube"
    c = Case(case_dir)
    print(f"Loaded case: {c.name}")
    
    # Load default profile
    # Case.init loads base_config. 
    # We need to mimic what runner does to create SimulationConfig
    
    # Replicate runner logic roughly
    cfg = c.config
    
    # Check if 'wind_tunnel_scenario' is present
    if 'wind_tunnel_scenario' in cfg:
        print("Wind tunnel scenario found.")
        wt = cfg['wind_tunnel_scenario']
        print(f"Cube height: {wt['cube_height_mm']} mm")
        print(f"Tunnel dims: {wt['tunnel_dims_m']} m")
    else:
        print("ERROR: Wind tunnel scenario NOT found.")
        return

    # Create SimulationConfig
    # We use some dummy values for reynolds and ref_length from the config itself to test
    re_list = cfg['simulation']['reynolds_list']
    ref_list = cfg['simulation']['ref_length_list']
    
    re = re_list[0]
    ref = ref_list[0] # This should be N_H (cells per height) e.g. 32
    
    print(f"Testing with Re={re}, N_H={ref}")
    
    sim_config = SimulationConfig(re, ref, "test_out", cfg)
    
    print(f"Domain Size (LU): {sim_config.domain_size}")
    print(f"Dims: {sim_config.dim}")
    print(f"Max Velocity (LU): {sim_config.maximal_velocity}")
    print(f"Viscosity (LU): {sim_config.kinematic_viscosity}")
    print(f"Geometry Config: {sim_config.geometry}")
    
    # Check Geometry placement
    from ablsim.geometry.cube import CubeGeometry
    geo = CubeGeometry(sim_config)
    
    # Create a small grid to probe or just check internal logic if possible
    # We can't really call geo(x,y,z) without arrays, but we can check if it initializes
    print("Geometry component initialized successfully.")
    
    # Check domain proportions
    # Tunnel is 7.5 x 1.85 x 1.1
    # Ratio L/H = 7.5/0.15 = 50
    # Ratio W/H = 1.85/0.15 = 12.33
    # Ratio H/H = 1.1/0.15 = 7.33
    
    nx, ny, nz = sim_config.domain_size
    print(f"Nx/Ref: {nx/ref:.2f} (Target 50.0)")
    print(f"Ny/Ref: {ny/ref:.2f} (Target 12.33)")
    print(f"Nz/Ref: {nz/ref:.2f} (Target 7.33)")

if __name__ == "__main__":
    test_silsoe_config()
