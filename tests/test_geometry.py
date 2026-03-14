import sys
import os
import numpy as np

# Adjust path to import from silsoe-cube folder
sys.path.append(os.path.join(os.getcwd(), 'silsoe-cube'))

from geometry import GeometryFactory
from config_sim import SimulationConfig

# Mock configuration object
class MockConfig:
    def __init__(self):
        self.output_interval = 10
        self.vel_video_fps = 10
        self.vort_video_fps = 10
        self.reynolds_number = 100
        self.reference_length = 10
        self.maximal_velocity = 0.1
        self.kinematic_viscosity = 0.01
        self.initial_velocity = (0.1, 0.0, 0.0)
        self.domain_size = (50, 20, 10) # 50x20x10 grid
        self.geometry = {
            'type': 'cube',
            'x_position_factor': 0.2, # Center at x=10
            'length_x_factor': 1.0,   # Width 10
            'length_y_factor': 1.0,   # Depth 10
            'length_z_factor': 1.0    # Height 10
        }

def test_cube_geometry():
    cfg = MockConfig()
    
    # Create coordinate grids
    # lbmpy/numpy meshgrid usually requires index='ij' for 3D array matching
    x = np.arange(cfg.domain_size[0]).reshape(-1, 1, 1)
    y = np.arange(cfg.domain_size[1]).reshape(1, -1, 1)
    z = np.arange(cfg.domain_size[2]).reshape(1, 1, -1)
    
    # Instantiate geometry
    geo_func = GeometryFactory.get_geometry(cfg)
    mask = geo_func(x, y, z)
    
    # Expected bounds
    # Ref L = 10
    # Center X = 0.2 * 50 = 10.  Range [5, 15]
    # Center Y = 20 // 2 = 10.   Range [5, 15]
    # Z starts at 0. Height 10.  Range [0, 10]
    
    print(f"Mask shape: {mask.shape}")
    print(f"Total cells marked as obstacle: {np.sum(mask)}")
    
    # Check a point inside the cube (e.g., 10, 10, 5)
    inside = mask[10, 10, 5]
    print(f"Point (10, 10, 5) is obstacle? {inside} (Expected: True)")
    
    # Check a point outside (e.g., 40, 10, 5)
    outside = mask[40, 10, 5]
    print(f"Point (40, 10, 5) is obstacle? {outside} (Expected: False)")
    
    if inside and not outside and np.sum(mask) > 0:
        print("✅ Geometry test PASSED")
    else:
        print("❌ Geometry test FAILED")

if __name__ == "__main__":
    test_cube_geometry()
