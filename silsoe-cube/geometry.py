from abc import ABC, abstractmethod

"""
Coordinate System Definition:
   Y (Vertical / Wall-normal)
   ^
   |    _______ (Cube)
   |   |       |
   |___|_______|______> X (Streamwise / Flow direction)
  /
 /
Z (Spanwise / Lateral)
"""

class GeometryFactory:
    @staticmethod
    def get_geometry(config):
        """
        Factory method to select geometry based on config.
        For now, defaults to 'cube' (the silsoe cube).
        Future extensions can switch on config.geometry['type'].
        """
        geo_type = config.geometry.get('type', 'cube')
        
        if geo_type == 'cube':
            return CubeGeometry(config)
        else:
            raise ValueError(f"Unknown geometry type: {geo_type}")

class GeometryBase(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def __call__(self, x, y, z):
        pass

class CubeGeometry(GeometryBase):
    def __call__(self, x, y, z):
        # Original logic:
        # mid = (int(0.15 * config.domain_size[0]), config.domain_size[1] // 2, 0)
        # half_size = (config.reference_length // 2, config.reference_length // 2, config.reference_length // 2)
        # box_z = (0 <= z) & (z < config.reference_length)
        
        # Parametrized logic using config.geometry if available, else defaults matching legacy code
        
        ref = self.config.reference_length
        dom = self.config.domain_size
        
        # Default placement logic from legacy set_ship
        default_x_rel = 0.15
        default_width_rel = 1.0  # relative to L
        default_height_rel = 1.0 # relative to L
        
        # Override from config if present
        # We expect config.geometry to be a dict
        geo = self.config.geometry
        
        x_start_factor = geo.get('x_position_factor', default_x_rel) 
        
        # Cube dimensions
        length_x = ref * geo.get('length_x_factor', default_width_rel)
        length_y = ref * geo.get('length_y_factor', default_width_rel)
        length_z = ref * geo.get('length_z_factor', default_height_rel)

        # Center position
        # X: 0.15 * DomainX
        # Y: 0.5 * DomainY
        # Z: 0 (bottom aligned)
        
        center_x = int(x_start_factor * dom[0])
        center_y = int(dom[1] // 2)
        
        # Define ranges
        # X: centered at center_x
        x_min = center_x - length_x / 2
        x_max = center_x + length_x / 2
        
        # Y: centered at center_y
        y_min = center_y - length_y / 2
        y_max = center_y + length_y / 2
        
        # Z: 0 to height (typically sitting on floor)
        # Original code: 0 <= z < ref
        z_min = 0
        z_max = length_z
        
        mask_x = (x >= x_min) & (x < x_max)
        mask_y = (y >= y_min) & (y < y_max)
        mask_z = (z >= z_min) & (z < z_max)
        
        return mask_x & mask_y & mask_z
