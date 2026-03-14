# ablsim/geometry/cube.py
from .base import GeometryBase

class CubeGeometry(GeometryBase):
    def __call__(self, x, y, z):
        """
        Default placement logic for Silsoe Cube or generic block.
        """
        ref = self.config.reference_length
        dom = self.config.domain_size
        
        # Default defaults
        default_x_rel = 0.15
        default_width_rel = 1.0
        default_height_rel = 1.0
        
        geo = self.config.geometry if self.config.geometry else {}
        
        # Factor from config or default
        x_start_factor = geo.get('x_position_factor', default_x_rel)
        lx_factor = geo.get('length_x_factor', default_width_rel)
        ly_factor = geo.get('length_y_factor', default_width_rel)
        lz_factor = geo.get('length_z_factor', default_height_rel)

        # Dimensions in LU
        length_x = ref * lx_factor
        length_y = ref * ly_factor
        length_z = ref * lz_factor

        # Center X
        center_x = int(x_start_factor * dom[0])
        x_min = center_x - length_x / 2
        x_max = center_x + length_x / 2
        
        # Center Y (Width of tunnel)
        center_y = int(dom[1] // 2)
        y_min = center_y - length_y / 2
        y_max = center_y + length_y / 2
        
        # Z (Height from floor)
        z_min = 0
        z_max = length_z
        
        mask_x = (x >= x_min) & (x < x_max)
        mask_y = (y >= y_min) & (y < y_max)
        mask_z = (z >= z_min) & (z < z_max)
        
        return mask_x & mask_y & mask_z
