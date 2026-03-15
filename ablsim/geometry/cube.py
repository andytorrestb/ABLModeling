# ablsim/geometry/cube.py
from .base import GeometryBase

class CubeGeometry(GeometryBase):
    def __call__(self, x, y, z):
        """
        Placement logic for Silsoe Cube or generic block.
        Now supports 'wind_tunnel_scenario' for explicit placement.
        """
        # Check for wind tunnel scenario in the stored config dict
        cfg_dict = getattr(self.config, 'config_dict', {})
        wt = cfg_dict.get('wind_tunnel_scenario')
        
        if wt and 'placement' in wt:
            # -----------------------------------------------------------
            # Explicit Benchmark Placement
            # -----------------------------------------------------------
            N_H = self.config.reference_length # In this mode, this is N_H
            placement = wt['placement']
            
            # Upstream distance (from inlet to cube front face)
            upstream_H = placement.get('upstream_dist_H', 5.0)
            
            # Dimensions
            length_x = N_H
            length_y = N_H
            length_z = N_H
            
            # X Calculation
            # x=0 is inlet. Cube starts at upstream_H * H
            x_min = int(upstream_H * N_H)
            x_max = x_min + length_x
            
            # Y Calculation (Lateral)
            # Assuming centered
            W_tun_lu = self.config.domain_size[1]
            center_y = W_tun_lu // 2
            y_min = center_y - (length_y // 2)
            y_max = center_y + (length_y // 2)
            
            # Z Calculation (Vertical)
            if placement.get('grounded', True):
                z_min = 0
                z_max = length_z
            else:
                # Centered vertically or floating? defaulting to centered if not grounded
                H_tun_lu = self.config.domain_size[2]
                center_z = H_tun_lu // 2
                z_min = center_z - length_z // 2
                z_max = center_z + length_z // 2

        else:
            # -----------------------------------------------------------
            # Legacy/Generic Placement
            # -----------------------------------------------------------
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
