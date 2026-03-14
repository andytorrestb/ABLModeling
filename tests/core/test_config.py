import unittest
import json
import os
import tempfile
from ablsim.core.config import load_config, SimulationConfig

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.valid_config = {
            "simulation": {
                "n_time_steps": 100,
                "output_interval": 10,
                "vel_video_fps": 10,
                "vort_video_fps": 10
            },
            "domain": {
                "maximal_velocity": 0.1,
                "initial_velocity": [0.1, 0, 0],
                "mult": 1
            },
            "physical": {
                "L_phy": 1.0,
                "nu_phy": 1e-5
            },
            "geometry": {
                "type": "cube"
            },
            "boundary_conditions": {
                "W": "inflow", "E": "outflow"
            }
        }

    def write_config(self, cfg_dict):
        fd, path = tempfile.mkstemp(suffix=".json", text=True)
        with os.fdopen(fd, 'w') as f:
            json.dump(cfg_dict, f)
        return path

    def test_load_valid_config(self):
        path = self.write_config(self.valid_config)
        try:
            cfg = load_config(path)
            self.assertEqual(cfg['simulation']['n_time_steps'], 100)
            self.assertEqual(cfg['geometry']['type'], 'cube')
        finally:
            os.remove(path)

    def test_simulation_config_class(self):
        # Mock load
        c = SimulationConfig(reynolds_number=1000, reference_length=10, outdir="test_out", cfg=self.valid_config)
        self.assertEqual(c.reference_length, 10)
        self.assertEqual(c.reynolds_number, 1000)
        # Check derived values
        # nu = (L * U) / Re
        expected_nu = (10 * 0.1) / 1000
        self.assertAlmostEqual(c.kinematic_viscosity, expected_nu)

if __name__ == '__main__':
    unittest.main()
