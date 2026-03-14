import sys
import os
import unittest
import json
import tempfile

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'silsoe-cube'))

from config_sim import load_config, SimulationConfig

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.valid_config = {
            "simulation": {
                "n_time_steps": 100,
                "output_interval": 10,
                "reynolds_list": [100],
                "ref_length_list": [10],
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

    def test_missing_section(self):
        invalid = self.valid_config.copy()
        del invalid['geometry']  # Should not fail if validation allows empty, but check impl
        # In my refactor, 'geometry' IS required in required_sections, just keys list is empty
        
        path = self.write_config(invalid)
        try:
            with self.assertRaises(KeyError):
                load_config(path)
        finally:
            os.remove(path)

    def test_simulation_config_object(self):
        # Test the class wrapper
        cfg = self.valid_config
        sim_conf = SimulationConfig(1000, 20, "test_out", cfg)
        
        self.assertEqual(sim_conf.reynolds_number, 1000)
        self.assertEqual(sim_conf.reference_length, 20)
        self.assertEqual(sim_conf.maximal_velocity, 0.1)
        # Check derived viscosity
        # nu = (L * U) / Re = (20 * 0.1) / 1000 = 2 / 1000 = 0.002
        self.assertAlmostEqual(sim_conf.kinematic_viscosity, 0.002)

if __name__ == '__main__':
    unittest.main()
