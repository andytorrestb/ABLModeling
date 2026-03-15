import unittest
import tempfile
import shutil
import json
from pathlib import Path
from ablsim.core.case import Case

class TestCaseClass(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory structure for testing
        self.test_dir = Path(tempfile.mkdtemp())
        self.case_path = self.test_dir / "test_case"
        self.case_path.mkdir()
        
        self.base_config = {
            "simulation": {"n_time_steps": 100},
            "profiles": {
                "fast": {"simulation": {"n_time_steps": 10}}
            }
        }
        
        with open(self.case_path / "case.json", "w") as f:
            json.dump(self.base_config, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_and_profiles(self):
        c = Case(self.case_path)
        self.assertEqual(c.name, "test_case")
        self.assertIn("fast", c.list_profiles())
        
        # Test default
        self.assertEqual(c.config["simulation"]["n_time_steps"], 100)
        
        # Test profile application
        c.apply_profile("fast")
        self.assertEqual(c.config["simulation"]["n_time_steps"], 10)
        self.assertEqual(c.active_profile_name, "fast")

    def test_output_dir(self):
        c = Case(self.case_path)
        od = c.get_output_dir(100, 10)
        self.assertTrue(od.name.startswith("Re_100_L_10"))
        
        c.apply_profile("fast")
        od = c.get_output_dir(100, 10)
        self.assertIn("fast", str(od))

    def test_validation(self):
        # Valid case
        c = Case(self.case_path)
        valid, _ = c.validate()
        self.assertTrue(valid)
        
        # Invalid case (missing file)
        (self.case_path / "case.json").unlink()
        c2 = Case(self.case_path) # Should fail or define issues? Case constructor tries to read file.
        # Currently Case constructor raises FileNotFoundError or ValueError if load fails.
        # But let's test validate method assuming object exists (maybe passed slightly broken init?)
        pass

if __name__ == "__main__":
    unittest.main()