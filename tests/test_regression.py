import sys
import shutil
import unittest
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablsim.core.case import Case

class TestRegression(unittest.TestCase):
    def setUp(self):
        self.case_name = "silsoe_cube"
        self.case_path = PROJECT_ROOT / "cases" / self.case_name
        self.results_dir = self.case_path / "results"
        
        # Clean up previous smoke results
        smoke_results = self.results_dir / "smoke"
        if smoke_results.exists():
            shutil.rmtree(smoke_results)

    def test_smoke_profile(self):
        """
        Run the 'smoke' profile for silsoe_cube and verify outputs.
        """
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_case.py"),
            self.case_name,
            "--profile", "smoke",
            "--workers", "1"
        ]
        
        # Run process
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check return code
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        
        self.assertEqual(result.returncode, 0, "Smoke run failed")
        
        # Verify output directory structure
        # smoke profile: Re=1000, L=5 (from my edit earlier)
        expected_dir = self.results_dir / "smoke" / "Re_1000_L_5"
        self.assertTrue(expected_dir.exists(), f"Output directory not created: {expected_dir}")
        self.assertTrue((expected_dir / "run.log").exists(), "Run log not created")

    def test_list_cases(self):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_case.py"),
            "--list"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("silsoe_cube", result.stdout)

if __name__ == "__main__":
    unittest.main()