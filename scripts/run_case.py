import argparse
import sys
import logging
from pathlib import Path

# Ensure ablsim package is importable (assuming script is run from project root or installed)
# Add project root to sys.path if running from scripts/
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ablsim.core.runner import SimulationRunner
from ablsim.core.logging_utils import setup_console_logging
from ablsim.core.case import Case

def main():
    setup_console_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run ABL Simulation Case")
    parser.add_argument("case_path", type=Path, help="Path to the case directory")
    parser.add_argument("--re", type=float, nargs='+', help="Specific Reynolds number(s) to run")
    parser.add_argument("--ref", type=int, nargs='+', help="Specific Reference Length(s) to run")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")
    parser.add_argument("--workers", "-j", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--post", action="store_true", help="Run post-processing after simulation")

    args = parser.parse_args()

    if not args.case_path.exists():
        logger.error(f"Case path not found: {args.case_path}")
        sys.exit(1)

    try:
        runner = SimulationRunner(args.case_path)
        logger.info(f"Loaded case: {runner.case.name}")

        cfg = runner.case.config
        
        # Resolve parameters
        re_list = args.re if args.re else cfg.get('simulation', {}).get('reynolds_list', [])
        ref_list = args.ref if args.ref else cfg.get('simulation', {}).get('ref_length_list', [])
        
        if not re_list or not ref_list:
            logger.warning("No Reynolds numbers or reference lengths specified (check case.json).")
            # Maybe list what IS available if empty?
            
        logger.info(f"Planned runs for Re={re_list}, L={ref_list}")
        
        success = runner.run(
            re_list=re_list, 
            ref_list=ref_list, 
            dry_run=args.dry_run, 
            workers=args.workers,
            do_post=args.post
        )
        
        if not success:
            logger.error("Some tasks failed.")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Failed to run case: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
