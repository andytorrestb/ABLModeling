import argparse
import sys
import logging
from pathlib import Path

# Ensure ablsim package is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ablsim.core.runner import SimulationRunner
from ablsim.core.logging_utils import setup_console_logging
from ablsim.core.case import Case

def find_case_path(name_or_path, root_dir):
    """
    Resolve a case name or path to an absolute path.
    """
    p = Path(name_or_path)
    if p.exists() and (p / "case.json").exists():
        return p
    
    # Check in cases/ directory
    cases_dir = root_dir / "cases"
    candidate = cases_dir / name_or_path
    if candidate.exists() and (candidate / "case.json").exists():
        return candidate
        
    return None

def main():
    setup_console_logging()
    logger = logging.getLogger("CLI")

    parser = argparse.ArgumentParser(description="ABL Simulation Manager")
    
    # Major modes
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", "-l", action="store_true", help="List all available cases")
    
    parser.add_argument("case_name", nargs="?", help="Name or path of the case to run/inspect")
    
    parser.add_argument("--info", "-i", action="store_true", help="Show details about the case")
    parser.add_argument("--profile", "-p", help="Run with a specific profile (e.g., 'smoke', 'validation')")
    
    parser.add_argument("--re", type=float, nargs='+', help="Override Reynolds number(s)")
    parser.add_argument("--ref", type=int, nargs='+', help="Override Reference Length(s)")
    
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")
    parser.add_argument("--workers", "-j", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--post", action="store_true", help="Force run post-processing")
    parser.add_argument("--no-post", action="store_true", help="Disable post-processing")

    args = parser.parse_args()

    # LIST CASES
    if args.list:
        cases = Case.find_cases(project_root / "cases")
        if not cases:
            print("No cases found in cases/ directory.")
        else:
            print(f"Available cases ({len(cases)}):")
            for c in cases:
                print(f"  - {c.name:20s} ({c.path})")
                profiles = c.list_profiles()
                if profiles:
                    print(f"    Profiles: {', '.join(profiles)}")
        return

    if not args.case_name:
        parser.print_help()
        sys.exit(1)

    # RESOLVE CASE
    case_path = find_case_path(args.case_name, project_root)
    if not case_path:
        logger.error(f"Case '{args.case_name}' not found.")
        sys.exit(1)

    try:
        runner = SimulationRunner(case_path)
        case = runner.case
    except Exception as e:
        logger.error(f"Failed to load case: {e}")
        sys.exit(1)

    # SHOW INFO
    if args.info:
        print(case.describe())
        valid, issues = case.validate()
        if valid:
            print("\nValidation: OK")
        else:
            print("\nValidation: FAILED")
            for issue in issues:
                print(f"  - {issue}")
        return

    # APPLY PROFILE
    if args.profile:
        try:
            case.apply_profile(args.profile)
            logger.info(f"Applied profile: {args.profile}")
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

    # RUN
    logger.info(f"Preparing to run case: {case.name}")

    cfg = case.config # This is potentially modified by profile
    
    # CLI Overrides take precedence over Profile/Config
    re_list = args.re if args.re else cfg.get('simulation', {}).get('reynolds_list', [])
    ref_list = args.ref if args.ref else cfg.get('simulation', {}).get('ref_length_list', [])

    # Post-processing logic
    # Default from config
    do_post = cfg.get('simulation', {}).get('run_post_processing', False)
    # CLI overrides
    if args.post: do_post = True
    if args.no_post: do_post = False
    
    if not re_list or not ref_list:
        logger.error("No run parameters (Re/L) defined. Check configuration or provide CLI arguments.")
        sys.exit(1)

    success = runner.run(
        re_list=re_list, 
        ref_list=ref_list, 
        dry_run=args.dry_run, 
        workers=args.workers,
        do_post=do_post
    )

    if not success:
        logger.error("Simulation run failed.")
        sys.exit(1)
    
    logger.info("Simulation run completed successfully.")

if __name__ == "__main__":
    main()
