import argparse
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
import concurrent.futures

from config_sim import load_config
from silsoe_cube import run_simulation  # uses signature run_simulation(re, L, cfg)
try:
    from postprocess_case import process_case
except ImportError:
    process_case = None

from logging_setup import setup_console_logging, attach_file_logger

def run_single_case(Re, L, cfg, dry_run, do_post):
    """
    Worker function to execute a single simulation case.
    Designed to be picklable for multiprocessing.
    """
    outdir = f"output_Re_{int(Re):d}_L_{L:d}"
    
    # In a subprocess, we need to re-attach the file logger because
    # logging handlers are not inherited or shared across process boundaries in the same way.
    # We configure a logger specific to this process/run.
    attach_file_logger(outdir)
    log = logging.getLogger(f"case_Re{Re}_L{L}")

    log.info("Starting case Re=%s L=%s -> outdir=%s", Re, L, outdir)
    
    if dry_run:
        log.info("Dry-run: skipping simulation execution for this case.")
        return True

    try:
        sim_conf = run_simulation(Re, L, cfg)
        out_path = getattr(sim_conf, "outdir", outdir)
        log.info("Completed case Re=%s L=%s; output directory: %s", Re, L, out_path)
        
        if do_post:
            if process_case is None:
                log.warning("Refusing --post because postprocess_case.py failed to import.")
            else:
                log.info("Starting post-processing for %s...", out_path)
                process_case(out_path)
                log.info("Post-processing complete.")
        return True
    except Exception as e:
        log.exception("Run failed for Re=%s L=%s: %s", Re, L, e)
        return False

def main():
    setup_console_logging()

    parser = argparse.ArgumentParser(description="Orchestrate ship-wake LBM simulations")
    parser.add_argument("--config", "-c", default="config.json", help="Path to config.json")
    parser.add_argument("--re", type=float, help="Single Reynolds number to run (overrides list in config)")
    parser.add_argument("--ref", type=int, help="Single reference length to run (overrides list in config)")
    parser.add_argument("--post", action="store_true", help="Run post-processing immediately after simulation")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and show planned runs without executing")
    parser.add_argument("--workers", "-j", type=int, default=1, help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    logging.getLogger(__name__).info("Loading configuration from %s", cfg_path)
    cfg = load_config(str(cfg_path))

    sim_cfg = cfg.get("simulation", {})
    phys = cfg.get("physical", {})
    domain = cfg.get("domain", {})

    reynolds_list = [args.re] if args.re is not None else sim_cfg.get("reynolds_list", [])
    ref_length_list = [args.ref] if args.ref is not None else sim_cfg.get("ref_length_list", [])

    if not reynolds_list or not ref_length_list:
        logging.getLogger(__name__).error("No Reynolds numbers or reference lengths specified (check config or args).")
        return

    # Log summary of the run plan and derived values worth tracking
    total_cases = len(reynolds_list) * len(ref_length_list)
    logging.getLogger(__name__).info("Planned runs: %d (Re) x %d (L) = %d cases",
                                     len(reynolds_list), len(ref_length_list), total_cases)
    logging.getLogger(__name__).debug("simulation block: %s", sim_cfg)
    logging.getLogger(__name__).debug("physical block: %s", phys)
    logging.getLogger(__name__).debug("domain block: %s", domain)

    # Example useful derived value (if available)
    L_phy = phys.get("L_phy")
    nu_phy = phys.get("nu_phy")
    if L_phy is not None and nu_phy is not None:
        for Re in reynolds_list:
            u_phy = (Re * nu_phy) / L_phy
            logging.getLogger(__name__).info("Derived physical velocity for Re=%s : u_phy=%.4g (units per config)", Re, u_phy)

    # Prepare list of tasks
    tasks = []
    for Re in reynolds_list:
        for L in ref_length_list:
            tasks.append((Re, L))

    start_all = time.time()
    
    # Execute tasks using ProcessPoolExecutor
    max_workers = args.workers if args.workers > 0 else 1
    logging.getLogger(__name__).info("Executing with %d workers...", max_workers)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(run_single_case, Re, L, cfg, args.dry_run, args.post): (Re, L) for Re, L in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            Re, L = futures[future]
            try:
                success = future.result()
                status = "succeeded" if success else "failed"
                logging.getLogger(__name__).info("Task Re=%s L=%s finished (%s)", Re, L, status)
            except Exception as e:
                logging.getLogger(__name__).error("Task Re=%s L=%s generated an exception: %s", Re, L, e)

    total_elapsed = time.time() - start_all
    logging.getLogger(__name__).info("All cases finished in %.2fs", total_elapsed)

if __name__ == "__main__":
    main()