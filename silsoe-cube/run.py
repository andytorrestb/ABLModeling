import argparse
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time

from config_sim import load_config
from silsoe_cube import run_simulation  # uses signature run_simulation(re, L, cfg)

from logging_setup import setup_console_logging, attach_file_logger

def main():
    setup_console_logging()

    parser = argparse.ArgumentParser(description="Orchestrate ship-wake LBM simulations")
    parser.add_argument("--config", "-c", default="config.json", help="Path to config.json")
    parser.add_argument("--re", type=float, help="Single Reynolds number to run (overrides list in config)")
    parser.add_argument("--ref", type=int, help="Single reference length to run (overrides list in config)")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and show planned runs without executing")
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
    logging.getLogger(__name__).info("Planned runs: %d (Re) x %d (L) = %d cases",
                                     len(reynolds_list), len(ref_length_list), len(reynolds_list) * len(ref_length_list))
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

    start_all = time.time()
    for Re in reynolds_list:
        for L in ref_length_list:
            outdir = f"output_Re_{int(Re):d}_L_{L:d}"
            # attach per-run file logger so each run writes its own log file
            attach_file_logger(outdir)

            logging.getLogger(__name__).info("Starting case Re=%s L=%s -> outdir=%s", Re, L, outdir)
            if args.dry_run:
                logging.getLogger(__name__).info("Dry-run: skipping simulation execution for this case.")
                continue

            try:
                sim_conf = run_simulation(Re, L, cfg)
                logging.getLogger(__name__).info("Completed case Re=%s L=%s; output directory: %s", Re, L, getattr(sim_conf, "outdir", outdir))
            except Exception as e:
                logging.getLogger(__name__).exception("Simulation failed for Re=%s L=%s: %s", Re, L, e)

    total_elapsed = time.time() - start_all
    logging.getLogger(__name__).info("All cases finished in %.2fs", total_elapsed)

if __name__ == "__main__":
    main()