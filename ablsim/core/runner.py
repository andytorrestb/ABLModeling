import concurrent.futures
import logging
import time
from pathlib import Path
from ablsim.core.case import Case
from ablsim.core.logging_utils import attach_file_logger
from ablsim.solvers.lbm import run_simulation
from ablsim.post.driver import process_case  # Post-processing callable

logger = logging.getLogger(__name__)

def run_single_case_worker(args):
    """
    Worker function to execute a single simulation case.
    args: (reynolds_number, ref_length, case_config_dict, output_dir, dry_run, do_post)
    """
    re, L, cfg, outdir, dry_run, do_post = args
    
    # Configure per-process file logger
    attach_file_logger(outdir, filename="run.log")
    worker_log = logging.getLogger(f"worker_Re{int(re)}_L{L}")
    
    worker_log.info("Starting worker for Re=%.0e, L=%d -> %s", re, L, outdir)
    
    if dry_run:
        worker_log.info("Dry-run: skipping execution.")
        return True

    try:
        # Update config to include geometry overrides if any?
        # The cfg passed here is fully resolved.
        
        # Override outdir in cfg if needed?
        # run_simulation takes (re, L, cfg). Inside run_simulation, it reconstructs SimulationConfig and re-calculates outdir?
        # Let's check run_simulation logic:
        # outdir = f"output_Re_{int(reynolds_number):d}_L_{ref_length:d}"
        # config = SimulationConfig(..., outdir=outdir, ...)
        
        # This is hardcoded inside run_simulation to use current directory relative path.
        # We need to change run_simulation (in lbm.py) to respect "outdir" passed or accept explicit outdir.
        # But run_simulation signature is fixed (re, L, cfg).
        # We can pass 'output_dir' inside cfg?
        # config.py reads 'outdir' from argument to SimulationConfig constructor.
        # But lbm.py:run_simulation calls SimulationConfig constructor with hardcoded path string.
        
        # To fix this cleanly without editing lbm.py deeply:
        # lbm.py run_simulation calculates `outdir = f"output_Re..."`.
        # This creates it in current working directory.
        # We usually run from case directory, so CWD is `cases/case_X`.
        # So "output_Re..." will be inside `cases/case_X`.
        # That's acceptable for now.
        
        # But if we want absolute paths or "results/" subfolder (as planned in Case.get_output_dir), we need to patch lbm.py.
        # Patching lbm.py run_simulation to respect a passed output directory path would be best.
        
        # I'll let lbm.py do its default thing for now (create output_Re... in CWD) 
        # but change CWD before running worker? No, changing CWD is global.
        # Worker process? Yes, os.chdir in worker is okay-ish but confusing.
        
        # Better: run_simulation returns the config object which has .outdir.
        sim_conf = run_simulation(re, L, cfg)
        out_path = sim_conf.outdir # This is what lbm.py decided.
        
        worker_log.info("Simulation finished. Output at: %s", out_path)
        
        if do_post:
            worker_log.info("Starting post-processing...")
            try:
                # determine video settings from config
                gen_video = cfg.get('simulation', {}).get('generate_video', True)
                process_case(out_path, generate_video=gen_video) 
            except Exception as e:
                worker_log.error(f"Post-processing failed: {e}")
                
        return True
    except Exception as e:
        worker_log.exception(f"Run failed: {e}")
        return False

class SimulationRunner:
    def __init__(self, case_path):
        self.case = Case(case_path)
        
    def run(self, re_list=None, ref_list=None, dry_run=False, workers=1, do_post=False):
        """
        Execute the parameter sweep defined in case config or overrides.
        """
        cfg = self.case.config
        
        # Determine sweep parameters
        reynolds_list = re_list if re_list else cfg.get('simulation', {}).get('reynolds_list', [])
        ref_length_list = ref_list if ref_list else cfg.get('simulation', {}).get('ref_length_list', [])
        
        if not reynolds_list or not ref_length_list:
            logger.error("No Reynolds numbers or reference lengths to run.")
            return

        # Check for post-processing config default
        config_do_post = cfg.get('simulation', {}).get('run_post_processing', False)
        # CLI overrides config IF set (but here 'do_post' is boolean from CLI default False)
        # Strategy: Enable if EITHER is True.
        effective_do_post = do_post or config_do_post

        # Prepare tasks
        tasks = []
        for re in reynolds_list:
            for L in ref_length_list:
                # We can predict the outdir if we know lbm.py logic, or let it run.
                # Just placeholder outdir for logger setup (lbm.py will create same name)
                outdir = f"output_Re_{int(re)}_L_{L:d}"
                # If we run from script, CWD is often case dir.
                # We should clarify CWD expectation.
                
                tasks.append((re, L, cfg, outdir, dry_run, effective_do_post))
        
        logger.info(f"Planned {len(tasks)} runs with {workers} workers.")
        
        # Execute
        if workers > 1 and not dry_run:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(run_single_case_worker, tasks))
            success = all(results)
        else:
            # Sequential execution for debug or dry-run
            success = True
            for task in tasks:
                if not run_single_case_worker(task):
                    success = False
        
        return success
