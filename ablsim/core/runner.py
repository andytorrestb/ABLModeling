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
        # Pass the pre-calculated output directory to the solver
        sim_conf = run_simulation(re, L, cfg, output_dir=outdir)
        out_path = sim_conf.outdir
        
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
                # Use strict output directory structure from Case
                outdir = self.case.get_output_dir(re, L)
                
                tasks.append((re, L, cfg, str(outdir), dry_run, effective_do_post))
        
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
