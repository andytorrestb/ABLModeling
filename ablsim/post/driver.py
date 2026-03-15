import argparse
import logging
import os
import time
import json
import glob
from ablsim.post.processing import process_output_directory, create_video_from_frames, extract_validation_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def process_case(case_dir, generate_video=True):
    """
    Main processing logic for a single simulation case directory.
    Uses library functions from postprocessing.py
    """
    logger.info(f"Starting post-processing for: {case_dir}")
    t_start_post = time.perf_counter()
    
    # 0. Extraction of Validation Metrics (Profiles)
    # This is non-graphical and fast
    try:
        extract_validation_metrics(case_dir)
    except Exception as e:
        logger.error(f"Error during validation extraction: {e}")
    
    # 1. Plot generation
    t_start_plot = time.perf_counter()
    try:
        process_output_directory(case_dir)
    except Exception as e:
        logger.error(f"Error during plot generation: {e}")

    t_end_plot = time.perf_counter()
    plot_duration = t_end_plot - t_start_plot
    logger.info(f"  Plots generated in {plot_duration:.2f}s")

    # 2. Video generation
    video_duration = 0.0
    if generate_video:
        t_start_video = time.perf_counter()
        try:
            create_video_from_frames(case_dir, "vel_magnitude_output", "vel_magnitude.mp4", fps=20)
            create_video_from_frames(case_dir, "vorticity_output", "vorticity.mp4", fps=20)
        except Exception as e:
            logger.error(f"  Video generation failed: {e}")
        
        video_duration = time.perf_counter() - t_start_video
        logger.info(f"  Videos generated in {video_duration:.2f}s")

    t_total_post = time.perf_counter() - t_start_post
    logger.info(f"Finished post-processing for {case_dir}. Total time: {t_total_post:.2f}s")


    # 3. Consolidate Timing
    # Try to load simulation timing if it exists
    sim_timing_file = os.path.join(case_dir, "timing_sim.json")
    sim_stats = {}
    if os.path.exists(sim_timing_file):
        try:
            with open(sim_timing_file, 'r') as f:
                sim_stats = json.load(f)
        except:
            pass
    
    post_stats = {
        "plot_generation_time_sec": plot_duration,
        "video_generation_time_sec": video_duration,
        "total_postprocessing_time_sec": t_total_post
    }
    
    # helper for safe float addition
    def get_time(d, k): return d.get(k, 0.0)

    grand_total = get_time(sim_stats, "total_simulation_time_sec") + t_total_post

    combined_stats = {
        "simulation": sim_stats,
        "postprocessing": post_stats,
        "grand_total_time_sec": grand_total
    }
    
    # Write summary
    summary_file = os.path.join(case_dir, "timing_summary.json")
    with open(summary_file, "w") as f:
        json.dump(combined_stats, f, indent=2)
    
    logger.info(f"Timing summary saved to {summary_file}")
    logger.info(f"Grand Total Time: {grand_total:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone post-processing for Silsoe Cube LBM")
    parser.add_argument("case_dirs", nargs='+', help="List of output directories to process")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    args = parser.parse_args()

    for case_dir in args.case_dirs:
        if os.path.isdir(case_dir):
            process_case(case_dir, generate_video=(not args.no_video))
        else:
            logger.warning(f"Skipping invalid directory: {case_dir}")
