from lbmpy.session import *
import os
import numpy as np
import re
import glob
import json
import logging
import argparse
import sys
import matplotlib.pyplot as plotting  # Standard matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def compute_slice_indices(domain_size, z_ratio=0.1):
    """
    Compute indices for mid-plane slices.
    """
    nx, ny, nz = int(domain_size[0]), int(domain_size[1]), int(domain_size[2])
    z_idx = max(0, min(nz - 1, int(z_ratio * nz)))
    y_idx = max(0, min(ny - 1, int(ny / 2)))
    return y_idx, z_idx

def save_velocity_slices_npz(z_slice, y_slice, y_idx, z_idx, time_step, config):
    """
    Save raw velocity slice data to compressed numpy file.
    Matches legacy signature from silsoe_cube.py usage.
    """
    save_velocity_slices(time_step, z_slice, y_slice, config.outdir, y_idx, z_idx)

def save_velocity_slices(time_step, z_slice_data, y_slice_data, outdir, y_idx=0, z_idx=0, filename_pattern="vel_slices_t{:06d}.npz"):
    """
    Clean implementation of saving slices.
    """
    slice_dir = os.path.join(outdir, "slice_data")
    os.makedirs(slice_dir, exist_ok=True)
    filename = filename_pattern.format(int(time_step))
    filepath = os.path.join(slice_dir, filename)
    
    np.savez_compressed(
        filepath,
        time_step=int(time_step),
        z_slice=z_slice_data, 
        y_slice=y_slice_data, 
        y_idx=y_idx,
        z_idx=z_idx,
        zslice_u=z_slice_data[..., 0],
        zslice_v=z_slice_data[..., 1],
        yslice_u=y_slice_data[..., 0],
        yslice_w=y_slice_data[..., 1]
    )


def load_velocity_slices_npz(filepath):
    """
    Load slice data from npz.
    """
    with np.load(filepath) as data:
        if 'z_slice' in data:
            return int(data['time_step']), data['z_slice'], data['y_slice']
        
        z_u = data['zslice_u']
        z_v = data['zslice_v']
        y_u = data['yslice_u']
        y_w = data['yslice_w']
        z_slice = np.stack([z_u, z_v], axis=-1)
        y_slice = np.stack([y_u, y_w], axis=-1)
        step = int(data['step']) if 'step' in data else 0
        return step, z_slice, y_slice

def vorticity_2d_fallback(velocity_field):
    """
    Compute 2D vorticity (curl) from a velocity field (..., 2).
    w = dv/dx - du/dy
    Assumes standard grid spacing dx=dy=1.
    """
    u = velocity_field[..., 0]
    v = velocity_field[..., 1]
    
    # Simple central difference, could be improved
    dy_u = np.gradient(u, axis=1)
    dx_v = np.gradient(v, axis=0)
    
    return dx_v - dy_u

vorticity_2d = vorticity_2d_fallback

def plot_velocity_slices(z_slice, y_slice, time_step, outdir):
    """
    Reconstruct the velocity visualization from saved 2D slices.
    z_slice: shape (nx, ny, 2) -> u, v components (slice const Z)
    y_slice: shape (nx, nz, 2) -> u, w components (slice const Y)
    """
    pass # Placeholder for logic not being modified in this query

def extract_validation_metrics(case_dir):
    """
    Compute validation relevant metrics from saved slices.
    Specifically targets Silsoe cube benchmark profiles if applicable.
    """
    slice_dir = os.path.join(case_dir, "slice_data")
    if not os.path.exists(slice_dir):
        logger.warning(f"No slice data found in {slice_dir}, skipping validation extraction.")
        return

    # 1. Load all slices to compute mean fields
    slice_files = sorted(glob.glob(os.path.join(slice_dir, "vel_slices_t*.npz")))
    if not slice_files:
        return

    # We only care about the centerline vertical plane (y_slice) for vertical profiles
    # y_slice shape is likely (nx, nz, 2) [u, w]
    
    sum_y_slice = None
    count = 0
    
    # Simple time averaging over all available frames
    # In a real run, you might want to discard the startup transient
    start_fraction = 0.2
    start_idx = int(len(slice_files) * start_fraction)
    process_files = slice_files[start_idx:]
    
    if not process_files:
        process_files = slice_files # Fallback if too few files

    logger.info(f"Computing mean profiles from {len(process_files)} frames...")
    
    # Read first file to get dimensions and params
    try:
        _, z_s, y_s = load_velocity_slices_npz(process_files[0])
        # y_s shape: (nx, nz, 2)
        sum_y_slice = np.zeros_like(y_s, dtype=np.float64)
        nx, nz = y_s.shape[:2]
    except Exception as e:
        logger.error(f"Failed to load first slice: {e}")
        return

    for f in process_files:
        try:
            _, _, y_s = load_velocity_slices_npz(f)
            sum_y_slice += y_s
            count += 1
        except Exception:
            continue
            
    if count == 0:
        return

    mean_y_slice = sum_y_slice / count
    mean_u = mean_y_slice[..., 0] # (nx, nz)
    mean_w = mean_y_slice[..., 1] # (nx, nz)
    
    # 2. Try to find config to identify cube location
    # We look for config_used.json or generic known positions
    # If not found, we blindly guess center or use fixed probes
    cube_x_idx = nx // 2 # Default guess
    cube_h_idx = 32      # Default guess (N_H)
    
    # Try to load config
    config_path = os.path.join(case_dir, "config_used.json")
    if not os.path.exists(config_path):
        # validation dir check
        config_path = os.path.join(os.path.dirname(case_dir), "config_used.json")

    probes = {}
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                
            # Try to extract geometry info
            # If wind_tunnel_scenario exists
            wt = cfg.get('wind_tunnel_scenario')
            geo = cfg.get('geometry', {})
            dom = cfg.get('domain_size_lu') # Check if we saved this
            
            # If we don't have explicit stored domain size in config, we might have to infer
            # But we do know the grid size from the slice (nx, nz)
            
            if wt and 'placement' in wt:
                # Based on our explicit placement logic
                # placement upstream_dist_H
                place = wt['placement']
                up_H = place.get('upstream_dist_H', 5.0)
                
                # We need to recover N_H (cells per cube height)
                # If ref_length was stored...
                N_H = cfg.get('reference_length', 32)
                cube_h_idx = N_H
                
                # Cube front face is at up_H * N_H
                # Cube center X is at (up_H + 0.5) * N_H
                cube_front_idx = int(up_H * N_H)
                cube_center_idx = int((up_H + 0.5) * N_H)
                cube_x_idx = cube_center_idx
            else:
                # Generic relative placement
                x_fac = geo.get('x_position_factor', 0.15)
                cube_x_idx = int(nx * x_fac)
                # Height?
                # We typically rely on config
                N_H = cfg.get('reference_length', 32)
                cube_h_idx = N_H
                
            # Define probes relative to cube location (in indices)
            # stored as (label, x_index_offset_from_center)
            # x_index = cube_x_idx + offset * N_H
            
            probe_defs = [
                ("inlet", -cube_x_idx), # x=0 roughly
                ("upstream_2H", -2),
                ("upstream_1H", -1),
                ("cube_center", 0),
                ("wake_1H", 1),
                ("wake_3H", 3),
                ("wake_5H", 5),
                ("outlet", (nx - 1 - cube_x_idx) / N_H)
            ]
            
            for label, offset in probe_defs:
                idx = int(cube_x_idx + offset * N_H)
                idx = max(0, min(nx - 1, idx))
                
                profile_u = mean_u[idx, :]
                profile_w = mean_w[idx, :]
                
                # Normalize Z coordinate by H
                z_axis = np.arange(nz) / N_H
                
                probes[label] = {
                    "z_over_H": z_axis.tolist(),
                    "mean_u": profile_u.tolist(),
                    "mean_w": profile_w.tolist(),
                    "x_index": idx
                }
                
    except Exception as e:
        logger.warning(f"Could not extract specific probe locations: {e}")
        
    # 3. Save Validation Data
    val_out = os.path.join(case_dir, "validation_metrics.json")
    output_data = {
        "averaged_frames": count,
        "profiles": probes,
        "reference_scales": {
            "cube_height_cells": cube_h_idx,
            "cube_x_index": cube_x_idx
        }
    }
    
    with open(val_out, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved validation metrics to {val_out}")

def process_output_directory(outdir):


    # --- Z-Slice (XY Plane, Side View usually if Y is vertical) ---
    # Data is [x, y, 2]
    # We transpose for imshow (rows=y, cols=x)
    u_z, v_z = z_slice[..., 0], z_slice[..., 1]
    mag_z = np.sqrt(u_z**2 + v_z**2)
    
    im1 = ax1.imshow(mag_z.T, origin='lower', cmap='viridis', aspect='auto')
    ax1.set_title(f'Vel Mag (XY-plane / Side) T={time_step}')
    fig.colorbar(im1, ax=ax1)

    # --- Y-Slice (XZ Plane, Top View usually if Y is vertical) ---
    # Data is [x, z, 2]
    u_y, w_y = y_slice[..., 0], y_slice[..., 1]
    mag_y = np.sqrt(u_y**2 + w_y**2)
    
    im2 = ax2.imshow(mag_y.T, origin='lower', cmap='viridis', aspect='auto')
    ax2.set_title(f'Vel Mag (XZ-plane / Top) T={time_step}')
    fig.colorbar(im2, ax=ax2)

    output_dir = os.path.join(outdir, "vel_magnitude_output")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'velocity_field_timestep_{time_step:06d}.png'), dpi=100)
    plt.close(fig)


def plot_vorticity_slices(z_slice, y_slice, time_step, outdir):
    """
    Compute and plot vorticity from loaded slices.
    """
    vort_z = vorticity_2d(z_slice) 
    vort_y = vorticity_2d(y_slice)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2))

    vlim = np.max(np.abs(vort_z)) * 0.8
    if vlim == 0: vlim = 1e-6
    im1 = ax1.imshow(vort_z.T, origin='lower', cmap='seismic', aspect='auto', vmin=-vlim, vmax=vlim)
    ax1.set_title(f'Vorticity (XY-plane) T={time_step}')
    fig.colorbar(im1, ax=ax1)

    vlim_y = np.max(np.abs(vort_y)) * 0.8
    if vlim_y == 0: vlim_y = 1e-6
    im2 = ax2.imshow(vort_y.T, origin='lower', cmap='seismic', aspect='auto', vmin=-vlim_y, vmax=vlim_y)
    ax2.set_title(f'Vorticity (XZ-plane) T={time_step}')
    fig.colorbar(im2, ax=ax2)

    output_dir = os.path.join(outdir, "vorticity_output")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'vorticity_timestep_{time_step:06d}.png'), dpi=100)
    plt.close(fig)

def create_video_from_frames(output_dir, image_folder_name, video_filename, fps=20):
    """
    Create an MP4 video from PNG frames in a subdirectory.
    """
    image_folder = os.path.join(output_dir, image_folder_name)
    video_path = os.path.join(output_dir, video_filename)
    
    if not os.path.exists(image_folder):
        logger.warning(f"Image folder not found: {image_folder}")
        return

    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not images:
        logger.warning(f"No images found in {image_folder}")
        return

    frame = cv2.imread(images[0])
    if frame is None:
        logger.error(f"Failed to read first image: {images[0]}")
        return
        
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(image)
        if frame is not None:
            video.write(frame)
    
    video.release()
    logger.info(f"Video saved to {video_path}")
    
import cv2

def process_output_directory(outdir):
    """
    Scan slice_data folder and generate plots for all .npz files.
    """
    slice_dir = os.path.join(outdir, "slice_data")
    if not os.path.exists(slice_dir):
        logger.warning(f"No slice_data found in {outdir}")
        return
    
    files = sorted(glob.glob(os.path.join(slice_dir, "*.npz")))
    if not files:
        logger.warning(f"No .npz files in {slice_dir}")
        return
        
    logger.info(f"Found {len(files)} slice files. Generating plots...")
    for fpath in files:
        try:
            ts, z_slice, y_slice = load_velocity_slices_npz(fpath)
            # Check if plots already exist to avoid re-work? 
            # For now, just overwrite or regenerate.
            plot_velocity_slices(z_slice, y_slice, ts, outdir)
            plot_vorticity_slices(z_slice, y_slice, ts, outdir)
        except Exception as e:
            logger.error(f"Failed to process {fpath}: {e}")

