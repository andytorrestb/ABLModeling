from lbmpy.session import *
import os
import numpy as np
import re
import glob
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2))

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

