from lbmpy.session import *
import os
import numpy as np
import re
import glob
import logging
import argparse
import sys
import matplotlib.pyplot as plotting  # Standard matplotlib

# Try to use lbmpy plotting if available, else standard fallback
try:
    # If lbmpy.session is imported, it might set 'plt', but we want explicit control
    import matplotlib.pyplot as plt
except ImportError:
    pass

logger = logging.getLogger(__name__)

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

# Attempt to use lbmpy version if available in namespace, else use fallback
if 'vorticity_2d' not in globals():
    vorticity_2d = vorticity_2d_fallback

def plot_velocity_slices(z_slice, y_slice, time_step, outdir):
    """
    Reconstruct the velocity visualization from saved 2D slices.
    z_slice: shape (nx, ny, 2) -> u, v components
    y_slice: shape (nx, nz, 2) -> u, w components
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2))

    # --- Z-Slice (Top/down view) ---
    u_z, v_z = z_slice[..., 0], z_slice[..., 1]
    mag_z = np.sqrt(u_z**2 + v_z**2)
    
    im1 = ax1.imshow(mag_z.T, origin='lower', cmap='viridis', aspect='auto')
    ax1.set_title(f'Vel Mag (xy-plane) T={time_step}')
    
    # --- Y-Slice (Side view) ---
    u_y, w_y = y_slice[..., 0], y_slice[..., 1]
    mag_y = np.sqrt(u_y**2 + w_y**2)
    
    im2 = ax2.imshow(mag_y.T, origin='lower', cmap='viridis', aspect='auto')
    ax2.set_title(f'Vel Mag (xz-plane) T={time_step}')

    output_dir = os.path.join(outdir, "vel_magnitude_output")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'velocity_field_timestep_{time_step}.png'), dpi=100)
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
    ax1.set_title(f'Vorticity (xy-plane) T={time_step}')
    fig.colorbar(im1, ax=ax1)

    vlim_y = np.max(np.abs(vort_y)) * 0.8
    if vlim_y == 0: vlim_y = 1e-6
    im2 = ax2.imshow(vort_y.T, origin='lower', cmap='seismic', aspect='auto', vmin=-vlim_y, vmax=vlim_y)
    ax2.set_title(f'Vorticity (xz-plane) T={time_step}')
    fig.colorbar(im2, ax=ax2)

    output_dir = os.path.join(outdir, "vorticity_output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'vorticity_field_timestep_{time_step}.png'), dpi=100)
    plt.close(fig)


def compute_slice_indices(domain_size, z_ratio=0.1):
    """Helper to compute z- and y-slice indices with simple clamping."""
    nx, ny, nz = int(domain_size[0]), int(domain_size[1]), int(domain_size[2])
    z_idx = max(0, min(nz - 1, int(z_ratio * nz)))
    y_idx = max(0, min(ny - 1, int(ny / 2)))
    return y_idx, z_idx

def save_velocity_slices_npz(z_slice, y_slice, y_idx, z_idx, time_step, config):
    """
    Runtime function: Saves compressed slices.
    called from silsoe_cube.py
    
    z_slice: (nx, ny, 2) array [u, v]
    y_slice: (nx, nz, 2) array [u, w]
    """
    out_dir = os.path.join(config.outdir, "slice_data")
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, f"vel_slices_t{int(time_step):06d}.npz")
    np.savez_compressed(
        npz_path,
        zslice_u=z_slice[:, :, 0],
        zslice_v=z_slice[:, :, 1],
        yslice_u=y_slice[:, :, 0],
        yslice_w=y_slice[:, :, 1],
        y_index=int(y_idx),
        z_index=int(z_idx),
        step=int(time_step),
        domain_size=np.array(config.domain_size, dtype=int),
    )
    return npz_path

def process_output_directory(outdir):
    """
    Offline function: Scans a directory for .npz files and generates plots.
    """
    slice_dir = os.path.join(outdir, "slice_data")
    if not os.path.exists(slice_dir):
        logger.error("No slice_data found in %s", outdir)
        return

    files = sorted(glob.glob(os.path.join(slice_dir, "vel_slices_t*.npz")))
    if not files:
        logger.warning("No .npz slice files found in %s", slice_dir)
        return

    logger.info("Found %d slice files to process in %s", len(files), outdir)

    for fpath in files:
        try:
            with np.load(fpath) as data:
                # Reconstruct slices
                # Z-slice (nx, ny, 2)
                z_u = data['zslice_u']
                z_v = data['zslice_v']
                z_slice = np.stack([z_u, z_v], axis=-1)

                # Y-slice (nx, nz, 2)
                y_u = data['yslice_u']
                y_w = data['yslice_w']
                y_slice = np.stack([y_u, y_w], axis=-1)

                step = data['step']

                # Generate plots
                plot_velocity_slices(z_slice, y_slice, step, outdir)
                plot_vorticity_slices(z_slice, y_slice, step, outdir)
                
        except Exception as e:
            logger.error("Failed to process %s: %s", fpath, e)

    logger.info("Post-processing complete for %s", outdir)

def create_video_from_frames(output_dir, image_folder, video_name, fps=20):
    """
    Create a video from a sequence of images using OpenCV.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python not installed. Skipping video generation.")
        return

    image_dir = os.path.join(output_dir, image_folder)
    if not os.path.exists(image_dir):
        logger.warning(f"Image directory {image_dir} does not exist. Skipping video generation.")
        return

    images = sorted([img for img in os.listdir(image_dir) if img.endswith(".png")])
    if not images:
        logger.warning(f"No images found in {image_dir}. Skipping video generation.")
        return

    # Read the first image to determine frame size
    first_frame_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape

    video_path = os.path.join(output_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    logger.info(f"Creating video {video_name} from {len(images)} frames...")
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_dir, image)))
    
    video.release()
    logger.info(f"Video saved to {video_path}")


def load_velocity_slices(slice_dir, pattern='vel_slices_t*.npz'):
    """
    Load all velocity slice .npz files from a directory.
    Returns:
        y_slices: list of (nx, nz, 2) arrays, components (u, w)
        z_slices: list of (nx, ny, 2) arrays, components (u, v)
        timesteps: sorted list of integers
    """
    files = sorted(glob.glob(os.path.join(slice_dir, pattern)))
    
    y_slices = []
    z_slices = []
    timesteps = []
    
    for fpath in files:
        with np.load(fpath) as data:
            # Reconstruct Z-slice (nx, ny, 2)
            z_u = data['zslice_u']
            z_v = data['zslice_v']
            z_slices.append(np.stack([z_u, z_v], axis=-1))

            # Reconstruct Y-slice (nx, nz, 2)
            y_u = data['yslice_u']
            y_w = data['yslice_w']
            y_slices.append(np.stack([y_u, y_w], axis=-1))
            
            # Step
            if 'step' in data:
                timesteps.append(int(data['step']))
            else:
                # Fallback: parse from filename
                m = re.search(r't(\d+)', os.path.basename(fpath))
                timesteps.append(int(m.group(1)) if m else len(timesteps))
                
    return y_slices, z_slices, timesteps

def load_velocity_slices(slice_dir, pattern='vel_slices_t*.npz'):
    """
    Load all velocity slice .npz files from a directory.
    Returns:
        y_slices: list of (nx, nz, 2) arrays, components (u, w)
        z_slices: list of (nx, ny, 2) arrays, components (u, v)
        timesteps: sorted list of integers
    """
    files = sorted(glob.glob(os.path.join(slice_dir, pattern)))
    
    y_slices = []
    z_slices = []
    timesteps = []
    
    for fpath in files:
        with np.load(fpath) as data:
            # Reconstruct Z-slice (nx, ny, 2)
            z_u = data['zslice_u']
            z_v = data['zslice_v']
            z_slices.append(np.stack([z_u, z_v], axis=-1))

            # Reconstruct Y-slice (nx, nz, 2)
            y_u = data['yslice_u']
            y_w = data['yslice_w']
            y_slices.append(np.stack([y_u, y_w], axis=-1))
            
            # Step
            if 'step' in data:
                timesteps.append(int(data['step']))
            else:
                try:
                    m = re.search(r't(\d+)', os.path.basename(fpath))
                    timesteps.append(int(m.group(1)) if m else len(timesteps))
                except:
                    timesteps.append(len(timesteps))
                
    return y_slices, z_slices, timesteps
