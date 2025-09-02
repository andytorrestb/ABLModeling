import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
import postprocessing as post

def create_video(y_slices, z_slices, timesteps, out_path, cmap='viridis', fps=10):
    """Create an animation from y- and z-slices stacked vertically.

    y_slices: list of (nx, nz, 2) arrays (u,w)
    z_slices: list of (nx, ny, 2) arrays (u,v)
    timesteps: list of ints corresponding to frames
    out_path: mp4 output path
    """
    assert len(y_slices) == len(z_slices) == len(timesteps), "slice lists and timesteps must align"

    # Compute magnitudes
    def mag(arr2):
        return np.sqrt(arr2[..., 0]**2 + arr2[..., 1]**2)

    # Figure & axes
    fig, (ax_y, ax_z) = plt.subplots(2, 1, figsize=(8, 8))
    fig.tight_layout(pad=2.0)

    # Initial frames
    im_y = ax_y.imshow(mag(y_slices[0]).T, origin='lower', cmap=cmap, aspect='auto')
    ax_y.set_title(f'Y-slice | t={timesteps[0]}')
    cbar_y = fig.colorbar(im_y, ax=ax_y, fraction=0.046, pad=0.04)

    im_z = ax_z.imshow(mag(z_slices[0]).T, origin='lower', cmap=cmap, aspect='auto')
    ax_z.set_title(f'Z-slice | t={timesteps[0]}')
    cbar_z = fig.colorbar(im_z, ax=ax_z, fraction=0.046, pad=0.04)

    # Normalize color scale across all frames for consistency
    vmax = 0.0
    for ys, zs in zip(y_slices, z_slices):
        vmax = max(vmax, mag(ys).max(), mag(zs).max())
    im_y.set_clim(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
    im_z.set_clim(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)

    writer = FFMpegWriter(fps=fps, metadata=dict(artist='ABLModeling'))
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with writer.saving(fig, out_path, dpi=150):
        for frame_idx, (ys, zs, t) in enumerate(zip(y_slices, z_slices, timesteps)):
            im_y.set_data(mag(ys).T)
            ax_y.set_title(f'Y-slice | t={t}')
            im_z.set_data(mag(zs).T)
            ax_z.set_title(f'Z-slice | t={t}')
            writer.grab_frame()
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create velocity slice animation from CSV files')
    parser.add_argument('--slice-dir', default='output_Re_100000_L_11/slice_data', help='Directory containing CSV slice files')
    parser.add_argument('--out', default='slice_animation.mp4', help='Output MP4 path')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    args = parser.parse_args()

    result = post.load_velocity_slices(args.slice_dir)
    # Support both (y,z,t) and (y,z) returns
    if isinstance(result, tuple) and len(result) == 3:
        y_slices, z_slices, timesteps = result
    elif isinstance(result, tuple) and len(result) == 2:
        y_slices, z_slices = result
        timesteps = list(range(len(y_slices)))
    else:
        raise SystemExit('Unexpected return from load_velocity_slices. Expected 2 or 3 values.')
    if not timesteps:
        raise SystemExit(f'No matched y/z CSV files found under {args.slice_dir}')

    print(f'Loaded {len(timesteps)} frames from {args.slice_dir}. Writing {args.out} ...')
    try:
        create_video(y_slices, z_slices, timesteps, args.out, fps=args.fps)
    except Exception as e:
        # Fallback to Pillow if ffmpeg is missing
        print(f'FFMpegWriter failed ({e}); falling back to PillowWriter (GIF).')
        out_gif = os.path.splitext(args.out)[0] + '.gif'
        # Minimal re-render using Pillow
        def mag(a):
            return (a[...,0]**2 + a[...,1]**2) ** 0.5
        fig, (ax_y, ax_z) = plt.subplots(2, 1, figsize=(8, 8))
        im_y = ax_y.imshow(mag(y_slices[0]).T, origin='lower', cmap='viridis', aspect='auto')
        im_z = ax_z.imshow(mag(z_slices[0]).T, origin='lower', cmap='viridis', aspect='auto')
        vmax = 0.0
        for ys, zs in zip(y_slices, z_slices):
            vmax = max(vmax, mag(ys).max(), mag(zs).max())
        im_y.set_clim(0, vmax if vmax>0 else 1)
        im_z.set_clim(0, vmax if vmax>0 else 1)
        writer = PillowWriter(fps=args.fps)
        with writer.saving(fig, out_gif, dpi=100):
            for ys, zs, t in zip(y_slices, z_slices, timesteps):
                im_y.set_data(mag(ys).T); ax_y.set_title(f'Y-slice | t={t}')
                im_z.set_data(mag(zs).T); ax_z.set_title(f'Z-slice | t={t}')
                writer.grab_frame()
        plt.close(fig)
        print(f'Wrote {out_gif}')
    print('Done')