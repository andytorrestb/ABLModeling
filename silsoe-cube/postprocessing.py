from lbmpy.session import *
import os
import numpy as np
import re
import glob
import pandas as pd
def plot_velocity(vel, domain_size, time_step, config):
    z_index_slice = int(0.1*domain_size[2])
    y_index_slice = int(domain_size[1] / 2)

    # Create a figure with two subplots side-by-side using the custom plt library
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2))

    # Plot z-slice velocity field on the first subplot
    plt.sca(ax1)
    plt.vector_field_magnitude(vel[:, :, z_index_slice, :2])
    # plt.vector_field(vel[:, :, z_index_slice, :2])
    # print(f'type(z_quiver): {type(z_quiver)}')
    ax1.set_title(f'Velocity Field (z-slice) Timestep {time_step}')

    # Plot y-slice velocity field on the second subplot
    plt.sca(ax2)
    y_slice_data = vel[:, y_index_slice, :, :][:, :, [0, 2]]  # X and Z components
    # print(f"Y-slice data shape: {np.shape(y_slice_data)}")  # Should be (120, 40, 2)
    plt.vector_field_magnitude(y_slice_data)

    # print(np.shape(vel[:, y_index_slice, :, [0, 2]]))
    # plt.vector_field(vel[:, y_index_slice, :, [0, 2]])
    # print(f'type(y_quiver): {type(y_quiver)}')
    ax2.set_title(f'Velocity Field (y-slice) Timestep {time_step}')

    # Save the combined figure
    output_dir = os.path.join(config.outdir, "vel_magnitude_output")
    os.makedirs(output_dir, exist_ok=True)
    # print(output_dir)
    # input()
    plt.savefig(os.path.join(output_dir, f'velocity_field_timestep_{time_step}.png'))
    plt.close(fig)
    # print(f'Successfully wrote combined velocity field for timestep {time_step}.')
    return

def plot_vorticity_frame(vel, domain_size, time_step, config):
    z_index_slice = int(0.2*domain_size[2])
    y_index_slice = int(domain_size[1] / 2)

    vorticity_z = vorticity_2d(vel[:, :, z_index_slice, :2])
    vorticity_y = vorticity_2d(vel[:, y_index_slice, :, :][:, :, [0, 2]])

    # print(f'vorticity_z: {vorticity_z.shape}, vorticity_y: {vorticity_y.shape}')

    import matplotlib.pyplot as plot  # Import here to avoid global conflicts

    fig, (ax1, ax2) = plot.subplots(1, 2, figsize=(12, 2))

    # Plot z-slice vorticity field
    im1 = ax1.imshow(vorticity_z.T, origin='lower', cmap='jet', aspect='auto')
    ax1.set_title(f'Vorticity Field (z-slice) Timestep {time_step}')
    fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)

    # Plot y-slice vorticity field
    im2 = ax2.imshow(vorticity_y.T, origin='lower', cmap='jet', aspect='auto')
    ax2.set_title(f'Vorticity Field (y-slice) Timestep {time_step}')
    fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)

    output_dir = os.path.join(config.outdir, "vorticity_output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'vorticity_field_timestep_{time_step}.png'))
    plot.close(fig)
    # print(f'Successfully wrote combined vorticity field for timestep {time_step}.')


def _compute_slice_indices(domain_size, z_ratio=0.1):
    """Helper to compute z- and y-slice indices with simple clamping.
    - z slice at z_ratio of domain depth (default 10%)
    - y slice at mid-height
    """
    nx, ny, nz = int(domain_size[0]), int(domain_size[1]), int(domain_size[2])
    z_idx = max(0, min(nz - 1, int(z_ratio * nz)))
    y_idx = max(0, min(ny - 1, int(ny / 2)))
    return y_idx, z_idx


def save_velocity_slices_csv(vel, domain_size, time_step, config, precision=6):
    """Save velocity field slices to CSV for easy downstream plotting.

    This mirrors the slice choices used in plot_velocity:
      - z-slice at ~10% depth: uses components (u, v) = vel[..., 0:2]
      - y-slice at mid-height: uses components (u, w) = vel[..., (0, 2)]

    Output files (created under {config.outdir}/slice_data):
      - vel_zslice_t{time_step}.csv with columns: x,y,z,u,v
      - vel_yslice_t{time_step}.csv with columns: x,y,z,u,w
    """
    # Shapes: vel is expected as (nx, ny, nz, 3)
    nx, ny, nz, _ = vel.shape
    y_idx, z_idx = _compute_slice_indices(domain_size)

    out_dir = os.path.join(config.outdir, "slice_data")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Z-slice (x-y plane) using components u, v
    z_slice = vel[:, :, z_idx, :2]  # (nx, ny, 2)
    xg, yg = np.meshgrid(np.arange(ny), np.arange(nx))  # note: meshgrid returns (nx, ny) if we set indexing='ij'
    # Fix ordering to x over rows, y over cols
    xg = np.arange(nx)[:, None].repeat(ny, axis=1)
    yg = np.arange(ny)[None, :].repeat(nx, axis=0)
    rows_z = np.column_stack([
        xg.ravel().astype(int),
        yg.ravel().astype(int),
        np.full(nx * ny, z_idx, dtype=int),
        z_slice[:, :, 0].ravel(),
        z_slice[:, :, 1].ravel(),
    ])
    z_csv = os.path.join(out_dir, f"vel_zslice_t{int(time_step):06d}.csv")
    np.savetxt(
        z_csv,
        rows_z,
        delimiter=",",
        header="x,y,z,u,v",
        comments="",
        fmt=["%d", "%d", "%d", f"%.{precision}g", f"%.{precision}g"],
    )

    # ---- Y-slice (x-z plane) using components u, w
    yz_slice = vel[:, y_idx, :, :][:, :, [0, 2]]  # (nx, nz, 2)
    xg2 = np.arange(nx)[:, None].repeat(nz, axis=1)
    zg2 = np.arange(nz)[None, :].repeat(nx, axis=0)
    rows_y = np.column_stack([
        xg2.ravel().astype(int),
        np.full(nx * nz, y_idx, dtype=int),
        zg2.ravel().astype(int),
        yz_slice[:, :, 0].ravel(),
        yz_slice[:, :, 1].ravel(),
    ])
    y_csv = os.path.join(out_dir, f"vel_yslice_t{int(time_step):06d}.csv")
    np.savetxt(
        y_csv,
        rows_y,
        delimiter=",",
        header="x,y,z,u,w",
        comments="",
        fmt=["%d", "%d", "%d", f"%.{precision}g", f"%.{precision}g"],
    )

    return {"z_csv": z_csv, "y_csv": y_csv}


def save_velocity_slices_npz(vel, domain_size, time_step, config):
    """Save velocity field slices to a compressed NPZ with metadata for compact storage.

    Arrays stored:
      - zslice_u, zslice_v for the z-slice (x-y plane)
      - yslice_u, yslice_w for the y-slice (x-z plane)
      - y_index, z_index, domain_size
    """
    nx, ny, nz, _ = vel.shape
    y_idx, z_idx = _compute_slice_indices(domain_size)

    out_dir = os.path.join(config.outdir, "slice_data")
    os.makedirs(out_dir, exist_ok=True)

    z_slice = vel[:, :, z_idx, :2]           # (nx, ny, 2) -> u,v
    y_slice = vel[:, y_idx, :, :][:, :, [0, 2]]  # (nx, nz, 2) -> u,w

    npz_path = os.path.join(out_dir, f"vel_slices_t{int(time_step):06d}.npz")
    np.savez_compressed(
        npz_path,
        zslice_u=z_slice[:, :, 0],
        zslice_v=z_slice[:, :, 1],
        yslice_u=y_slice[:, :, 0],
        yslice_w=y_slice[:, :, 1],
        y_index=int(y_idx),
        z_index=int(z_idx),
        domain_size=np.array(domain_size, dtype=int),
    )

    return npz_path


def _reshape_slice_from_rows(rows, kind):
    """Reshape flat CSV rows to grid arrays.
    kind: 'z' for z-slice (columns: x,y,z,u,v) -> (nx, ny, 2)
          'y' for y-slice (columns: x,y,z,u,w) -> (nx, nz, 2)
    """
    rows = np.asarray(rows)
    if kind == 'z':
        x, y, z, u, v = (rows[:, 0].astype(int), rows[:, 1].astype(int),
                         rows[:, 2].astype(int), rows[:, 3], rows[:, 4])
        nx, ny = x.max() + 1, y.max() + 1
        grid = np.zeros((nx, ny, 2), dtype=u.dtype)
        grid[x, y, 0] = u
        grid[x, y, 1] = v
        return grid
    elif kind == 'y':
        x, y, z, u, w = (rows[:, 0].astype(int), rows[:, 1].astype(int),
                         rows[:, 2].astype(int), rows[:, 3], rows[:, 4])
        nx, nz = x.max() + 1, z.max() + 1
        grid = np.zeros((nx, nz, 2), dtype=u.dtype)
        grid[x, z, 0] = u
        grid[x, z, 1] = w
        return grid
    else:
        raise ValueError("kind must be 'z' or 'y'")


def load_velocity_slices(slice_dir):
    """Load y- and z-slice CSV files and reshape into grid arrays per timestep.

    Returns:
        y_slices: list of (nx, nz, 2) arrays with components (u, w)
        z_slices: list of (nx, ny, 2) arrays with components (u, v)
        timesteps: sorted list of integer timesteps loaded
    """
    z_files = glob.glob(os.path.join(slice_dir, 'vel_zslice_t*.csv'))
    y_files = glob.glob(os.path.join(slice_dir, 'vel_yslice_t*.csv'))
    t_re = re.compile(r"t(\d+)\.csv$")

    def index_by_t(files):
        idx = {}
        for f in files:
            m = t_re.search(os.path.basename(f))
            if m:
                idx[int(m.group(1))] = f
        return idx

    z_idx = index_by_t(z_files)
    y_idx = index_by_t(y_files)
    common_t = sorted(set(z_idx.keys()) & set(y_idx.keys()))

    y_slices, z_slices = [], []
    for t in common_t:
        z_rows = np.loadtxt(z_idx[t], delimiter=',', skiprows=1)
        y_rows = np.loadtxt(y_idx[t], delimiter=',', skiprows=1)
        z_grid = _reshape_slice_from_rows(z_rows, 'z')
        y_grid = _reshape_slice_from_rows(y_rows, 'y')
        z_slices.append(z_grid)
        y_slices.append(y_grid)

    return y_slices, z_slices, common_t
