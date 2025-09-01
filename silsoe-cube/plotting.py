from lbmpy.session import *
import os

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