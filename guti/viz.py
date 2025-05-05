import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from guti.core import (
    BRAIN_RADIUS,
    SKULL_RADIUS,
    SCALP_RADIUS,
    get_source_positions,
    get_sensor_positions,
    get_sensor_positions_spiral,
    get_voxel_mask,
    get_source_positions_halton,
)

import plotly.graph_objects as go


def visualize_sensors(sensors: np.ndarray):

    # Create figure
    fig = go.Figure()

    # Add source positions
    fig.add_trace(
        go.Scatter3d(
            x=sensors.src_pos[:, 0],
            y=sensors.src_pos[:, 1],
            z=sensors.src_pos[:, 2],
            mode="markers",
            marker=dict(size=2, symbol="circle", color="red"),
            name="Sources",
        )
    )

    # Add detector positions
    fig.add_trace(
        go.Scatter3d(
            x=sensors.det_pos[:, 0],
            y=sensors.det_pos[:, 1],
            z=sensors.det_pos[:, 2],
            mode="markers",
            marker=dict(size=2, symbol="circle", color="blue"),
            name="Detectors",
        )
    )

    # Update layout
    fig.update_layout(
        title="Sensor Positions",
        scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
        showlegend=True,
    )

    fig.show()


def visualize_brain_model(
    sources=None, sensors=None, field=None, resolution=1, alpha=0.2, figsize=(10, 8)
):
    """
    Visualize the three-layer brain model (brain, skull, scalp) with optional sources, sensors, and field data.

    Parameters:
    -----------
    sources : np.ndarray, optional
        Array of shape (n_sources, 3) containing source positions.
    sensors : np.ndarray, optional
        Array of shape (n_sensors, 3) containing sensor positions.
    field : np.ndarray, optional
        3D array representing a field to visualize (must match the voxel mask dimensions).
    resolution : float, optional
        Resolution of the voxel mask in mm.
    alpha : float, optional
        Transparency of the brain layers.
    figsize : tuple, optional
        Figure size.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Get voxel mask for the three-layer model
    mask = get_voxel_mask(resolution)
    nx, ny, nz = mask.shape

    # Create coordinate grids
    x = np.linspace(0, (BRAIN_RADIUS * 2), nx)
    y = np.linspace(0, (BRAIN_RADIUS * 2), ny)
    z = np.linspace(0, BRAIN_RADIUS, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Visualize the three layers if no field is provided
    if field is None:
        # Brain (layer 1)
        brain_points = np.where(mask == 1)
        ax.scatter(
            X[brain_points],
            Y[brain_points],
            Z[brain_points],
            c="red",
            alpha=0.01,
            label="Brain",
        )

        # Skull (layer 2)
        skull_points = np.where(mask == 2)
        ax.scatter(
            X[skull_points],
            Y[skull_points],
            Z[skull_points],
            c="beige",
            alpha=0.01,
            label="Skull",
        )

        # Scalp (layer 3)
        scalp_points = np.where(mask == 3)
        ax.scatter(
            X[scalp_points],
            Y[scalp_points],
            Z[scalp_points],
            c="bisque",
            alpha=0.01,
            label="Scalp",
        )
    else:
        # Visualize the field
        if field.shape != mask.shape:
            raise ValueError(
                f"Field shape {field.shape} does not match mask shape {mask.shape}"
            )

        # Only show field values where mask is non-zero
        field_masked = np.copy(field)
        field_masked[mask == 0] = np.nan

        # Plot non-zero field values
        points = np.where(~np.isnan(field_masked))
        sc = ax.scatter(
            X[points],
            Y[points],
            Z[points],
            c=field_masked[points],
            cmap="viridis",
            alpha=alpha,
        )
        plt.colorbar(sc, ax=ax, label="Field Value")

    # Add sources if provided
    if sources is not None:
        ax.scatter(
            sources[:, 0],
            sources[:, 1],
            sources[:, 2],
            c="blue",
            s=50,
            marker="*",
            label="Sources",
        )

    # Add sensors if provided
    if sensors is not None:
        ax.scatter(
            sensors[:, 0],
            sensors[:, 1],
            sensors[:, 2],
            c="green",
            s=30,
            marker="^",
            label="Sensors",
        )

    # Set labels and title
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Brain Model Visualization")

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 0.5])  # Adjust for hemisphere

    # Add legend
    ax.legend()

    return fig, ax


def plot_slice(
    field, slice_dim="z", slice_index=None, mask=None, resolution=1, figsize=(8, 6)
):
    """
    Plot a 2D slice of the 3D field.

    Parameters:
    -----------
    field : np.ndarray
        3D array representing the field to visualize.
    slice_dim : str, optional
        Dimension to slice along ('x', 'y', or 'z').
    slice_index : int, optional
        Index of the slice. If None, the middle slice is used.
    mask : np.ndarray, optional
        3D array representing the brain mask.
    resolution : float, optional
        Resolution of the voxel mask in mm.
    figsize : tuple, optional
        Figure size.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if mask is None:
        mask = get_voxel_mask(resolution)

    nx, ny, nz = field.shape

    # Create coordinate grids
    x = np.linspace(-BRAIN_RADIUS / resolution, BRAIN_RADIUS / resolution, nx)
    y = np.linspace(-BRAIN_RADIUS / resolution, BRAIN_RADIUS / resolution, ny)
    z = np.linspace(0, BRAIN_RADIUS / resolution, nz)

    # Set default slice index to middle if not provided
    if slice_index is None:
        if slice_dim == "x":
            slice_index = nx // 2
        elif slice_dim == "y":
            slice_index = ny // 2
        elif slice_dim == "z":
            slice_index = nz // 2

    fig, ax = plt.subplots(figsize=figsize)

    # Extract the slice
    if slice_dim == "x":
        slice_data = field[slice_index, :, :]
        mask_slice = mask[slice_index, :, :]
        extent = [y[0], y[-1], z[0], z[-1]]
        xlabel, ylabel = "Y (mm)", "Z (mm)"
        title = f"X-Slice at {x[slice_index]:.1f} mm"
    elif slice_dim == "y":
        slice_data = field[:, slice_index, :]
        mask_slice = mask[:, slice_index, :]
        extent = [x[0], x[-1], z[0], z[-1]]
        xlabel, ylabel = "X (mm)", "Z (mm)"
        title = f"Y-Slice at {y[slice_index]:.1f} mm"
    elif slice_dim == "z":
        slice_data = field[:, :, slice_index]
        mask_slice = mask[:, :, slice_index]
        extent = [x[0], x[-1], y[0], y[-1]]
        xlabel, ylabel = "X (mm)", "Y (mm)"
        title = f"Z-Slice at {z[slice_index]:.1f} mm"

    # Mask out regions outside the brain model
    masked_data = np.copy(slice_data)
    masked_data[mask_slice == 0] = np.nan

    # Plot the slice
    im = ax.imshow(
        masked_data.T, origin="lower", extent=extent, cmap="viridis", aspect="equal"
    )

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Field Value")

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig, ax
