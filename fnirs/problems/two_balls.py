import numpy as np
from medium import Medium
from sensor_geometry import SensorGeometry

# head constants
HEAD_RADIUS_MM = 70


def two_balls_2d_medium(
    contrast=1.1, voxels_per_dim=150, r_ball_mm=5, depth_mm=20, ball_separation_mm=10
):
    """
    Builds a 2D medium with two balls of different optical properties.

    Parameters
    ----------
    contrast : float
        The relative change in absorption of the perturbation.
    voxels_per_dim : int, optional
        The number of voxels along each dimension of the medium. Default is 150.
    r_ball_mm : float, optional
        The radius of the ball in millimeters. Default is 5 mm.
    depth_mm : float, optional
        The depth of the balls in millimeters. Default is 20 mm.
    ball_separation_mm : float, optional
        The separation between the balls in millimeters. Default is 10 mm.

    Returns
    -------
    medium : Medium
        The 2D medium with two balls of different optical properties.
    """
    nz = 1
    ny = voxels_per_dim // 2
    nx = voxels_per_dim
    medium = Medium(
        (nz, ny, nx),
        metadata=f"contrast={contrast}_separation={ball_separation_mm}_radius={r_ball_mm}_depth={depth_mm}",
    )

    # head
    medium.add_ball((nz // 2, ny, nx // 2), HEAD_RADIUS_MM, 1)

    # set optical properties
    g = 0.9  # anisotropy factor
    mua0 = 0.02  # background absorption [1/mm]
    mus0 = 0.67 / (1 - g)  # background scattering [1/mm]
    refr_index = 1.4  # refractive index

    medium_bg = medium.copy()

    # turn off the black formatter for this section
    # fmt: off
    medium.optical_properties = np.array(
        [[0, 0, 1, 1],
         [mua0, mus0, g, refr_index]])

    medium_bg.optical_properties = np.array([
        [0, 0, 1, 1],
        [mua0, mus0, g, refr_index]]) # background only
    # fmt: on

    return medium_bg, medium


def two_balls_2d_sensors(noptodes: int, medium: Medium):
    """
    Builds a 2D sensor geometry for the two balls medium.

    Parameters
    ----------
    noptodes : int
        The number of optodes.
    medium : Medium
        The 2D medium with two balls of different optical properties.

    Returns
    -------
    sensors : SensorGeometry
        The 2D sensor geometry for the two balls medium.
    """
    det_pos = np.zeros((noptodes, 2))
    src_pos = np.zeros((noptodes, 2))

    # Create an array of i values
    i_values = np.arange(noptodes)

    # Calculate phi for q and m
    phi_sensors = i_values / noptodes * np.pi / 2 + np.pi / 4
    phi_sources = (i_values + 0.5) / noptodes * np.pi / 2 + np.pi / 4

    # Calculate sensor and source positions
    scaling_factor = 1

    center_point = np.array([0, medium.nx // 2])
    det_pos = (
        scaling_factor
        * HEAD_RADIUS_MM
        * np.vstack((np.sin(phi_sensors), np.cos(phi_sensors))).T
        + center_point
    )
    src_pos = (
        HEAD_RADIUS_MM * np.vstack((np.sin(phi_sources), np.cos(phi_sources))).T
        + center_point
    )
    src_dirs = center_point - src_pos
    src_dirs = src_dirs / np.linalg.norm(src_dirs, axis=1)[:, None]  # normalize

    # add a column of zeros to cast as 3D
    src_pos = np.hstack((medium.nz // 2 * np.ones((noptodes, 1)), src_pos))
    src_dirs = np.hstack((np.zeros((noptodes, 1)), src_dirs))
    det_pos = np.hstack((medium.nz // 2 * np.ones((noptodes, 1)), det_pos))
    det_pos = np.hstack((det_pos, np.ones((noptodes, 1))))

    sensors = SensorGeometry(src_pos, det_pos, src_dirs)
    return sensors
