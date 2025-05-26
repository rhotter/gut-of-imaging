import numpy as np


def cw_sensitivity(pos, source_det_dist, mu_eff):
    """
    Calculate the continuous wave sensitivity function for given positions and source-detector distance.
    The sensitivity function is with respect to mu_a at pos_mm.

    Parameters
    ----------
    pos : numpy.ndarray
        The positions to calculate sensitivity for. Shape can be either (3,) for a single point
        or (n_points, 3) for multiple points.
    source_det_dist : float
        The distance between the source and detector in mm.
    mu_eff : float
        The effective attenuation coefficient in mm^-1.

    Returns
    -------
    sensitivity : numpy.ndarray
        The sensitivity function. Shape is (n_points,) if pos has shape (n_points, 3),
        or a scalar if pos has shape (3,).
    """
    # Handle both single point and multiple points
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)

    # the detector is located at (source_det_dist, 0, 0)
    det_pos = np.array([source_det_dist, 0, 0])

    # Calculate distances for all points at once
    r = np.linalg.norm(pos, axis=1)
    r_det = np.linalg.norm(det_pos - pos, axis=1)

    # Calculate sensitivity for all points at once
    sensitivity = (
        pos[:, 2] ** 2
        * (mu_eff + 1 / r)
        * (mu_eff + 1 / r_det)
        * np.exp(-mu_eff * (r + r_det))
        / (r**2 * r_det**2)
    )

    # Return scalar if input was a single point
    return sensitivity[0] if pos.shape[0] == 1 else sensitivity
