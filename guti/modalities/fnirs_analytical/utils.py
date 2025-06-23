import numpy as np


def compute_perpendicular_distance(pos, source_pos, detector_pos):
    """
    Compute the perpendicular distance from points to the source-detector line.
    
    This represents the z-coordinate as if the source were at the origin 
    and the detector were at (d_source_detector, 0, 0).
    
    Parameters
    ----------
    pos : numpy.ndarray
        The positions to calculate perpendicular distance for. Shape (n_points, 3).
    source_pos : numpy.ndarray
        The position of the source in mm.
    detector_pos : numpy.ndarray
        The position of the detector in mm.
        
    Returns
    -------
    z : numpy.ndarray
        Perpendicular distances from each point to the source-detector line.
    """
    # Distance from source to detector
    d_source_detector = np.linalg.norm(detector_pos - source_pos)
    
    # Transform coordinates: translate so source is at origin
    pos_rel = pos - source_pos
    
    # Unit vector from source to detector
    dir_source_detector = (detector_pos - source_pos) / d_source_detector
    
    # Project each relative position onto the source-detector line
    d_proj = np.dot(pos_rel, dir_source_detector)
    
    # Vector from each point to its projection on the line
    vec_perp = pos_rel - d_proj[:, np.newaxis] * dir_source_detector
    
    # z is the magnitude of the perpendicular vector
    z = np.linalg.norm(vec_perp, axis=1)
    
    return z


def cw_sensitivity(pos, source_pos, detector_pos, mu_eff):
    """
    Calculate the continuous wave sensitivity function for given positions and source-detector distance.
    The sensitivity function is with respect to mu_a at pos_mm.
    
    From eq (14.8) in the textbook Quantitative Biomedical Optics by Bigio and Fantini (p. 427).

    Parameters
    ----------
    pos : numpy.ndarray
        The positions to calculate sensitivity for. Shape can be either (3,) for a single point
        or (n_points, 3) for multiple points.
    source_pos : numpy.ndarray
        The position of the source in mm.
    detector_pos : numpy.ndarray
        The position of the detector in mm.
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

    # Calculate distances
    d_source_pos = np.linalg.norm(source_pos - pos, axis=1)
    d_detector_pos = np.linalg.norm(detector_pos - pos, axis=1)

    # compute z as if the source were at the origin and the detector were at (d_detector_pos, 0, 0)
    z = compute_perpendicular_distance(pos, source_pos, detector_pos)
    
    # Calculate sensitivity
    sensitivity = (
        z ** 2
        * (mu_eff + 1 / d_source_pos)
        * (mu_eff + 1 / d_detector_pos)
        * np.exp(-mu_eff * (d_source_pos + d_detector_pos))  # TODO: check if this is correct
        / (d_source_pos**2 * d_detector_pos**2)
    )

    # Return scalar if input was a single point
    return sensitivity[0] if pos.shape[0] == 1 else sensitivity
