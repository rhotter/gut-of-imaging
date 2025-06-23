import numpy as np
import torch
from typing import Tuple


def compute_perpendicular_distance(pos: torch.Tensor, source_pos: torch.Tensor, detector_pos: torch.Tensor) -> torch.Tensor:
    """
    Compute the perpendicular distance from points to the source-detector line using PyTorch.
    
    Parameters
    ----------
    pos : torch.Tensor
        The positions to calculate perpendicular distance for. Shape (n_points, 3).
    source_pos : torch.Tensor
        The source positions. Shape (n_pairs, 3).
    detector_pos : torch.Tensor
        The detector positions. Shape (n_pairs, 3).
        
    Returns
    -------
    z : torch.Tensor
        Perpendicular distances. Shape (n_pairs, n_points).
    """
    # Distance from source to detector for each pair
    d_source_detector = torch.norm(detector_pos - source_pos, dim=1)  # (n_pairs,)
    
    # Transform coordinates: translate so source is at origin
    # pos: (n_points, 3), source_pos: (n_pairs, 3) -> pos_rel: (n_pairs, n_points, 3)
    pos_rel = pos[None, :, :] - source_pos[:, None, :]
    
    # Unit vector from source to detector for each pair
    dir_source_detector = (detector_pos - source_pos) / d_source_detector[:, None]  # (n_pairs, 3)
    
    # Project each relative position onto the source-detector line
    # pos_rel: (n_pairs, n_points, 3), dir_source_detector: (n_pairs, 3) -> d_proj: (n_pairs, n_points)
    d_proj = torch.sum(pos_rel * dir_source_detector[:, None, :], dim=2)
    
    # Vector from each point to its projection on the line
    vec_perp = pos_rel - d_proj[:, :, None] * dir_source_detector[:, None, :]
    
    # z is the magnitude of the perpendicular vector
    z = torch.norm(vec_perp, dim=2)  # (n_pairs, n_points)
    
    return z


def cw_sensitivity(pos: torch.Tensor, source_pos: torch.Tensor, detector_pos: torch.Tensor, mu_eff: float) -> torch.Tensor:
    """
    Calculate the continuous wave sensitivity function using PyTorch on GPU.
    The sensitivity function is with respect to mu_a at pos.
    
    From eq (14.8) in the textbook Quantitative Biomedical Optics by Bigio and Fantini (p. 427).

    Parameters
    ----------
    pos : torch.Tensor
        The positions to calculate sensitivity for. Shape (n_points, 3).
    source_pos : torch.Tensor
        The source positions. Shape (n_pairs, 3).
    detector_pos : torch.Tensor
        The detector positions. Shape (n_pairs, 3).
    mu_eff : float
        The effective attenuation coefficient in mm^-1.

    Returns
    -------
    sensitivity : torch.Tensor
        The sensitivity function. Shape (n_pairs, n_points).
    """
    # Calculate distances
    # pos: (n_points, 3), source_pos: (n_pairs, 3) -> d_source_pos: (n_pairs, n_points)
    d_source_pos = torch.norm(source_pos[:, None, :] - pos[None, :, :], dim=2)
    d_detector_pos = torch.norm(detector_pos[:, None, :] - pos[None, :, :], dim=2)

    # Compute perpendicular distance z
    z = compute_perpendicular_distance(pos, source_pos, detector_pos)
    
    # Calculate sensitivity
    sensitivity = (
        z ** 2
        * (mu_eff + 1 / d_source_pos)
        * (mu_eff + 1 / d_detector_pos)
        * torch.exp(-mu_eff * (d_source_pos + d_detector_pos))
        / (d_source_pos**2 * d_detector_pos**2)
    )

    return sensitivity


def get_valid_source_detector_pairs(sensor_positions_mm: torch.Tensor, max_dist: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get all valid source-detector pairs that satisfy distance criteria using PyTorch.
    
    Parameters
    ----------
    sensor_positions_mm : torch.Tensor
        Sensor positions with shape (n_sensors, 3).
    max_dist : float
        Maximum allowed distance between source and detector.
        
    Returns
    -------
    sources : torch.Tensor
        Source positions for valid pairs. Shape (n_pairs, 3).
    detectors : torch.Tensor
        Detector positions for valid pairs. Shape (n_pairs, 3).
    """
    # Calculate pairwise distances
    n_sensors = sensor_positions_mm.shape[0]
    d_mat = torch.norm(
        sensor_positions_mm[:, None, :] - sensor_positions_mm[None, :, :],
        dim=2
    )
    
    # Find valid pairs (different sensors and within max distance)
    mask = (d_mat <= max_dist) & (d_mat > 0)
    src_idx, det_idx = torch.nonzero(mask, as_tuple=True)
    
    sources = sensor_positions_mm[src_idx]
    detectors = sensor_positions_mm[det_idx]
    
    return sources, detectors


