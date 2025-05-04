
"""
Simple CT simulation & reconstruction utilities
===============================================

This module depends on *scikit‑image* (``pip install scikit-image``).

Workflow
--------
1. Build a 2‑D attenuation map from the brain geometry in ``guti.core``:
   • brain voxels get random μ values  
   • skull, scalp, and air get fixed coefficients  
2. Generate a parallel‑beam sinogram by forward‑projecting the slice.  
3. Reconstruct the slice via filtered back‑projection (FBP).

Example
-------
Run this file directly to see a demo::

    python -m guti.modalities.ct.ct
"""
# %%
from __future__ import annotations

import numpy as np
from skimage.transform import radon, iradon

from guti.core import get_voxel_mask

# ----------------------------------------------------------------------
#  Linear attenuation coefficients (unit: mm⁻¹, but only relative matters)
# ----------------------------------------------------------------------
MU_AIR: float = 0.00
MU_SCALP: float = 0.15
MU_SKULL: float = 0.50
MU_BRAIN_MEAN: float = 0.20
MU_BRAIN_STD: float = 0.02


# ----------------------------------------------------------------------
def build_attenuation_map(
    resolution: float = 1.0,
    slice_index: int | None = None,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Create a 2‑D attenuation map (μ) for an axial slice.

    Parameters
    ----------
    resolution
        Spatial resolution (mm/voxel) passed to :func:`guti.core.get_voxel_mask`.
    slice_index
        z‑index of the slice (defaults to the central slice).
    rng
        Seed or :class:`numpy.random.Generator` for sampling brain μ values.

    Returns
    -------
    ndarray
        2‑D array (ny, nx) containing linear attenuation coefficients.
    """
    mask = get_voxel_mask(resolution)
    if slice_index is None:
        slice_index = mask.shape[2] // 2  # middle axial slice

    m2d = mask[:, :, slice_index].astype(np.int8)

    rng = np.random.default_rng(rng)
    mu = np.empty_like(m2d, dtype=np.float32)

    mu[m2d == 0] = MU_AIR
    mu[m2d == 3] = MU_SCALP
    mu[m2d == 2] = MU_SKULL

    brain_idx = m2d == 1
    mu[brain_idx] = rng.normal(MU_BRAIN_MEAN, MU_BRAIN_STD, size=brain_idx.sum())

    return mu


# ----------------------------------------------------------------------
def simulate_projections(
    mu: np.ndarray,
    n_angles: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a parallel‑beam sinogram (Radon transform) for a slice.

    Parameters
    ----------
    mu
        2‑D attenuation map produced by :func:`build_attenuation_map`.
    n_angles
        Number of projection angles spanning 180°.

    Returns
    -------
    sinogram
        2‑D array (n_detectors, n_angles) of line integrals.
    theta
        1‑D array of projection angles in degrees.
    """
    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    sinogram = radon(mu, theta=theta, circle=False)
    return sinogram, theta


# ----------------------------------------------------------------------
def reconstruct(
    sinogram: np.ndarray,
    theta: np.ndarray,
    filter_name: str = "ramp",
) -> np.ndarray:
    """
    Reconstruct a slice from its sinogram via filtered back‑projection.

    Parameters
    ----------
    sinogram
        Sinogram returned by :func:`simulate_projections`.
    theta
        Projection angles corresponding to the sinogram columns.
    filter_name
        Frequency filter (``'ramp'``, ``'shepp‑logan'``, ``'hann'``, ...).

    Returns
    -------
    ndarray
        Reconstructed 2‑D attenuation map.
    """
    return iradon(sinogram, theta=theta, filter_name=filter_name, circle=False)


# ----------------------------------------------------------------------
def demo(
    resolution: float = 1.0,
    n_angles: int = 360,
    rng: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience routine: forward project & reconstruct a slice.

    Returns
    -------
    truth, sinogram, recon : (ndarray, ndarray, ndarray)
    """
    truth = build_attenuation_map(resolution=resolution, rng=rng)
    sinogram, theta = simulate_projections(truth, n_angles=n_angles)
    recon = reconstruct(sinogram, theta)
    return truth, sinogram, recon


# %%
# Quick visual sanity check
truth, sino, recon = demo()

import matplotlib.pyplot as plt  # Lazy import for demo only

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(truth, cmap="gray")
ax[0].set_title("Ground truth μ")
ax[0].axis("off")

ax[1].imshow(sino, cmap="gray", aspect="auto")
ax[1].set_title("Sinogram")
ax[1].axis("off")

ax[2].imshow(recon, cmap="gray")
ax[2].set_title("Reconstruction")
ax[2].axis("off")

plt.tight_layout()
plt.show()
# %%
