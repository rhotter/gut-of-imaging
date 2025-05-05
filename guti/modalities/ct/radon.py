"""
Radon transform matrix construction for CT imaging
================================================

This module provides utilities for building the system matrix for the Radon transform,
which represents how each pixel in the image contributes to each detector measurement
in a parallel-beam CT system.
"""

# %%
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def build_radon_matrix(
    image_size: int,
    n_angles: int,
    detector_size: int | None = None,
) -> np.ndarray:
    """
    Build the system matrix for the Radon transform in parallel-beam CT.
    
    This matrix represents how each pixel in the image contributes to each detector
    measurement. The matrix shape is (n_angles * detector_size, image_size * image_size).
    
    Parameters
    ----------
    image_size : int
        Size of the square image (number of pixels in each dimension)
    n_angles : int
        Number of projection angles spanning 180 degrees
    detector_size : int, optional
        Number of detector elements. If None, uses image_size * sqrt(2) rounded up
        to ensure the entire image is covered at all angles.
        
    Returns
    -------
    np.ndarray
        System matrix of shape (n_angles * detector_size, image_size * image_size)
        where each row represents a single detector measurement and each column
        represents a pixel in the image.
    """
    if detector_size is None:
        # Ensure detector covers the entire image at all angles
        detector_size = int(np.ceil(image_size * np.sqrt(2)))
    
    # Create coordinate grids for the image
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    X, Y = np.meshgrid(x, y)
    
    # Create coordinate grid for the detector
    detector_positions = np.linspace(-1, 1, detector_size)
    
    # Create angle grid
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    
    # Initialize the system matrix
    matrix_size = n_angles * detector_size
    image_pixels = image_size * image_size
    A = np.zeros((matrix_size, image_pixels))
    
    # Build the matrix row by row
    for i, angle in enumerate(angles):
        # Calculate projection coordinates for this angle
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # For each detector position
        for j, detector_pos in enumerate(detector_positions):
            row_idx = i * detector_size + j
            
            # Calculate the contribution of each pixel to this detector measurement
            # This is the line integral along the ray
            ray = detector_pos - (X * cos_theta + Y * sin_theta)
            
            # Use a Gaussian kernel to approximate the line integral
            # This creates a smooth transition between pixels
            sigma = 1.0 / image_size  # Adjust this parameter to control the width of the line
            weights = np.exp(-0.5 * (ray / sigma) ** 2)
            weights = weights / np.sum(weights)  # Normalize
            
            # Store the weights in the matrix
            A[row_idx, :] = weights.flatten()
    
    return A


def apply_radon_matrix(
    A: np.ndarray,
    image: np.ndarray,
    image_size: int,
    n_angles: int,
    detector_size: int | None = None,
) -> np.ndarray:
    """
    Apply the Radon transform matrix to an image.
    
    Parameters
    ----------
    A : np.ndarray
        The Radon transform matrix
    image : np.ndarray
        Input image of shape (image_size, image_size)
    image_size : int
        Size of the square image
    n_angles : int
        Number of projection angles
    detector_size : int, optional
        Number of detector elements
        
    Returns
    -------
    np.ndarray
        Sinogram of shape (detector_size, n_angles)
    """
    if detector_size is None:
        detector_size = int(np.ceil(image_size * np.sqrt(2)))
    
    # Flatten the image
    x = image.flatten()
    
    # Apply the matrix
    sino_flat = A @ x
    
    # Reshape to sinogram format
    return sino_flat.reshape(n_angles, detector_size).T


def compute_svd_spectrum(
    image_size: int,
    n_angles: int,
    detector_size: int | None = None,
) -> np.ndarray:
    """
    Compute the singular value spectrum of the Radon transform matrix.
    
    Parameters
    ----------
    image_size : int
        Size of the square image
    n_angles : int
        Number of projection angles
    detector_size : int, optional
        Number of detector elements
        
    Returns
    -------
    np.ndarray
        Array of singular values
    """
    # Build the matrix
    A = build_radon_matrix(image_size, n_angles, detector_size)
    
    # Compute SVD
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    
    return s


def plot_svd_spectra(
    image_sizes: list[int] = [32, 64, 128],
    n_angles: int = 180,
    detector_size: int | None = 10,
) -> None:
    """
    Compute and plot SVD spectra for different image sizes.
    
    Parameters
    ----------
    image_sizes : list of int
        List of different image sizes to compare
    n_angles : int
        Number of projection angles
    detector_size : int, optional
        Number of detector elements
    """
    plt.figure(figsize=(10, 6))
    
    for size in image_sizes:
        # Compute SVD spectrum
        s = compute_svd_spectrum(size, n_angles, detector_size)
        
        # Plot singular values
        plt.semilogy(s, label=f'{size}x{size} image')
    
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title('Radon Transform Singular Value Spectra for Different Image Sizes')
    plt.grid(True)
    plt.legend()
    plt.show()


def demo_radon_matrix():
    """
    Demonstrate the Radon transform matrix construction and application.
    """
    # Parameters
    image_size = 32
    n_angles = 180
    detector_size = 45
    
    # Build the matrix
    A = build_radon_matrix(image_size, n_angles, detector_size)
    
    # Create a test image (a simple circle)
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    X, Y = np.meshgrid(x, y)
    test_image = (X**2 + Y**2 < 0.5).astype(float)
    
    # Apply the transform
    sinogram = apply_radon_matrix(A, test_image, image_size, n_angles, detector_size)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.imshow(test_image, cmap='gray')
    ax1.set_title('Test Image')
    ax1.axis('off')
    
    ax2.imshow(sinogram, cmap='gray', aspect='auto')
    ax2.set_title('Sinogram')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


# %%
# Run the demo
demo_radon_matrix()

# Plot SVD spectra for different image sizes
plot_svd_spectra()

# %%
