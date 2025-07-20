import torch
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, vmap
from jax.scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, svds

from guti.modalities.us.utils import create_medium
from guti.core import get_source_positions, get_sensor_positions_spiral
from guti.data_utils import save_svd
from guti.core import BRAIN_RADIUS, SKULL_RADIUS, SCALP_RADIUS

# ── Physical Constants (converted to correct units) ──────────────────────────
# Geometry (in mm, consistent with core.py)
R_brain = BRAIN_RADIUS      # 80 mm - inner radius (brain)
R_skull = SKULL_RADIUS      # 86 mm - outer radius (skull)  
R_scalp = SCALP_RADIUS      # 92 mm - scalp radius
h_skull = R_skull - R_brain # 6 mm - skull thickness

# Attenuation coefficients (convert dB to Neper)
# Brain: 1 dB/cm = 0.1 dB/mm → 0.1 * ln(10)/20 ≈ 0.01151 Np/mm
# Skull: 10 dB/cm = 1.0 dB/mm → 1.0 * ln(10)/20 ≈ 0.1151 Np/mm
alpha_brain = 0.01151  # Neper/mm (brain attenuation)
beta_skull = 0.1151    # Neper/mm (skull attenuation)

# Integration parameters
D_integration = 0.1    # Integration window half-width
n_integration = 50     # Grid size for integration

print(f"Using geometry: brain={R_brain}mm, skull={R_skull}mm, scalp={R_scalp}mm")
print(f"Skull thickness: {h_skull}mm")
print(f"Attenuation: brain={alpha_brain:.4f} Np/mm, skull={beta_skull:.4f} Np/mm")

# ── Coordinate conversion functions ──────────────────────────────────────────
def cartesian_to_spherical(positions, center):
    """Convert Cartesian coordinates to spherical (r, theta, phi)
    
    Args:
        positions: (N, 3) array of [x, y, z] coordinates
        center: (3,) array of center coordinates
    
    Returns:
        r: radial distance
        theta: polar angle (from z-axis)
        phi: azimuthal angle (from x-axis)
    """
    # Translate to centered coordinates
    x, y, z = (positions - center).T
    
    # Spherical coordinates
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(jnp.clip(z / (r + 1e-8), -1, 1))  # polar angle from z-axis
    phi = jnp.arctan2(y, x)  # azimuthal angle
    
    return r, theta, phi

def get_detector_angles(detector_pos, center):
    """Get gamma and phi angles for detectors as defined in analytical model
    
    Args:
        detector_pos: (N, 3) detector positions  
        center: (3,) center position
        
    Returns:
        gamma: angle from horizontal to detector direction
        phi: azimuthal angle
    """
    # Translate to centered coordinates
    rel_pos = detector_pos - center
    x, y, z = rel_pos.T
    
    # Horizontal distance from center
    r_horizontal = jnp.sqrt(x**2 + y**2)
    
    # gamma: angle from horizontal to detector (like polar angle but from horizontal plane)
    gamma = jnp.arctan2(z, r_horizontal)
    
    # phi: azimuthal angle (angle around z-axis)
    phi = jnp.arctan2(y, x)
    
    return gamma, phi

def get_source_params(source_pos, center):
    """Get source parameters R and source_angle
    
    Args:
        source_pos: (N, 3) source positions
        center: (3,) center position
        
    Returns:
        R: horizontal distance from center to source projection
        source_angle: angle from horizontal to source direction
    """
    # Translate to centered coordinates  
    rel_pos = source_pos - center
    x, y, z = rel_pos.T
    
    # R: horizontal distance from center (sqrt(x^2 + y^2))
    R = jnp.sqrt(x**2 + y**2)
    
    # source_angle: angle from horizontal to source
    source_angle = jnp.arctan2(z, R)
    phi = jnp.arctan2(y, x)
    
    return R, source_angle, phi

# ── Analytical integrand function ─────────────────────────────────────────────
@jit
def analytical_integrand(Dphi, Dgamma, gamma0, phi0, R, source_angle, phi_src):
    """Analytical integrand for ultrasound propagation through brain and skull
    
    Based on the analytical expressions from pat_analytical.py, adapted for
    the correct geometry and attenuation coefficients.
    """
    # Convert to local variables for clarity
    r = R_brain  # Inner radius (brain)
    alpha = alpha_brain
    beta = beta_skull
    h = h_skull
    phi = phi0 - phi_src
    
    # Clamp source_angle to avoid tan() singularities
    source_angle_safe = jnp.clip(source_angle, -jnp.pi/2 + 1e-6, jnp.pi/2 - 1e-6)
    t = jnp.tan(source_angle_safe)
    
    # Distance calculation (from analytical model)
    d_squared = (
        (r * jnp.sin(gamma0) - R * t)**2
        + R**2
        + (r * jnp.cos(gamma0))**2
        - 2 * r * R * jnp.cos(phi)
    )
    
    # Ensure d_squared is positive and add small epsilon for numerical stability
    d_squared = jnp.maximum(d_squared, 1e-9)
    d = jnp.sqrt(d_squared)
    
    # Add minimum threshold for d to prevent division by zero
    d_safe = jnp.maximum(d, 1e-8)
    
    # Terms for the exponential
    term1 = h**2 + (r * Dphi)**2 + (r * Dgamma)**2
    term2 = (
        d_safe
        + Dgamma * r * R * t * jnp.cos(gamma0) / d_safe
        - r * R * jnp.sin(phi) * Dphi / d_safe
    )
    
    # Exponential argument with safety checks
    expo_part1 = -beta * term1 * term2 / (d_safe**2)
    expo_part2 = alpha * (d_safe 
                         - r * R * t * jnp.cos(gamma0) / d_safe
                         + r * R * jnp.sin(phi) / d_safe)
    
    expo = expo_part1 + expo_part2
    
    # Clamp exponential argument to prevent overflow/underflow
    expo_safe = jnp.clip(expo, -500, 500)  # exp(-500) ≈ 0, exp(500) is very large but finite
    
    return jnp.exp(expo_safe)

# ── Integration function ──────────────────────────────────────────────────────
@jit
def compute_analytical_response(gamma0, phi0, R, source_angle, phi_src):
    """Compute the analytical ultrasound response for given source-detector pair"""
    # Create integration grids
    dp = jnp.linspace(-D_integration, D_integration, n_integration)
    dg = jnp.linspace(-D_integration, D_integration, n_integration)
    DP, DG = jnp.meshgrid(dp, dg, indexing='xy')
    
    # Compute integrand values
    vals = analytical_integrand(DP, DG, gamma0, phi0, R, source_angle, phi_src)
    
    # 2D trapezoidal integration using JAX's trapezoid function
    I_phi = trapezoid(vals, x=dp, axis=0)
    I_total = trapezoid(I_phi, x=dg, axis=0)
    
    return I_total

# Vectorize for multiple source-detector pairs
compute_analytical_response_batch = vmap(compute_analytical_response, in_axes=(0, 0, 0, 0, 0))

# ── Main analytical solver ────────────────────────────────────────────────────
def analytical_solver(source_positions, detector_positions, center_mm):
    """
    Compute analytical ultrasound signals for given source and detector positions
    
    Args:
        source_positions: (N_sources, 3) array of source positions in mm
        detector_positions: (N_detectors, 3) array of detector positions in mm  
        center_mm: (3,) center position in mm
        
    Returns:
        signals: (N_detectors, N_sources) array of analytical signals
    """
    N_sources = source_positions.shape[0]
    N_detectors = detector_positions.shape[0]
    
    # Convert positions to JAX arrays
    sources_jax = jnp.array(source_positions)
    detectors_jax = jnp.array(detector_positions) 
    center_jax = jnp.array(center_mm)
    
    # Get detector angles
    gamma_det, phi_det = get_detector_angles(detectors_jax, center_jax)
    
    # Get source parameters
    R_sources, source_angles, phi_sources = get_source_params(sources_jax, center_jax)
    
    # Compute all source-detector pairs
    signals = jnp.zeros((N_detectors, N_sources))
    
    for i_src in range(N_sources):
        # Broadcast source parameters to all detectors
        R_src = jnp.full(N_detectors, R_sources[i_src])
        angle_src = jnp.full(N_detectors, source_angles[i_src])
        phi_src = jnp.full(N_detectors, phi_sources[i_src])
        # Compute responses for this source to all detectors
        responses = compute_analytical_response_batch(gamma_det, phi_det, R_src, angle_src, phi_src)
        signals = signals.at[:, i_src].set(responses)
        
        if i_src % 100 == 0:
            print(f"Processed source {i_src}/{N_sources}")
    
    return signals

# ── Integration with existing framework ───────────────────────────────────────
def create_analytical_jacobian():
    """Create Jacobian matrix using analytical expressions"""
    
    print("Setting up geometry and medium...")
    domain, medium, time_axis, brain_mask, skull_mask, scalp_mask = create_medium()
    
    # Get source positions (inside brain)
    print("Creating sources...")
    source_positions_mm = get_source_positions() - np.array([R_brain, R_brain, 0])
    
    # Get detector positions (on scalp surface)  
    print("Creating detectors...")
    n_detectors = 400
    detector_positions_mm = get_sensor_positions_spiral(n_sensors=n_detectors, offset=10) - np.array([SCALP_RADIUS, SCALP_RADIUS, 0])
  
    # Compute analytical signals
    signals = analytical_solver(source_positions_mm, detector_positions_mm, np.array([0, 0, 0]))
    
    # Create Jacobian (derivative with respect to source strengths)
    # For analytical model, this is just the signal matrix
    jacobian = np.array(signals)
    
    print(f"Jacobian shape: {jacobian.shape}")
    
    return jacobian, source_positions_mm, detector_positions_mm

def extract_positions_mm(sensors_or_sources, domain):
    """Extract positions in mm from sensors or sources object"""
    # Get voxel positions  
    positions_voxels = np.array([sensors_or_sources.positions[0], 
                                sensors_or_sources.positions[1], 
                                sensors_or_sources.positions[2]]).T
    
    # Convert to mm coordinates
    dx_mm = np.array(domain.dx) * 1000  # Convert from m to mm
    positions_mm = positions_voxels * dx_mm
    
    return positions_mm

# ── Plotting functions ────────────────────────────────────────────────────────
def plot_sensor_detector_positions(source_pos, detector_pos, center_mm):
    """Plot 3D positions of sources and detectors"""
    fig = plt.figure(figsize=(15, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot sources (inside brain)
    ax1.scatter(source_pos[:, 0], source_pos[:, 1], source_pos[:, 2], 
               c='red', s=20, alpha=0.6, label='Sources (Brain)')
    
    # Plot detectors (on scalp)
    ax1.scatter(detector_pos[:, 0], detector_pos[:, 1], detector_pos[:, 2], 
               c='blue', s=20, alpha=0.6, label='Detectors (Scalp)')
    
    # Plot center
    ax1.scatter(center_mm[0], center_mm[1], center_mm[2], 
               c='black', s=100, marker='*', label='Center')
    
    # Draw brain, skull, and scalp spheres
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    # Brain sphere
    x_brain = R_brain * np.outer(np.cos(u), np.sin(v)) + center_mm[0]
    y_brain = R_brain * np.outer(np.sin(u), np.sin(v)) + center_mm[1]
    z_brain = R_brain * np.outer(np.ones(np.size(u)), np.cos(v)) + center_mm[2]
    ax1.plot_wireframe(x_brain, y_brain, z_brain, alpha=0.2, color='red', linewidth=0.5)
    
    # Scalp sphere
    x_scalp = R_scalp * np.outer(np.cos(u), np.sin(v)) + center_mm[0]
    y_scalp = R_scalp * np.outer(np.sin(u), np.sin(v)) + center_mm[1]
    z_scalp = R_scalp * np.outer(np.ones(np.size(u)), np.cos(v)) + center_mm[2]
    ax1.plot_wireframe(x_scalp, y_scalp, z_scalp, alpha=0.2, color='blue', linewidth=0.5)
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Source and Detector Positions')
    ax1.legend()
    
    # 2D projections
    # XY projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(source_pos[:, 0], source_pos[:, 1], c='red', s=20, alpha=0.6, label='Sources')
    ax2.scatter(detector_pos[:, 0], detector_pos[:, 1], c='blue', s=20, alpha=0.6, label='Detectors')
    ax2.scatter(center_mm[0], center_mm[1], c='black', s=100, marker='*', label='Center')
    
    # Draw brain and scalp circles
    theta = np.linspace(0, 2*np.pi, 100)
    brain_circle_x = R_brain * np.cos(theta) + center_mm[0]
    brain_circle_y = R_brain * np.sin(theta) + center_mm[1]
    scalp_circle_x = R_scalp * np.cos(theta) + center_mm[0]
    scalp_circle_y = R_scalp * np.sin(theta) + center_mm[1]
    
    ax2.plot(brain_circle_x, brain_circle_y, 'r--', alpha=0.5, label='Brain boundary')
    ax2.plot(scalp_circle_x, scalp_circle_y, 'b--', alpha=0.5, label='Scalp boundary')
    
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Projection')
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # XZ projection
    ax3 = fig.add_subplot(133)
    ax3.scatter(source_pos[:, 0], source_pos[:, 2], c='red', s=20, alpha=0.6, label='Sources')
    ax3.scatter(detector_pos[:, 0], detector_pos[:, 2], c='blue', s=20, alpha=0.6, label='Detectors')
    ax3.scatter(center_mm[0], center_mm[2], c='black', s=100, marker='*', label='Center')
    
    # Draw brain and scalp circles in XZ plane
    brain_circle_x = R_brain * np.cos(theta) + center_mm[0]
    brain_circle_z = R_brain * np.sin(theta) + center_mm[2]
    scalp_circle_x = R_scalp * np.cos(theta) + center_mm[0]
    scalp_circle_z = R_scalp * np.sin(theta) + center_mm[2]
    
    ax3.plot(brain_circle_x, brain_circle_z, 'r--', alpha=0.5, label='Brain boundary')
    ax3.plot(scalp_circle_x, scalp_circle_z, 'b--', alpha=0.5, label='Scalp boundary')
    
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Projection')
    ax3.axis('equal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nPosition Statistics:")
    print(f"Sources: {len(source_pos)} points")
    print(f"  - Mean position: [{np.mean(source_pos, axis=0)[0]:.1f}, {np.mean(source_pos, axis=0)[1]:.1f}, {np.mean(source_pos, axis=0)[2]:.1f}] mm")
    print(f"  - Distance from center: {np.mean(np.linalg.norm(source_pos - center_mm, axis=1)):.1f} ± {np.std(np.linalg.norm(source_pos - center_mm, axis=1)):.1f} mm")
    
    print(f"Detectors: {len(detector_pos)} points")
    print(f"  - Mean position: [{np.mean(detector_pos, axis=0)[0]:.1f}, {np.mean(detector_pos, axis=0)[1]:.1f}, {np.mean(detector_pos, axis=0)[2]:.1f}] mm")
    print(f"  - Distance from center: {np.mean(np.linalg.norm(detector_pos - center_mm, axis=1)):.1f} ± {np.std(np.linalg.norm(detector_pos - center_mm, axis=1)):.1f} mm")

def plot_coupling_matrix(jacobian, source_pos, detector_pos):
    """Plot the coupling matrix between sources and detectors"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Log scale coupling matrix (main plot)
    ax1 = axes[0]
    jacobian_positive = np.abs(jacobian) + 1e-12  # Add small value to avoid log(0)
    log_jacobian = np.log10(jacobian_positive)
    im1 = ax1.imshow(log_jacobian, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('Source Index')
    ax1.set_ylabel('Detector Index')
    ax1.set_title(f'Log10 Coupling Matrix ({jacobian.shape[0]}×{jacobian.shape[1]})')
    plt.colorbar(im1, ax=ax1, label='Log10 Coupling Strength')
    
    # Linear scale coupling matrix (for comparison)
    ax2 = axes[1]
    im2 = ax2.imshow(jacobian, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_xlabel('Source Index')
    ax2.set_ylabel('Detector Index')
    ax2.set_title('Linear Coupling Matrix')
    plt.colorbar(im2, ax=ax2, label='Coupling Strength')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nCoupling Matrix Statistics:")
    print(f"Shape: {jacobian.shape}")
    print(f"Min: {np.min(jacobian):.2e}")
    print(f"Max: {np.max(jacobian):.2e}")
    print(f"Mean: {np.mean(jacobian):.2e}")
    print(f"Std: {np.std(jacobian):.2e}")
    print(f"Number of infinite values: {np.sum(np.isinf(jacobian))}")
    print(f"Number of NaN values: {np.sum(np.isnan(jacobian))}")
    print(f"Number of finite values: {np.sum(np.isfinite(jacobian))}")
    print(f"Condition number: {np.linalg.cond(jacobian):.2e}")

# ── Debugging functions ──────────────────────────────────────────────────────
def debug_infinite_values(jacobian, source_pos, detector_pos, center_mm, n_samples=50):
    """Debug which source-detector pairs produce infinite values"""
    
    # Find infinite values
    inf_mask = np.isinf(jacobian)
    inf_indices = np.where(inf_mask)
    
    print(f"\nDebugging Infinite Values:")
    print(f"Total infinite values: {np.sum(inf_mask)}")
    print(f"Matrix shape: {jacobian.shape}")
    print(f"Percentage infinite: {100 * np.sum(inf_mask) / jacobian.size:.2f}%")
    
    if np.sum(inf_mask) == 0:
        print("No infinite values found!")
        return
    
    # Get detector and source indices for infinite values
    detector_indices = inf_indices[0]  # Row indices (detectors)
    source_indices = inf_indices[1]    # Column indices (sources)
    
    print(f"\nInfinite values found at {len(detector_indices)} positions")
    
    # Sample random subset if too many
    n_inf = len(detector_indices)
    if n_inf > n_samples:
        print(f"Sampling {n_samples} random pairs from {n_inf} infinite values...")
        sample_idx = np.random.choice(n_inf, n_samples, replace=False)
        sample_det_idx = detector_indices[sample_idx]
        sample_src_idx = source_indices[sample_idx]
    else:
        print(f"Showing all {n_inf} infinite value pairs...")
        sample_det_idx = detector_indices
        sample_src_idx = source_indices
    
    print(f"\nSource-Detector pairs producing infinite values:")
    print(f"{'Pair':<5} {'Det_Idx':<7} {'Src_Idx':<7} {'Detector Position (mm)':<25} {'Source Position (mm)':<25} {'Distance':<10}")
    print("-" * 90)
    
    for i, (det_idx, src_idx) in enumerate(zip(sample_det_idx, sample_src_idx)):
        det_pos = detector_pos[det_idx]
        src_pos = source_pos[src_idx]
        distance = np.linalg.norm(det_pos - src_pos)
        
        print(f"{i+1:<5} {det_idx:<7} {src_idx:<7} "
              f"[{det_pos[0]:6.1f},{det_pos[1]:6.1f},{det_pos[2]:6.1f}]     "
              f"[{src_pos[0]:6.1f},{src_pos[1]:6.1f},{src_pos[2]:6.1f}]     "
              f"{distance:6.1f} mm")
    
    # Analyze patterns
    print(f"\nAnalysis of infinite value pairs:")
    
    # Distance statistics
    all_inf_distances = []
    all_inf_det_pos = []
    all_inf_src_pos = []
    
    for det_idx, src_idx in zip(detector_indices, source_indices):
        det_pos = detector_pos[det_idx]
        src_pos = source_pos[src_idx]
        distance = np.linalg.norm(det_pos - src_pos)
        all_inf_distances.append(distance)
        all_inf_det_pos.append(det_pos)
        all_inf_src_pos.append(src_pos)
    
    all_inf_distances = np.array(all_inf_distances)
    all_inf_det_pos = np.array(all_inf_det_pos)
    all_inf_src_pos = np.array(all_inf_src_pos)
    
    print(f"Distance statistics for infinite pairs:")
    print(f"  Min distance: {np.min(all_inf_distances):.2f} mm")
    print(f"  Max distance: {np.max(all_inf_distances):.2f} mm")
    print(f"  Mean distance: {np.mean(all_inf_distances):.2f} mm")
    print(f"  Std distance: {np.std(all_inf_distances):.2f} mm")
    
    # Check coordinate conversion values for a few samples
    print(f"\nCoordinate conversion analysis for first 5 infinite pairs:")
    from guti.modalities.us.pat_analytical_integrated import get_detector_angles, get_source_params
    
    for i in range(min(5, len(sample_det_idx))):
        det_idx = sample_det_idx[i]
        src_idx = sample_src_idx[i]
        
        det_pos = detector_pos[det_idx:det_idx+1]  # Keep as array
        src_pos = source_pos[src_idx:src_idx+1]   # Keep as array
        
        gamma_det, phi_det = get_detector_angles(det_pos, center_mm)
        R_src, source_angle, phi_src = get_source_params(src_pos, center_mm)
        
        print(f"  Pair {i+1}: det_idx={det_idx}, src_idx={src_idx}")
        print(f"    Detector angles: gamma={gamma_det[0]:.4f}, phi={phi_det[0]:.4f}")
        print(f"    Source params: R={R_src[0]:.4f}, angle={source_angle[0]:.4f}")
        print(f"    Source angle (degrees): {np.degrees(source_angle[0]):.2f}°")
        
        # Check if source angle is close to ±90 degrees (tan singularity)
        if abs(abs(source_angle[0]) - np.pi/2) < 0.1:
            print(f"    *** WARNING: Source angle near ±90° - tan() singularity! ***")

# ── Main execution ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Computing analytical ultrasound Jacobian...")
    
    jacobian, source_pos, detector_pos = create_analytical_jacobian()
    
    # Plot sensor and detector positions
    print("\nPlotting sensor and detector positions...")
    plot_sensor_detector_positions(source_pos, detector_pos, np.array([0, 0, 0]))
    
    # Plot coupling matrix
    print("\nPlotting coupling matrix...")
    plot_coupling_matrix(jacobian, source_pos, detector_pos)

    debug_infinite_values(jacobian, source_pos, detector_pos, np.array([0, 0, 0]))
    
    # Compute SVD
    print("Computing SVD...")
    u, s, vh = np.linalg.svd(jacobian, full_matrices=False)
    
    # Plot singular value spectrum
    plt.figure(figsize=(10, 6))
    plt.semilogy(s)
    plt.grid(True)
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Value Spectrum - Analytical Ultrasound Model')
    plt.show()
    
    # Save results
    save_svd(s, 'pat_analytical')
    print(f"Saved {len(s)} singular values")
    
    
    print("Done!") 