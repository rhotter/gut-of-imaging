import os
import numpy as np
from pathlib import Path


np.random.seed(239)

BRAIN_RADIUS = 80  # mm
SKULL_RADIUS = 86
SCALP_RADIUS = 92

AIR_CONDUCTIVITY = 0
SCALP_CONDUCTIVITY = 1
BRAIN_CONDUCTIVITY = 1
SKULL_CONDUCTIVITY = 0.03

N_SOURCES_DEFAULT = 100
N_SENSORS_DEFAULT = 100


def get_source_positions(n_sources: int = N_SOURCES_DEFAULT) -> np.ndarray:
    # sample random positions inside a hemisphere
    positions = np.zeros((n_sources, 3))
    count = 0

    while count < n_sources:
        # Sample random point in a cube with side length 2*BRAIN_RADIUS
        x = np.random.uniform(0, BRAIN_RADIUS * 2)
        y = np.random.uniform(0, BRAIN_RADIUS * 2)
        z = np.random.uniform(0, BRAIN_RADIUS * 2)  # Only positive z for hemisphere

        # Check if point is inside the hemisphere
        distance = np.sqrt((x - BRAIN_RADIUS) ** 2 + (y - BRAIN_RADIUS) ** 2 + z**2)

        if distance <= BRAIN_RADIUS:
            positions[count] = [x, y, z]

            # # Vector from center of brain to the point
            # center = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0])
            # direction = np.array([x, y, z]) - center
            
            # # Normalize the direction vector and scale to brain radius
            # normalized_direction = direction / distance
            
            # # Place the point on the brain surface
            # positions[count] = center + normalized_direction * BRAIN_RADIUS
            count += 1
    return positions


def get_sensor_positions(n_sensors: int = N_SENSORS_DEFAULT) -> np.ndarray:
    # sample random positions on a hemisphere
    positions = np.random.randn(n_sensors, 3)
    positions[:, 2] = np.abs(positions[:, 2])
    positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    positions = positions * SCALP_RADIUS + np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0])
    return positions


def get_sensor_positions_spiral(n_sensors: int = N_SENSORS_DEFAULT, offset: float = 0, start_n: int = 0, end_n: int | None = None) -> np.ndarray:
    # Deterministic uniform sampling on a hemisphere using a spherical Fibonacci spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    indices = np.arange(n_sensors)
    # z coordinates uniformly spaced in [0,1)
    z = (indices + 0.5) / n_sensors
    # polar angle
    theta = np.arccos(z)
    # azimuthal angle using golden angle
    phi = golden_angle * indices
    # convert spherical to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    # unit hemisphere points
    positions = np.stack([x, y, z], axis=1)
    # scale to SCALP_RADIUS and translate to center at (BRAIN_RADIUS, BRAIN_RADIUS, 0)
    positions = positions * (SCALP_RADIUS + offset) + np.array([SCALP_RADIUS, SCALP_RADIUS, 0])
    return positions[start_n:end_n]


def get_voxel_mask(resolution: float = 1, offset: float = 0) -> np.ndarray:
    # create a voxel mask for the brain
    # the mask is a 3D array of size (nx, ny, nz)
    # the mask is 1 for the brain, 2 for the skull, 3 for the scalp and 0 for the rest
    radius = SCALP_RADIUS + offset
    nx = int(2 * radius / resolution)
    ny = int(2 * radius / resolution)
    nz = int(radius / resolution)
    mask = np.zeros((nx, ny, nz))

    # Create coordinate grids
    x = np.linspace(-radius, radius, nx)
    y = np.linspace(-radius, radius, ny)
    z = np.linspace(0, radius, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Calculate distances from origin for all points at once
    distances = np.sqrt(X**2 + Y**2 + Z**2)

    # Set mask values based on distances
    mask[distances <= BRAIN_RADIUS] = 1
    mask[(distances > BRAIN_RADIUS) & (distances <= SKULL_RADIUS)] = 2
    mask[(distances > SKULL_RADIUS) & (distances <= radius)] = 3

    return mask


def get_source_positions_halton(n_sources: int = N_SOURCES_DEFAULT) -> np.ndarray:
    # Deterministic uniform sampling inside a hemisphere via a low-discrepancy sequence
    def van_der_corput(num: int, base: int = 2) -> float:
        vdc, denom = 0.0, 1
        while num > 0:
            num, rem = divmod(num, base)
            denom *= base
            vdc += rem / denom
        return vdc

    positions = np.zeros((n_sources, 3))
    for i in range(n_sources):
        # radial coordinate via uniform volume sampling
        u = (i + 0.5) / n_sources
        r = BRAIN_RADIUS * u ** (1 / 3)
        # direction via Hammersley sequence (bases 2 and 3)
        theta = np.arccos(van_der_corput(i, 3))
        phi = 2 * np.pi * van_der_corput(i, 2)
        x = r * np.sin(theta) * np.cos(phi) + BRAIN_RADIUS
        y = r * np.sin(theta) * np.sin(phi) + BRAIN_RADIUS
        z = r * np.cos(theta)
        positions[i] = [x, y, z]
    return positions


def create_hemisphere(radius, n_phi=8, n_theta=8):
    """Create a hemisphere mesh.
    
    Parameters
    ----------
    radius : float
        Radius of the hemisphere
    n_phi : int
        Number of points in the azimuthal direction
    n_theta : int
        Number of points in the polar direction (hemisphere: 0 to π/2)
    
    Returns
    -------
    vertices : ndarray
        Vertex coordinates
    triangles : ndarray
        Triangle indices (0-based)
    """
    # Generate grid of points in spherical coordinates
    # For a hemisphere, theta goes from 0 to π/2
    theta = np.linspace(0, np.pi/2, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    
    # Create vertices
    vertices = []
    
    # Add the pole point at the top of the hemisphere
    vertices.append([0, 0, radius])
    
    # Add vertices for the rest of the hemisphere
    for t in theta[1:]:  # Skip the first theta (pole point already added)
        for p in phi[:-1]:  # Skip the last phi (duplicate of phi=0)
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.sin(t) * np.sin(p)
            z = radius * np.cos(t)
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    # Create triangles
    triangles = []
    
    # Number of unique phi points
    n_phi_actual = n_phi - 1
    
    # Create triangles connecting the pole to the first ring
    for i in range(n_phi_actual):
        v1 = 0  # Pole vertex
        v2 = i + 1
        v3 = (i + 1) % n_phi_actual + 1
        triangles.append([v1, v2, v3])
    
    # Create triangles for the rest of the hemisphere
    for i in range(n_theta - 2):  # -2 because we've handled the top row separately
        row_start = 1 + i * n_phi_actual
        next_row_start = 1 + (i + 1) * n_phi_actual
        
        for j in range(n_phi_actual):
            v1 = row_start + j
            v2 = row_start + (j + 1) % n_phi_actual
            v3 = next_row_start + j
            v4 = next_row_start + (j + 1) % n_phi_actual
            
            # Add two triangles for each quad
            triangles.append([v1, v2, v3])
            triangles.append([v2, v4, v3])
    
    triangles = np.array(triangles)
    
    return vertices, triangles

def write_tri(filename, vertices, triangles):
    """Write a .tri file following the BrainVisa format.
    
    Parameters
    ----------
    filename : str
        Path to the output file
    vertices : array
        Vertex coordinates (N, 3)
    triangles : array
        Triangle indices (M, 3), must use 0-based indexing
    """
    with open(filename, 'w') as f:
        # Write number of vertices
        f.write(f"- {len(vertices)}\n")
        
        # Write vertices with normals (normals = normalized vertex positions for a sphere)
        for v in vertices:
            # Calculate normal (just normalize the position vector for a sphere)
            n = v / np.linalg.norm(v)
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f} {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n")
        
        # Write number of triangles (repeated three times as per format)
        f.write(f"- {len(triangles)} {len(triangles)} {len(triangles)}\n")
        
        # Write triangles with 0-based indexing
        for t in triangles:
            f.write(f"{t[0]} {t[1]} {t[2]}\n")

def get_random_orientations(n_sources: int) -> np.ndarray:
    """Generate random unit vectors for dipole orientations.
    
    Parameters
    ----------
    n_sources : int
        Number of dipole orientations to generate
        
    Returns
    -------
    orientations : ndarray of shape (n_sources, 3)
        Random unit vectors representing dipole orientations
    """
    # Generate random vectors
    orientations = np.random.randn(n_sources, 3)
    # Normalize to unit vectors
    norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    orientations = orientations / norms
    return orientations

def create_bem_model():
    """Create a 3-layer hemispherical model."""
    # Ensure model directory exists
    os.makedirs('bem_model/', exist_ok=True)
    
    # Create the three hemispherical meshes
    for name, radius in [('brain', BRAIN_RADIUS), ('skull', SKULL_RADIUS), ('scalp', SCALP_RADIUS)]:
        vertices, triangles = create_hemisphere(radius, n_phi=16, n_theta=8)
        write_tri(f"bem_model/{name}_hemi.tri", vertices, triangles)
    
    # Create the geometry file
    with open('bem_model/hemi_head.geom', 'w') as f:
        f.write("# Domain Description 1.0\n\n")
        f.write("Interfaces 3\n\n")
        f.write("Interface Brain: \"brain_hemi.tri\"\n")
        f.write("Interface Skull: \"skull_hemi.tri\"\n")
        f.write("Interface Scalp: \"scalp_hemi.tri\"\n\n")
        f.write("Domains 4\n\n")
        f.write("Domain Brain: -Brain\n")
        f.write("Domain Skull: -Skull +Brain\n")
        f.write("Domain Scalp: -Scalp +Skull\n")
        f.write("Domain Air: +Scalp\n")
    
    # Create the conductivity file
    with open('bem_model/hemi_head.cond', 'w') as f:
        f.write("# Properties Description 1.0 (Conductivities)\n\n")
        f.write(f"Air         {AIR_CONDUCTIVITY}\n")
        f.write(f"Scalp       {SCALP_CONDUCTIVITY}\n")
        f.write(f"Brain       {BRAIN_CONDUCTIVITY}\n")
        f.write(f"Skull       {SKULL_CONDUCTIVITY}\n")
    
    # Generate dipole positions and orientations
    positions = get_source_positions_halton(N_SOURCES_DEFAULT)
    orientations = get_random_orientations(N_SOURCES_DEFAULT)
    
    # Write dipoles to file
    with open('bem_model/cortex_dipoles.txt', 'w') as f:
        for pos, ori in zip(positions, orientations):
            f.write(f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{ori[0]:.6f}\t{ori[1]:.6f}\t{ori[2]:.6f}\n")
