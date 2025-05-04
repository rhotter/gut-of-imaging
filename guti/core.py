import numpy as np


np.random.seed(239)

brain_radius = 80   # mm
skull_radius = 86
scalp_radius = 92

n_sources_default = 100
n_sensors_default = 100

def get_source_positions(n_sources: int = n_sources_default) -> np.ndarray:
    # sample random positions inside a hemisphere
    # Generate random positions in a cube and keep only those inside the hemisphere
    positions = np.zeros((n_sources, 3))
    count = 0
    
    while count < n_sources:
        # Sample random point in a cube with side length 2*brain_radius
        x = np.random.uniform(0, brain_radius * 2)
        y = np.random.uniform(0, brain_radius * 2)
        z = np.random.uniform(0, brain_radius * 2)  # Only positive z for hemisphere
        
        # Check if point is inside the hemisphere
        distance_from_origin = np.sqrt((x - brain_radius) **2 + (y - brain_radius)**2 + z**2)
        
        if distance_from_origin <= brain_radius:
            positions[count] = [x, y, z]
            count += 1
    return positions


def get_sensor_positions(n_sensors: int = n_sensors_default) -> np.ndarray:
    # sample random positions on a hemisphere
    positions = np.random.randn(n_sensors, 3)
    positions[:, 2] = np.abs(positions[:, 2])
    positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    positions = positions * scalp_radius + np.array([brain_radius, brain_radius, 0])
    return positions


def get_sensor_positions_spiral(n_sensors: int = n_sensors_default) -> np.ndarray:
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
    # scale to scalp_radius and translate to center at (brain_radius, brain_radius, 0)
    positions = positions * scalp_radius + np.array([brain_radius, brain_radius, 0])
    return positions

def get_voxel_mask(resolution: float = 1) -> np.ndarray:
    # create a voxel mask for the brain
    # the mask is a 3D array of size (nx, ny, nz)
    # the mask is 1 for the brain, 2 for the skull, 3 for the scalp and 0 for the rest
    nx = int(2 * brain_radius / resolution)
    ny = int(2 * brain_radius / resolution)
    nz = int(brain_radius / resolution)
    mask = np.zeros((nx, ny, nz))
    
    # Create coordinate grids
    x = np.linspace(-brain_radius, brain_radius, nx)
    y = np.linspace(-brain_radius, brain_radius, ny)
    z = np.linspace(0, brain_radius, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate distances from origin for all points at once
    distances = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Set mask values based on distances
    mask[distances <= brain_radius] = 1
    mask[(distances > brain_radius) & (distances <= skull_radius)] = 2
    mask[(distances > skull_radius) & (distances <= scalp_radius)] = 3
    
    return mask
