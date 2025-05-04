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


def get_voxel_mask(resolution: float = 1) -> np.ndarray:
    # create a voxel mask for the brain
    # the mask is a 3D array of size (nx, ny, nz)
    # the mask is 1 for the brain, 2 for the skull, 3 for the scalp and 0 for the rest
    nx = int(2 * brain_radius / resolution)
    ny = int(2 * brain_radius / resolution)
    nz = int(brain_radius / resolution)
    mask = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if np.sqrt(i**2 + j**2 + k**2) <= brain_radius:
                    mask[i, j, k] = 1
                elif np.sqrt(i**2 + j**2 + k**2) <= skull_radius:
                    mask[i, j, k] = 2
                elif np.sqrt(i**2 + j**2 + k**2) <= scalp_radius:
                    mask[i, j, k] = 3

    return mask


