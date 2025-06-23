# %%
# %load_ext autoreload
# %autoreload 2

from guti.core import get_grid_positions, get_sensor_positions_spiral

import numpy as np
import torch
from guti.modalities.fnirs_analytical.utils import cw_sensitivity, get_valid_source_detector_pairs
from tqdm.notebook import tqdm

# %%
grid_points_mm = get_grid_positions(10)
sensor_positions_mm = get_sensor_positions_spiral(500)
mu_a = 0.1  # cm^-1
mu_s_prime = 10  # cm^-1
mu_eff = np.sqrt(3 * mu_a * (mu_s_prime + mu_a))
mu_eff = mu_eff * 1e-1  # mm^-1

max_dist = 80  # mm

# Convert to torch tensors and move to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
grid_points_torch = torch.from_numpy(grid_points_mm).float().to(device)
sensor_positions_torch = torch.from_numpy(sensor_positions_mm).float().to(device)

# %%
# Get all valid source-detector pairs
sources, detectors = get_valid_source_detector_pairs(sensor_positions_torch, max_dist)

# Calculate sensitivities for all pairs and grid points in one vectorized operation
sensitivities = cw_sensitivity(grid_points_torch, sources, detectors, mu_eff)
print(f"Sensitivities shape: {sensitivities.shape}")

# %%
sensitivities.cpu().numpy().max()

#%%
from guti.svd import compute_svd_gpu

s = compute_svd_gpu(sensitivities.cpu().numpy())

#%%
import matplotlib.pyplot as plt

plt.semilogy(s)

# %%
# check sensor positions
from guti.viz import visualize_grid_and_sensors

fig = visualize_grid_and_sensors(
    grid_points_mm, 
    sensor_positions_mm, 
)


# %%
