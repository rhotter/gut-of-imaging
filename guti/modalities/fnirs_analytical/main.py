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
print("Finding valid source-detector pairs...")
sources, detectors = get_valid_source_detector_pairs(sensor_positions_torch, max_dist)
print(f"Found {len(sources)} valid source-detector pairs")

# Calculate sensitivities for all pairs and grid points in one vectorized operation
print("Calculating sensitivities...")
sensitivities = cw_sensitivity(grid_points_torch, sources, detectors, mu_eff)
print(f"Sensitivities shape: {sensitivities.shape}")

# %%
sensitivities.cpu().numpy().max()
# %%
# check sensor positions
import plotly.graph_objects as go

# Create 3D scatter plot with Plotly
fig = go.Figure()

# Add grid points trace
fig.add_trace(go.Scatter3d(
    x=grid_points_mm[:, 0],
    y=grid_points_mm[:, 1], 
    z=grid_points_mm[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color='blue',
        opacity=0.3
    ),
    name='Grid Points'
))

# Add sensor positions trace
fig.add_trace(go.Scatter3d(
    x=sensor_positions_mm[:, 0],
    y=sensor_positions_mm[:, 1],
    z=sensor_positions_mm[:, 2], 
    mode='markers',
    marker=dict(
        size=5,
        color='red',
        opacity=0.8
    ),
    name='Sensors'
))

# Update layout
fig.update_layout(
    title='3D Grid Points and Sensor Positions',
    scene=dict(
        xaxis_title='X (mm)',
        yaxis_title='Y (mm)', 
        zaxis_title='Z (mm)',
        aspectmode='cube'
    ),
    width=800,
    height=600
)

fig.show()

# Print some information
print(f"Number of grid points: {len(grid_points_mm)}")
print(f"Number of sensors: {len(sensor_positions_mm)}")
print(f"Grid points shape: {grid_points_mm.shape}")
print(f"Sensor positions shape: {sensor_positions_mm.shape}")
print(f"Number of valid pairs: {len(sources)}")
print(f"Sensitivities shape: {sensitivities.shape}")

