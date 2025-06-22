# %%
%load_ext autoreload
%autoreload 2

from guti.core import get_grid_positions, get_sensor_positions_spiral

# import plotly.graph_objects as go
import numpy as np
from guti.modalities.fnirs_analytical.utils import cw_sensitivity

# %%
grid_points_mm = get_grid_positions(10)
sensor_positions_mm = get_sensor_positions_spiral(500)
mu_a = 0.01  # cm^-1
mu_s_prime = 10  # cm^-1
mu_eff = np.sqrt(3 * mu_a * (mu_s_prime + mu_a))
mu_eff = mu_eff * 1e-1  # mm^-1

max_dist = 80  # mm

# %%
# Calculate sensitivities for all source-detector pairs and grid points
sensitivities = []
for i, source_pos in enumerate(sensor_positions_mm):
    for j, detector_pos in enumerate(sensor_positions_mm):
        if i == j:
            continue
        s_d_dist = np.linalg.norm(source_pos - detector_pos)
        if s_d_dist > max_dist:
            continue

        # Calculate sensitivity for all grid points at once
        sensitivity = cw_sensitivity(grid_points_mm, s_d_dist, mu_eff)
        sensitivities.append(sensitivity)
        break

# Convert to numpy array for easier manipulation
sensitivities = np.array(sensitivities)
