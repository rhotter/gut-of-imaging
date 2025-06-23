# WOO
# %%

import torch

def simulate_free_field_propagation(
    source_positions: torch.Tensor,
    receiver_positions: torch.Tensor, 
    source_signals: torch.Tensor,
    time_step: float,
    center_frequency: float,
    voxel_size: torch.Tensor,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Simulates free field propagation using a free field propagator.
    
    Args:
        source_positions: Source positions in voxel coordinates [n_sources, 3]
        receiver_positions: Receiver positions in voxel coordinates [n_receivers, 3]
        source_signals: Source waveforms [n_sources, n_time_steps]
        time_step: Time step in seconds
        center_frequency: Base frequency in Hz
        voxel_size: Voxel size in meters [3]
        device: Device to run computation on
        
    Returns:
        pressure_field: Pressure at receivers [n_receivers, n_time_steps]
    """
    # Free field medium properties (water-like)
    sound_speed = 1500.0  # Speed of sound in m/s
    
    # Move tensors to device
    source_positions = source_positions.to(device)
    receiver_positions = receiver_positions.to(device)
    source_signals = source_signals.to(device)
    voxel_size = voxel_size.to(device)
    
    # Calculate distances between all source-receiver pairs
    distances = torch.cdist(
        receiver_positions.float().unsqueeze(0) * voxel_size, 
        source_positions.float().unsqueeze(0) * voxel_size
    )[0]

    # Check for zero distances which would cause division by zero
    zero_distances = distances == 0
    if torch.any(zero_distances):
        print("Found zero distances which would cause division by zero")
        # Replace zeros with small epsilon to avoid division by zero
        distances = torch.where(zero_distances, torch.tensor(1e-10, device=device), distances)
    
    # Calculate propagator factor
    wavelength = sound_speed / center_frequency
    wavenumber = 2 * torch.pi / wavelength
    spatial_step = torch.mean(voxel_size)
    propagator_factor = (2 * wavenumber * spatial_step**2) / (4 * torch.pi * distances)
    
    # Calculate retardation times and time steps
    travel_times = distances / sound_speed
    delay_steps = (torch.floor(travel_times / time_step)).int()
    
    num_sources = source_signals.shape[0]
    num_receivers = receiver_positions.shape[0]
    num_time_steps = source_signals.shape[1]
    
    # Pad source waveforms with zeros at the beginning
    padded_source_signals = torch.cat([
        torch.zeros(num_sources, 1, device=device, dtype=source_signals.dtype), 
        source_signals
    ], dim=1)
    
    # Create source indices for broadcasting
    source_idx = torch.arange(num_sources).unsqueeze(0).expand(num_receivers, num_sources).to(device)
    
    # Vectorized computation across all time steps
    # Create time indices for all receivers and time steps
    t_grid = torch.arange(num_time_steps, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, num_time_steps]
    delay_grid = delay_steps.unsqueeze(-1)  # [num_receivers, num_sources, 1]
    
    # Calculate time indices for all combinations
    time_indices = t_grid - delay_grid + 1  # [num_receivers, num_sources, num_time_steps]
    time_indices = torch.clamp(time_indices, min=0, max=padded_source_signals.shape[1] - 1)
    
    # Gather delayed signals for all time steps at once
    source_idx_expanded = source_idx.unsqueeze(-1).expand(-1, -1, num_time_steps)
    delayed_signals = padded_source_signals[source_idx_expanded, time_indices]
    
    # Apply propagator factors
    propagator_expanded = propagator_factor.unsqueeze(-1)  # [num_receivers, num_sources, 1]
    weighted_signals = delayed_signals * propagator_expanded
    pressure_field = weighted_signals
    
    return pressure_field

# %%

from guti.modalities.us.utils import create_medium, create_sources, create_receivers, plot_medium, find_arrival_time

domain, medium_original, time_axis, brain_mask, skull_mask, scalp_mask = create_medium()

sources, source_mask = create_sources(domain, time_axis, freq_Hz=1e6, n_sources=4000, pad=0, inside=True)
sensors, sensors_all, receivers_mask = create_receivers(domain, time_axis, freq_Hz=1e6, n_sensors=4000, pad=0)

# %%

import numpy as np

source_positions = np.stack([np.array(x) for x in sources.positions]).T

sensor_positions = np.stack([np.array(x) for x in sensors.positions]).T

n_sources = source_positions.shape[0]

time_step = 1e-1 / 1e6
center_frequency = 1e6
time_duration = 120e-6
time_axis = np.arange(0, time_duration, time_step)
source_signals = np.sin(2 * np.pi * time_axis * center_frequency)
source_signals = np.tile(source_signals, (n_sources, 1))

voxel_size = np.array(domain.dx)

print(f"Coinciding source and sensor positions: {torch.argwhere(torch.all(torch.tensor(source_positions).unsqueeze(0) == torch.tensor(sensor_positions).unsqueeze(1), dim=-1))}")

#%%

pressure_field = simulate_free_field_propagation(
    torch.tensor(source_positions),
    torch.tensor(sensor_positions),
    torch.tensor(source_signals),
    time_step,
    center_frequency,
    torch.tensor(voxel_size)
) # [n_sensors, n_sources, n_time_steps]

# %%

# matrix = pressure_field.permute(0,2,1).reshape(-1, source_signals.shape[0])
# The matrix below should have the same singular values, but be easier to compute
matrix = pressure_field.max(dim=2).values

# %%

s = np.linalg.svd(np.array(matrix).astype(np.float64), compute_uv=False)

# %%

import matplotlib.pyplot as plt
from guti.data_utils import save_svd

plt.semilogy(s)

save_svd(s, 'us_analytical')

# %%
