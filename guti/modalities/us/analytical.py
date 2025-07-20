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
    device: str = "cpu",
    compute_time_series: bool = False,
    temporal_sampling: int = 1,
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
        compute_time_series: If True, returns the time-resolved pressure field
            sampled every ``temporal_sampling`` steps with shape
            [n_receivers, n_sources, ⌈n_time_steps/temporal_sampling⌉].
            If False (default), returns only the spatial propagator matrix
            [n_receivers, n_sources].
        temporal_sampling: Positive integer stride that determines the temporal
            subsampling factor applied when ``compute_time_series`` is True. A
            value of 1 (default) keeps the original resolution, whereas larger
            values reduce memory usage by computing only every *temporal_sampling*-th
            time sample.
        
    Returns:
        Tensor of shape depending on ``compute_time_series`` (see above).
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
    
    # Calculate expected pressure based on point source formula from acoustic wave theory
    # Combines point source pressure equation p = (4πR)^(-1) * m''(t-R/c) (see Pierce's acoustics formula 4.3.8)
    # with K-wave's pressure-to-density source conversion
    # The mass source acceleration m'' is related to pressure through the wavenumber k
    # Final formula: P = 1/(4πR) * (2π/λ) * peak_pressure * dx^2
    # where R is distance, λ is wavelength, dx is voxel size
    
    # Calculate propagator factor (spatial Green's function component)
    wavelength = sound_speed / center_frequency
    wavenumber = 2 * torch.pi / wavelength
    spatial_step = torch.mean(voxel_size)
    propagator_factor = (2 * wavenumber * spatial_step**2) / (4 * torch.pi * distances)
    
    # If only the propagator matrix is required, return early
    if not compute_time_series:
        return propagator_factor * torch.exp(-1j * wavenumber * distances)
    
    # Calculate retardation times and time steps (only needed when computing full time series)
    travel_times = distances / sound_speed
    delay_steps = (torch.floor(travel_times / time_step)).int()
    
    num_sources = source_signals.shape[0]
    num_receivers = receiver_positions.shape[0]
    num_time_steps = source_signals.shape[1]
    
    # Determine which time indices will be computed based on temporal sampling
    if temporal_sampling < 1:
        raise ValueError("temporal_sampling must be a positive integer >= 1")

    selected_time_indices = torch.arange(0, num_time_steps, temporal_sampling, device=device)
    
    # Pad source waveforms with zeros at the beginning
    padded_source_signals = torch.cat([
        torch.zeros(num_sources, 1, device=device, dtype=source_signals.dtype), 
        source_signals
    ], dim=1)
    
    # Create source indices for broadcasting
    source_idx = torch.arange(num_sources).unsqueeze(0).expand(num_receivers, num_sources).to(device)
    
    # ------------------------------------------------------------------
    # Serial computation across the selected time steps to save memory
    # ------------------------------------------------------------------
    n_selected = selected_time_indices.shape[0]
    pressure_field = torch.empty(
        (num_receivers, num_sources, n_selected),
        dtype=padded_source_signals.dtype,
        device=device,
    )

    for idx, t_idx in enumerate(selected_time_indices):
        # Compute delayed time indices for this specific time step
        time_idx_matrix = t_idx - delay_steps + 1
        time_idx_matrix = torch.clamp(time_idx_matrix, min=0, max=padded_source_signals.shape[1] - 1)

        # Gather delayed signals for the current time step
        delayed_signals_step = padded_source_signals[source_idx, time_idx_matrix]

        # Apply propagator factor and store in output tensor
        pressure_field[:, :, idx] = delayed_signals_step * propagator_factor

    # Return the serially computed pressure field
    return pressure_field

# %%

from guti.modalities.us.utils import create_medium, create_sources, create_receivers, plot_medium, find_arrival_time

domain, medium_original, time_axis, brain_mask, skull_mask, scalp_mask = create_medium()

center_frequency = 1.5e6
sources, source_mask = create_sources(domain, time_axis, freq_Hz=1e6, n_sources=8000, pad=0, inside=True)
sensors, sensors_all, receivers_mask = create_receivers(domain, time_axis, freq_Hz=center_frequency, n_sensors=8000, pad=0)

# %%

import numpy as np

source_positions = np.stack([np.array(x) for x in sources.positions]).T

sensor_positions = np.stack([np.array(x) for x in sensors.positions]).T

n_sources = source_positions.shape[0]

time_step = 1e-1 / center_frequency
time_duration = 120e-6
time_axis = np.arange(0, time_duration, time_step)
source_signals = np.sin(2 * np.pi * time_axis * center_frequency)
source_signals = np.tile(source_signals, (n_sources, 1))

voxel_size = np.array(domain.dx)

print(f"Coinciding source and sensor positions: {torch.argwhere(torch.all(torch.tensor(source_positions).unsqueeze(0) == torch.tensor(sensor_positions).unsqueeze(1), dim=-1))}")

#%%

# device = "cuda"
device = "cpu"

temporal_sampling = 5
# temporal_sampling = 1

# use_complex_ampitudes = True
use_complex_ampitudes = False

pressure_field = simulate_free_field_propagation(
    torch.tensor(source_positions).to(device),
    torch.tensor(sensor_positions).to(device),
    torch.tensor(source_signals).to(device),
    time_step,
    center_frequency,
    torch.tensor(voxel_size).to(device),
    device=device,
    compute_time_series=not use_complex_ampitudes,
    temporal_sampling=temporal_sampling
) # [n_sensors, n_sources]

# %%

pressure_field = pressure_field.reshape(-1, n_sources)

# IF WE WANTED TO COMPUTE PHASES
if use_complex_ampitudes:
    matrix = torch.cat([pressure_field.real, pressure_field.imag], dim=0).float()
else:
    matrix = pressure_field.float()  # already [n_receivers, n_sources]

# %%

# maybe can use something like this to simulate Born's approximation (see https://ausargeo.com/deepwave/scalar_born)
# matrix = matrix * matrix

matrix = matrix.cuda()

smaller_matrix = matrix.T @ matrix

# s = torch.linalg.svdvals(matrix, driver="gesvd").cpu().numpy()
s = torch.linalg.svdvals(smaller_matrix)

s = torch.sqrt(s)

s = s.cpu().numpy()

# %%

import matplotlib.pyplot as plt
from guti.data_utils import save_svd

plt.semilogy(s)

save_svd(s, 'us_analytical')

# %%
