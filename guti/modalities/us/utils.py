from guti.core import get_voxel_mask, get_sensor_positions_spiral, get_source_positions
from jwave import FourierSeries, FiniteDifferences
from jwave.geometry import Domain, Medium, TimeAxis
from jwave.geometry import Sources, Sensors
import jax.numpy as jnp
import matplotlib.pyplot as plt


def create_medium():
    # Simulation parameters
    dx_mm = 0.25
    # dx_mm = 0.5
    # dx_mm = 1.5
    # dx_mm = 1.5
    dx = (dx_mm * 1e-3, dx_mm * 1e-3, dx_mm * 1e-3)

    tissues_map = get_voxel_mask(dx_mm, offset=8) #0.5mm resolution
    tissues_map = tissues_map[30:-30,30:-30,30:]
    N = tuple(tissues_map.shape)
    print(f"Domain size: {N}")
    domain = Domain(N, dx)

    # Set sound speed values based on tissue type
    # 0: outside (1500 m/s)
    # 1: brain (1525 m/s)
    # 2: skull (2400 m/s)
    # 3: scalp (1530 m/s)
    speed = jnp.ones_like(tissues_map, dtype=jnp.float32) * 1500.
    speed = jnp.where(tissues_map == 1, 1525., speed)
    speed = jnp.where(tissues_map == 2, 2400., speed)
    speed = jnp.where(tissues_map == 3, 1530., speed)
    sound_speed = FourierSeries(speed, domain)
    # Create density map with same shell mask
    # Use typical densities: ~1000 kg/m³ for water, ~2000 kg/m³ for the skull
    density = jnp.where(tissues_map == 0, 1000., 1000.)
    density = jnp.where(tissues_map == 1, 1000., 1000.)
    density = jnp.where(tissues_map == 2, 2000., 1000.)
    density = jnp.where(tissues_map == 3, 1000., 1000.)
    density_field = FourierSeries(density, domain)

    # pml_size = 20
    pml_size = 7
    # Pad the domain by the PML size to ensure proper absorption at boundaries
    domain = Domain(N, dx)

    # Update the tissue masks to match the padded domain
    medium = Medium(domain=domain, sound_speed=sound_speed, density=density_field, pml_size=pml_size)
    time_axis = TimeAxis.from_medium(medium, cfl=0.15, t_end=50e-06)
    # time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=5e-06)

    brain_mask = tissues_map == 1
    skull_mask = tissues_map == 2
    scalp_mask = tissues_map == 3

    return domain, medium, time_axis, brain_mask, skull_mask, scalp_mask

def create_sources(domain, time_axis, freq_Hz=0.25e6, inside: bool = False, n_sources: int = 400, pad: int = 30):
    """
    Create sources and source mask.
    """
    N = domain.N
    dx = domain.dx
    # Get spiral sensor positions in world coordinates
    if not inside:
        sensor_positions = get_sensor_positions_spiral(n_sensors=n_sources, offset=8)
    else:
        sensor_positions = get_source_positions(n_sources=n_sources)
    # Convert to voxel indices
    sensor_positions_voxels = jnp.floor(sensor_positions / (jnp.array(dx) * 1e3)).astype(jnp.int32)
    x_real, y_real, z_real = sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2]
    x, y, z = sensor_positions_voxels[:, 0], sensor_positions_voxels[:, 1], sensor_positions_voxels[:, 2]
    pad_x = pad_y = pad_z = pad
    # Filter positions within the padded volume
    valid_indices = (
        (x_real >= pad_x * dx[0] * 1e3) & (x_real < N[0] * dx[0] * 1e3 + pad_x * dx[0] * 1e3) &
        (y_real >= pad_y * dx[1] * 1e3) & (y_real < N[1] * dx[1] * 1e3 + pad_y * dx[1] * 1e3) &
        (z_real >= pad_z * dx[2] * 1e3) & (z_real < N[2] * dx[2] * 1e3 + pad_z * dx[2] * 1e3)
    )
    x -= pad_x; y -= pad_y; z -= pad_z
    x, y, z = x[valid_indices], y[valid_indices], z[valid_indices]
    N_sources = x.shape[0]

    # Create source signals
    signal = jnp.sin(2 * jnp.pi * freq_Hz * time_axis.to_array())
    T = 1 / freq_Hz
    max_signal_index = int(T / time_axis.dt)
    signal = signal.at[max_signal_index * 2 :].set(0)
    signals = jnp.stack([signal] * N_sources)

    # Instantiate sources
    sources = Sources(positions=(x, y, z), signals=signals, dt=time_axis.dt, domain=domain)

    # Create source mask
    source_mask = jnp.full((1,) + N, False)
    source_mask = source_mask.at[:, x, y, z].set(True)

    return sources, source_mask


def create_receivers(domain, time_axis, freq_Hz=0.25e6, n_sensors: int = 400, pad: int = 30):
    N = domain.N
    dx = domain.dx
    # Get spiral sensor positions in world coordinates
    sensor_positions = get_sensor_positions_spiral(n_sensors=n_sensors, offset=8)
    # Convert to voxel indices
    sensor_positions_voxels = jnp.floor(sensor_positions / (jnp.array(dx) * 1e3)).astype(jnp.int32)
    x_real, y_real, z_real = sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2]
    x, y, z = sensor_positions_voxels[:, 0], sensor_positions_voxels[:, 1], sensor_positions_voxels[:, 2]
    pad_x = pad_y = pad_z = pad
    # Filter positions within the padded volume
    valid_indices = (
        (x_real >= pad_x * dx[0] * 1e3) & (x_real < N[0] * dx[0] * 1e3 + pad_x * dx[0] * 1e3) &
        (y_real >= pad_y * dx[1] * 1e3) & (y_real < N[1] * dx[1] * 1e3 + pad_y * dx[1] * 1e3) &
        (z_real >= pad_z * dx[2] * 1e3) & (z_real < N[2] * dx[2] * 1e3 + pad_z * dx[2] * 1e3)
    )
    x -= pad_x; y -= pad_y; z -= pad_z
    x, y, z = x[valid_indices], y[valid_indices], z[valid_indices]

    # Create receiver mask
    receivers_mask = jnp.full(N, False)
    receivers_mask = receivers_mask.at[x, y, z].set(True)
    receiver_positions = jnp.argwhere(receivers_mask)

    # Instantiate sensors
    sensors = Sensors(positions=tuple(receiver_positions.T.tolist()))
    sensors_all = Sensors(positions=tuple(jnp.argwhere(jnp.ones(N)).T.tolist()))

    return sensors, sensors_all, receivers_mask


def create_sources_receivers(domain, time_axis, freq_Hz=0.25e6, inside: bool = False, n_sources: int = 400, n_sensors: int = 400, pad: int = 30):
    sources, source_mask = create_sources(domain, time_axis, freq_Hz, inside, n_sources, pad)
    sensors, sensors_all, receivers_mask = create_receivers(domain, time_axis, freq_Hz, n_sensors, pad)
    return sources, sensors, sensors_all, source_mask, receivers_mask

def plot_medium(medium, source_mask, sources, time_axis, receivers_mask):
    N = medium.domain.N
    # Plot the speed of sound map
    plt.figure(figsize=(10, 8))
    plt.imshow(medium.sound_speed.on_grid[N[0]//2, :, :,0].T, cmap='viridis')
    plt.colorbar(label='Speed of Sound (m/s)')
    plt.title('Speed of Sound Distribution')
    plt.xlabel('y (grid points)')
    plt.ylabel('z (grid points)')
    plt.show()

    # Plot the source locations
    plt.figure(figsize=(10, 8))
    plt.imshow(jnp.max(source_mask[0, :, :, :], axis=0).T, cmap='binary', label='Sources')
    plt.title('Source Locations')
    plt.xlabel('y (grid points)')
    plt.ylabel('z (grid points)')
    plt.colorbar(label='Source Present')
    plt.show()

    # Plot the receivers locations
    plt.figure(figsize=(10, 8))
    plt.imshow(receivers_mask[N[0]//2, :, :].T, cmap='binary', label='Receivers')
    plt.title('Receivers Locations')
    plt.xlabel('y (grid points)')
    plt.ylabel('z (grid points)')
    plt.colorbar(label='Receivers Present')
    plt.show()

    # Plot the signal used for sources
    signal = sources.signals[0]
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis.to_array() * 1e6, signal)  # Convert time to microseconds
    plt.title('Source Signal')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()



# Define function that converts the waveforms at the receiver sensors, to the final sensor output. In this case, we define the final sensor output to be the arrival time of the waveform for each sensor.

def find_arrival_time(signal2, sources):
  signal = sources.signals[0]
  # Cross-correlate with signal to get arrival times

  correlation = jnp.correlate(signal2, signal, mode='full')[-len(signal):]

  max_idx = jnp.argmax(correlation)
  correlation = correlation / (jnp.max(correlation) + 1e-8)  # Normalize for numerical stability

  # Use softmax-based differentiable argmax
  # Temperature parameter controls sharpness of the softmax
  temperature = 1e-2
  softmax_weights = jax.nn.softmax(correlation / temperature, axis=0)

  # Compute weighted sum of time indices
  time_axis_array = time_axis.to_array()
  arrival_times = jnp.sum(softmax_weights * time_axis_array, axis=0)

  return arrival_times
