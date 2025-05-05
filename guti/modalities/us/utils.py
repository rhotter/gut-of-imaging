from guti.core import get_voxel_mask, get_sensor_positions_spiral, get_source_positions
from jwave import FourierSeries, FiniteDifferences
from jwave.geometry import Domain, Medium, TimeAxis
from jwave.geometry import Sources, Sensors
import jax.numpy as jnp
import matplotlib.pyplot as plt


def create_medium():
    # Simulation parameters
    # dx_mm = 0.5
    dx_mm = 2.0
    # dx_mm = 3
    dx = (dx_mm * 1e-3, dx_mm * 1e-3, dx_mm * 1e-3)

    tissues_map = get_voxel_mask(dx_mm) #0.5mm resolution
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
    time_axis = TimeAxis.from_medium(medium, cfl=0.15, t_end=100e-06)
    # time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=5e-06)

    brain_mask = tissues_map == 1
    skull_mask = tissues_map == 2
    scalp_mask = tissues_map == 3

    return domain, medium, time_axis, brain_mask, skull_mask, scalp_mask

def create_sources_receivers(domain, time_axis, freq_Hz=0.25e6):
    # Create a 128x128 square of sources centered in x and at y=25
    N = domain.N
    dx = domain.dx
    sensor_positions = get_sensor_positions_spiral()
    sensor_positions_voxels = jnp.floor(sensor_positions / (jnp.array(dx)*1e3)).astype(jnp.int32)
    x,y,z = sensor_positions_voxels[:, 0], sensor_positions_voxels[:, 1], sensor_positions_voxels[:, 2]
    print(f"Sensor positions: {sensor_positions}")

    N_sources = sensor_positions.shape[0]


    signal = jnp.sin(2*jnp.pi*freq_Hz*time_axis.to_array())
    T = 1/freq_Hz
    max_signal_index = int(T/time_axis.dt)
    print(f"Max signal index: {max_signal_index}")
    signal = signal.at[max_signal_index*2:].set(0)
    signals = jnp.stack([signal] * N_sources)  # One signal for each source point
    sources = Sources(positions=(x, y, z), signals=signals, dt=time_axis.dt, domain=domain)

    # Create a mask for the sources and receivers
    # source_mask = jnp.full((int(time_axis.Nt),) + N, False)
    source_mask = jnp.full((1,) + N, False)
    source_mask = source_mask.at[:, x, y, z].set(True)
    receivers_mask = jnp.full(N, False)



    # Create a mask for the receivers
    receivers_mask = receivers_mask.at[x, y, z].set(True)
    receiver_positions = jnp.argwhere(receivers_mask)

    sensors = Sensors(positions=tuple((receiver_positions.T).tolist()))
    sensors_all = Sensors(positions=tuple(jnp.argwhere(jnp.ones(N)).T.tolist()))

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
