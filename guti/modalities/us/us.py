#%%
from jax import jit

from jwave import FourierSeries, FiniteDifferences
from jwave.acoustics.time_varying import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis
from jwave.utils import load_image_to_numpy
from jwave.geometry import Sources, Sensors
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax import grad, value_and_grad
import jax

from guti.core import get_voxel_mask, get_sensor_positions_spiral, get_source_positions

#NOTE: There's a bug in jwave, where the gradients are not computed correctly when using FiniteDifferences. Therefore, we use the FourierSeries class instead.


def create_medium():

  # Simulation parameters
  dx_mm = 0.5
  dx = dx_mm * 1e-3

  tissues_map = get_voxel_mask(dx_mm) #0.5mm resolution
  N = tissues_map.shape
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

  pml_size = 20
  medium = Medium(domain=domain, sound_speed=sound_speed, density=density_field, pml_size=pml_size)
  time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=100e-06)

  brain_mask = tissues_map == 1
  skull_mask = tissues_map == 2
  scalp_mask = tissues_map == 3

  return domain, medium, time_axis, brain_mask, skull_mask, scalp_mask

def create_sources_receivers(domain, time_axis, freq_Hz=0.5e6):
  # Create a 128x128 square of sources centered in x and at y=25
  N = domain.N
  x_start = N[0] // 4  # Center the square horizontally
  x_coords = jnp.arange(x_start, x_start + N[0] // 2)
  x_pos = x_coords
  y_pos = jnp.full(N[0] // 2, N[1] // 4)  # Fixed y position at 25

  signal = jnp.sin(freq_Hz*time_axis.to_array())
  signal = signal.at[400:].set(0)
  signals = jnp.stack([signal] * (N[0] // 2))  # One signal for each source point
  sources = Sources(positions=(x_pos, y_pos), signals=signals, dt=time_axis.dt, domain=domain)

  # Create a mask for the sources and receivers
  source_mask = jnp.full((int(time_axis.Nt),) + N, False)
  source_mask = source_mask.at[:, x_pos, y_pos].set(True)
  receivers_mask = jnp.full(N, False)
  receivers_mask = receivers_mask.at[x_pos, y_pos+1].set(True)
  receiver_positions = jnp.argwhere(receivers_mask)

  sensors = Sensors(positions=tuple((receiver_positions.T).tolist()))
  sensors_all = Sensors(positions=tuple(jnp.argwhere(jnp.ones(N)).T.tolist()))

  return sources, sensors, sensors_all, source_mask, receivers_mask

domain, medium_original, time_axis, brain_mask, skull_mask, scalp_mask = create_medium()

sources, sensors, sensors_all, source_mask, receivers_mask = create_sources_receivers(domain, time_axis)

# Compile and create the solver functions
@jit
def solver_all(medium, sources):
  return simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors_all)

@jit
def solver_receiver(medium, sources):
  return simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors)

# %%

def plot_medium(medium, source_mask, receivers_mask):
  N = medium.domain.N
  # Plot the speed of sound map
  plt.figure(figsize=(10, 8))
  plt.imshow(medium.sound_speed.on_grid[N[0]//2, :, :,0], cmap='viridis')
  plt.colorbar(label='Speed of Sound (m/s)')
  plt.title('Speed of Sound Distribution')
  plt.xlabel('x (grid points)')
  plt.ylabel('y (grid points)')
  plt.show()

  # Plot the source locations
  plt.figure(figsize=(10, 8))
  plt.imshow(source_mask[0, N[0]//2, :, :].T, cmap='binary', label='Sources')
  plt.title('Source Locations')
  plt.xlabel('x (grid points)')
  plt.ylabel('y (grid points)')
  plt.colorbar(label='Source Present')
  plt.show()

  # Plot the receivers locations
  plt.figure(figsize=(10, 8))
  plt.imshow(receivers_mask[N[0]//2, :, :].T, cmap='binary', label='Receivers')
  plt.title('Receivers Locations')
  plt.xlabel('x (grid points)')
  plt.ylabel('y (grid points)')
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


plot_medium(medium_original, source_mask, receivers_mask)

#%%

# Plot of the forward simulation for sanity check
N = domain.N
pressure_0 = solver_all(medium_original, sources).reshape(-1, N[0], N[1], N[2], 1)
pressure_0_numpy = np.array(pressure_0)
pml_size = medium_original.pml_size

# Plot a slice at a specific time step (e.g., middle of the simulation)
time_step = pressure_0_numpy.shape[0] - 1  # Last time step
plt.figure(figsize=(10, 8))
plt.imshow(pressure_0_numpy[time_step, N[0]//2, pml_size:-pml_size, pml_size:-pml_size, 0].T, cmap='seismic')
plt.colorbar(label='Pressure')
plt.title(f'Pressure field at time step {time_step}')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#%%

# Jax function to map a speed of sound map to the corresponding pressure field
def output_field(s, sources, sensors):
    sound_speed2 = FourierSeries(s, domain)
    # Recreate the sound_speed and medium with the new speed
    density_field = medium_original.density
    medium = Medium(domain=domain, sound_speed=sound_speed2, density=density_field, pml_size=pml_size)
    # Get pressure field
    pressure = simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors)
    return pressure

#%%

# Define function that converts the waveforms at the receiver sensors, to the final sensor output. In this case, we define the final sensor output to be the arrival time of the waveform for each sensor.

def find_arrival_time(signal2, sources):

  signal = sources.signals[0]
  # Cross-correlate with signal to get arrival times
  # Compute cross-correlation in time domain
  correlation = jnp.correlate(signal2, signal, mode='full')[-len(signal):]

  # Use softmax-based differentiable argmax
  # Temperature parameter controls sharpness of the softmax
  temperature = 1e-2
  softmax_weights = jax.nn.softmax(correlation / temperature, axis=0)

  # Compute weighted sum of time indices
  time_axis_array = time_axis.to_array()
  arrival_times = jnp.sum(softmax_weights * time_axis_array, axis=0)

  return arrival_times

print(find_arrival_time(jnp.roll(sources.signals[0], 100), sources))

from jax import vmap

find_arrival_time_vectorized = vmap(lambda signal2: find_arrival_time(signal2, sources))

#%%

speed = medium_original.sound_speed.on_grid[...,0]

def receiver_output(speed_of_sound_brain):
    speed_of_sound = speed.at[brain_mask].set(speed_of_sound_brain)
    pressure = output_field(speed_of_sound, sources, sensors)
    # Compute Fourier transform along time dimension (axis 0)
    pressure_fft = jnp.fft.fft(pressure, axis=0)

    arrival_times = find_arrival_time_vectorized(pressure[:,:,0].T)
    # Get peak frequency amplitude
    peak_freq_amp = jnp.max(jnp.abs(pressure_fft), axis=0)
    
    # Combine metrics into single differentiable output
    return jnp.concatenate([peak_freq_amp.ravel(), arrival_times.ravel()])

speed_brain = speed[brain_mask]

jacobian = jax.jacobian(receiver_output)(speed_brain)
# %%

# Compute the singular value spectrum.

# Save the Jacobian matrix to a file
np.save('jacobian_amplitude_arrival.npy', np.array(jacobian))

# svd
u, s, vh = jnp.linalg.svd(jacobian)

# Plot singular value spectrum
plt.figure(figsize=(10, 6))
plt.semilogy(s, 'o-')
plt.grid(True)
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum for amplitude+arrival time based US imaging')
plt.show()

#%%


# Example of computing gradients of an objective function with respect to the speed of sound map.
# We need to perturb the speed of sound map so that the gradients are not zero.


target = jnp.array(np.array(solver_receiver(medium_original, sources)))

# Plot the target waveform at a specific point
plt.figure(figsize=(10, 6))
plt.plot(target[:,100,0])
plt.title('Target waveform at point (100,100)')
plt.xlabel('Time step')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


# Define a function that returns a scalar value from the pressure field
# @jit
def objective(s):

    pressure = output_field(s, sources, sensors)

    diff = (pressure - target)

    # Compute mean squared error
    return jnp.sum(diff**2)

speed2 = jnp.array(np.copy(medium_original.sound_speed.on_grid[...,0].at[brain_mask].set(1650.)))

# Plot the speed of sound map
plt.figure(figsize=(10, 8))
plt.imshow(speed2[N[0]//2, :, :], cmap='viridis')
plt.colorbar(label='Speed of Sound (m/s)')
plt.title('Speed of Sound Distribution')
plt.xlabel('x (grid points)')
plt.ylabel('y (grid points)')
plt.show()

# sound_speed2 = FourierSeries(speed2, domain)

# Test the gradient computation with current speed
obj_value, gradient = value_and_grad(objective)(speed2)

print(f"Objective value: {obj_value}")
print(f"Gradient shape: {gradient.shape}")
print(f"Gradient max: {gradient.max()}")
print(f"Gradient min: {gradient.min()}")

# Compute gradient

# Plot the gradient
plt.figure(figsize=(10, 8))
plt.imshow(gradient[N[0]//2, pml_size:-pml_size, pml_size:-pml_size].T, cmap='seismic')
plt.colorbar(label='Gradient')
plt.title('Gradient of objective with respect to sound speed')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
