#%%

# ---- JAX memory behaviour ---------------------------------------------
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'   # ⬅ no 75 % grab
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'   # optional, 30 %
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # safer allocator
# os.environ['CUDA_VISIBLE_DEVICES'] = ''                 # ← CPU-only fallback
# # ------------------------------------------------------------------------

from jax import jit

from jwave import FourierSeries, FiniteDifferences
from jwave.acoustics.time_varying import simulate_wave_propagation, TimeWavePropagationSettings
from jwave.geometry import Medium
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax import grad, value_and_grad
import jax
from jax import vmap

import jax

from scipy.sparse.linalg import LinearOperator, svds

from guti.modalities.us.utils import create_medium, create_sources_receivers, plot_medium, find_arrival_time
import scipy.sparse


#NOTE: There's a bug in jwave, where the gradients are not computed correctly when using FiniteDifferences. Therefore, we use the FourierSeries class instead.

"""Check if JAX is using CUDA."""
platforms = jax.devices()
is_cuda = any('cuda' in str(device).lower() for device in platforms)
print(f"JAX is using CUDA: {is_cuda}")

print("Creating medium")

domain, medium_original, time_axis, brain_mask, skull_mask, scalp_mask = create_medium()

print("Creating sources and receivers")

sources, sensors, sensors_all, source_mask, receivers_mask = create_sources_receivers(domain, time_axis, freq_Hz=0.1666e6)

find_arrival_time_vectorized = vmap(lambda signal2: find_arrival_time(signal2, sources))

print("Creating solver functions")
# Compile and create the solver functions
@jit
def solver_all(medium, sources):
  return simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors_all)

@jit
def solver_receiver(medium, sources):
  return simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors)

plot_medium(medium_original, source_mask, sources, time_axis, receivers_mask)

pml_size = medium_original.pml_size

#%%

# # Plot of the forward simulation for sanity check
# N = domain.N
# pressure_0 = solver_all(medium_original, sources).reshape(-1, N[0], N[1], N[2], 1)
# pressure_0_numpy = np.array(pressure_0)

# # Plot a slice at a specific time step (e.g., middle of the simulation)
# time_step = pressure_0_numpy.shape[0] - 1  # Last time step
# plt.figure(figsize=(10, 8))
# plt.imshow(pressure_0_numpy[time_step, N[0]//2, pml_size:-pml_size, pml_size:-pml_size, 0].T, cmap='seismic')
# plt.colorbar(label='Pressure')
# plt.title(f'Pressure field at time step {time_step}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

#%%

# Function mapping speed of sound field, sources, and sensors to pressure field

settings = TimeWavePropagationSettings(checkpoint=True)
# Jax function to map a speed of sound map to the corresponding pressure field
def output_field(s, sources, sensors):
    sound_speed2 = FourierSeries(s, domain)
    # Recreate the sound_speed and medium with the new speed
    density_field = jax.lax.stop_gradient(medium_original.density)
    medium = Medium(domain=domain, sound_speed=sound_speed2, density=density_field, pml_size=pml_size)
    # Get pressure field
    pressure = simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors, settings=settings)
    # pressure = simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors)
    return pressure

#%%

# Subsampling the voxels in the speed of sound field to create the contrast sources (the inputs to the imaging forward model)

speed = medium_original.sound_speed.on_grid[...,0]

contrast_sources_mask = jnp.full(speed.shape, False)

# Set the contrast sources mask to be every nth voxel inside the brain mask
n = 10  # Take every 10th voxel
brain_indices = jnp.argwhere(brain_mask)
selected_indices = brain_indices[::n]  # Take every nth index

# Create the contrast sources mask
contrast_sources_mask = contrast_sources_mask.at[tuple(selected_indices.T)].set(True)

# Print the number of contrast source points
num_contrast_points = jnp.sum(contrast_sources_mask)
print(f"Number of contrast source points: {num_contrast_points}")

n_inputs = len(selected_indices)


#%%

print("Computing Jacobian")
nt = time_axis.Nt

n_sensors = sensors.shape[0]

combined_jacobian = np.zeros((n_sensors * nt, n_inputs))

n_outputs_filled = 0

for i in range(20):

  print(f"Computing Jacobian for time shift {i}")

  def receiver_output(speed_contrast_sources):
      speed_of_sound = speed.at[contrast_sources_mask].set(speed_contrast_sources)
      pressure = output_field(speed_of_sound, sources, sensors)

      pressure_downsampled = pressure[i::20,:,0].flatten()
      
      return pressure_downsampled

  speed_contrast_sources = speed[contrast_sources_mask]


  jacobian = jax.jacrev(receiver_output)(speed_contrast_sources)

  combined_jacobian[n_outputs_filled:n_outputs_filled+jacobian.shape[0], :] = jacobian

  n_outputs_filled += jacobian.shape[0]



# %%

# Compute the singular value spectrum.
u, s, vh = np.linalg.svd(np.array(combined_jacobian))

# Plot singular value spectrum
plt.figure(figsize=(10, 6))
plt.semilogy(s)
plt.grid(True)
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum for amplitude+arrival time based US imaging')
plt.show()

#%%

# Save results

from guti.data_utils import save_svd

save_svd(s, 'us')

np.save('combined_jacobian.npy', combined_jacobian)

#%%

# # Example of computing gradients of an objective function with respect to the speed of sound map.
# # We need to perturb the speed of sound map so that the gradients are not zero.


# target = jnp.array(np.array(solver_receiver(medium_original, sources)))

# # Plot the target waveform at a specific point
# plt.figure(figsize=(10, 6))
# plt.plot(target[:,100,0])
# plt.title('Target waveform at point (100,100)')
# plt.xlabel('Time step')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()


# # Define a function that returns a scalar value from the pressure field
# # @jit
# def objective(s):

#     pressure = output_field(s, sources, sensors)

#     diff = (pressure - target)

#     # Compute mean squared error
#     return jnp.sum(diff**2)

# speed2 = jnp.array(np.copy(medium_original.sound_speed.on_grid[...,0].at[brain_mask].set(1650.)))

# # Plot the speed of sound map
# plt.figure(figsize=(10, 8))
# N = domain.N
# plt.imshow(speed2[N[0]//2, :, :], cmap='viridis')
# plt.colorbar(label='Speed of Sound (m/s)')
# plt.title('Speed of Sound Distribution')
# plt.xlabel('x (grid points)')
# plt.ylabel('y (grid points)')
# plt.show()

# # sound_speed2 = FourierSeries(speed2, domain)

# # Test the gradient computation with current speed
# obj_value, gradient = value_and_grad(objective)(speed2)

# print(f"Objective value: {obj_value}")
# print(f"Gradient shape: {gradient.shape}")
# print(f"Gradient max: {gradient.max()}")
# print(f"Gradient min: {gradient.min()}")

# # Compute gradient

# # Plot the gradient
# plt.figure(figsize=(10, 8))
# plt.imshow(gradient[N[0]//2, pml_size:-pml_size, pml_size:-pml_size].T, cmap='seismic')
# plt.colorbar(label='Gradient')
# plt.title('Gradient of objective with respect to sound speed')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# %%

