#%%
%load_ext autoreload
%autoreload 2

# %%
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

from guti.modalities.us.utils import create_medium, create_sources, create_receivers, plot_medium, find_arrival_time
import scipy.sparse


#NOTE: There's a bug in jwave, where the gradients are not computed correctly when using FiniteDifferences. Therefore, we use the FourierSeries class instead.

"""Check if JAX is using CUDA."""
platforms = jax.devices()
is_cuda = any('cuda' in str(device).lower() for device in platforms)
print(f"JAX is using CUDA: {is_cuda}")

domain, medium_original, time_axis, brain_mask, skull_mask, scalp_mask = create_medium()

sources, source_mask = create_sources(domain, time_axis, freq_Hz=0.1666e6, inside=True)
sensors, sensors_all, receivers_mask = create_receivers(domain, time_axis, freq_Hz=0.1666e6)


# settings = TimeWavePropagationSettings(checkpoint=True)

# Compile and create the solver functions

plot_medium(medium_original, source_mask, sources, time_axis, receivers_mask)

pml_size = medium_original.pml_size

#%%
# @jit
# def solver_all(medium, sources, sensors):
#   return simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors)

# @jit
# def solver_receiver(medium, sources, sensors):
#   return simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors)

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

source_mask = source_mask[0,:,:,:]
speed = medium_original.sound_speed.on_grid[...,0]

# Print the number of contrast source points
num_contrast_points = jnp.sum(source_mask)
print(f"Number of contrast source points: {num_contrast_points}")


def receiver_output_for_sensors(sensors_):
    def receiver_output(speed_contrast_sources):
        speed_of_sound = speed.at[source_mask].set(speed_contrast_sources)
        pressure = output_field(speed_of_sound, sources, sensors_)


        pressure_downsampled = pressure[::50,:,0].flatten()
        
        # Combine metrics into single differentiable output
        # return jnp.concatenate([peak_freq_amp.ravel(), arrival_times.ravel()])
        return pressure_downsampled
    return receiver_output

speed_contrast_sources = speed[source_mask]

#%%

# print("Computing Jacobian")

# from scipy.sparse.linalg import LinearOperator, svds

# # Forward  product  J @ v
# def J_mv(v):
#     return jax.jvp(receiver_output,
#                    (speed_contrast_sources,),
#                    (v,))[1]

# # Adjoint product Jᵀ @ u
# def JT_mv(u):
#     return jax.vjp(receiver_output,
#                    speed_contrast_sources)[1](u)[0]

# out_size = receiver_output(speed_contrast_sources).size
# in_size  = speed_contrast_sources.size

# # Compute Jacobian directly with JAX for GPU
# jacobian_fn = jax.jacrev(receiver_output)
# jacobian_matrix = jacobian_fn(speed_contrast_sources)

# # Compute SVD directly on GPU using JAX
# u, s, vh = jax.numpy.linalg.svd(jacobian_matrix, full_matrices=False)
# s = s[:1000]  # Take top 1000 singular values

n_sensors_per_batch = 200
n_batches = 10
results = []
for i in range(n_batches):
  
    sensors, sensors_all, receivers_mask = create_receivers(domain, time_axis, freq_Hz=0.1666e6, num_receivers=n_sensors_per_batch * n_batches, start_n=i*n_sensors_per_batch, end_n=(i+1)*n_sensors_per_batch)
    print(f"{len(sensors.positions)} sensors in batch {i}")
    jacobian = jax.jacrev(receiver_output_for_sensors(sensors))(speed_contrast_sources)
    results.append(jacobian)
    print(f"Computed Jacobian!!! shape: {jacobian.shape}")

jacobian = jnp.concatenate(results, axis=0)
print(f"Total Jacobian shape: {jacobian.shape}")

# %%
i=40
n_sensors_per_batch = 200
n_batches = 10
sensors, sensors_all, receivers_mask = create_receivers(domain, time_axis, freq_Hz=0.1666e6, num_receivers=400, start_n=0, end_n=200)
print(sensors.positions)

# %%

# # Compute the singular value spectrum.


# # Save the Jacobian matrix to a file
# np.save('jacobian_amplitude_arrival.npy', np.array(jacobian))
u, s, vh = np.linalg.svd(np.array(jacobian))

# u, s, vh = jax.numpy.linalg.svd(jacobian, full_matrices=False)

# u, s, vh = scipy.sparse.linalg.svds(jacobian, k=1000)


# svd
# Use numpy's SVD instead of JAX's to avoid GPU solver errors
# This is more stable for large matrices but will run on CPU
# Convert back to JAX arrays if needed for downstream operations
# u, s, vh = jnp.array(u), jnp.array(s), jnp.array(vh)

# Plot singular value spectrum
plt.figure(figsize=(10, 6))
plt.semilogy(s)
plt.grid(True)
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum for amplitude+arrival time based US imaging')
plt.show()

#%%

from guti.data_utils import save_svd

save_svd(s, 'pat')

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

