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

from scipy.sparse.linalg import LinearOperator, svds

from guti.modalities.us.utils import create_medium, create_sources_receivers, plot_medium


#NOTE: There's a bug in jwave, where the gradients are not computed correctly when using FiniteDifferences. Therefore, we use the FourierSeries class instead.

"""Check if JAX is using CUDA."""
platforms = jax.devices()
is_cuda = any('cuda' in str(device).lower() for device in platforms)
print(f"JAX is using CUDA: {is_cuda}")

domain, medium_original, time_axis, brain_mask, skull_mask, scalp_mask = create_medium()

sources, sensors, sensors_all, source_mask, receivers_mask = create_sources_receivers(domain, time_axis, freq_Hz=0.15e6)

# settings = TimeWavePropagationSettings(checkpoint=True)

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

# settings = TimeWavePropagationSettings(checkpoint=True)

# Jax function to map a speed of sound map to the corresponding pressure field
def output_field(s, sources, sensors):
    sound_speed2 = FourierSeries(s, domain)
    # Recreate the sound_speed and medium with the new speed
    density_field = jax.lax.stop_gradient(medium_original.density)
    medium = Medium(domain=domain, sound_speed=sound_speed2, density=density_field, pml_size=pml_size)
    # Get pressure field
    # pressure = simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors, settings=settings)
    pressure = simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors)
    return pressure

#%%

# Define function that converts the waveforms at the receiver sensors, to the final sensor output. In this case, we define the final sensor output to be the arrival time of the waveform for each sensor.

def find_arrival_time(signal2, sources):
  signal = sources.signals[0]
  # Cross-correlate with signal to get arrival times
  # Compute cross-correlation in time domain
  # Run correlation on CPU to avoid GPU segfault
  # Compute cross-correlation directly on GPU
  # Avoid using jnp.correlate which can cause segfaults
  # Implement correlation manually using FFT for better stability
  # Use time-domain approach to avoid segfaults
  # correlation = jnp.zeros(len(signal))
  # for i in range(len(signal)):
  #   shift = jnp.roll(signal, i)
  #   correlation = correlation.at[i].set(jnp.sum(signal2 * shift))

  correlation = jnp.correlate(signal2, signal, mode='full')[-len(signal):]
  
  # Alternative: use numpy for correlation which is more stable
  # signal2_np = np.array(signal2)
  # signal_np = np.array(signal)
  # correlation = jnp.array(np.correlate(signal2_np, signal_np, mode='full'))
  # correlation = correlation[len(signal)-1:2*len(signal)-1]  # Extract the relevant part
  # Apply a peak detection to find the maximum correlation
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

print(find_arrival_time(jnp.roll(sources.signals[0], 100), sources))


find_arrival_time_vectorized = vmap(lambda signal2: find_arrival_time(signal2, sources))

#%%

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


def receiver_output(speed_contrast_sources):
    speed_of_sound = speed.at[contrast_sources_mask].set(speed_contrast_sources)
    pressure = output_field(speed_of_sound, sources, sensors)

    # arrival_times = find_arrival_time_vectorized(pressure[:,:,0].T)
    # peak_freq_amp = jnp.max(jnp.abs(pressure), axis=0)

    pressure_downsampled = pressure[::10,:,0].flatten()
    
    # Combine metrics into single differentiable output
    # return jnp.concatenate([peak_freq_amp.ravel(), arrival_times.ravel()])
    return pressure_downsampled

speed_contrast_sources = speed[contrast_sources_mask]

#%%

print("Computing Jacobian")


# jacobian = jax.jacobian(receiver_output)(speed_contrast_sources)
# Reduce memory usage by using block-wise Jacobian computation
# n = speed_contrast_sources.size
# block_size = 1  # tune empirically based on available memory

# Compute a single row of the Jacobian to avoid memory issues
# Using a unit vector for the first parameter only
# unit_vector = jnp.zeros_like(speed_contrast_sources)
# unit_vector = unit_vector.at[0].set(1.0)
# y, partial_jacobian = jax.jvp(receiver_output, (speed_contrast_sources,), (unit_vector,))
# jax.vjp(receiver_output, speed_contrast_sources, jnp.ones_like(receiver_output(speed_contrast_sources)))

#%%

# Create a function that computes JVP for a batch of vectors
# jac_block = jax.vmap(
#     lambda v: jax.jvp(receiver_output, (speed_contrast_sources,), (v,))[1]
# )

# # Create basis vectors in blocks to reduce memory usage
# basis = jnp.eye(n)
# # Reshape basis into blocks and compute Jacobian block by block
# jacobian = jnp.concatenate([
#     jac_block(basis[:, i:min(i+block_size, n)].T) 
#     for i in range(0, n, block_size)
# ], axis=1)
# Use jacrev instead of jacfwd to save memory
# jacrev computes the Jacobian row-by-row which is more memory efficient
# jacobian = jax.jacrev(receiver_output)(speed_contrast_sources)
# jacobian = jax.linearize(receiver_output, speed_contrast_sources)

# y, pullback = jax.linearize(receiver_output, speed_contrast_sources)

# from scipy.sparse.linalg import LinearOperator, svds

# Forward  product  J @ v
def J_mv(v):
    return jax.jvp(receiver_output,
                   (speed_contrast_sources,),
                   (v,))[1]

# Adjoint product Jᵀ @ u
def JT_mv(u):
    return jax.vjp(receiver_output,
                   speed_contrast_sources)[1](u)[0]

out_size = receiver_output(speed_contrast_sources).size
in_size  = speed_contrast_sources.size

J_linop = LinearOperator((out_size, in_size),
                         matvec=J_mv,
                         rmatvec=JT_mv,
                         dtype=np.float32)

_, s, _ = svds(J_linop, k=1000)      # top-20 singular values only


# %%

# # Compute the singular value spectrum.

# import scipy.sparse

# # Save the Jacobian matrix to a file
# np.save('jacobian_amplitude_arrival.npy', np.array(jacobian))
# # u, s, vh = np.linalg.svd(np.array(jacobian), full_matrices=False)

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

