import mne
import numpy as np
import matplotlib.pyplot as plt

def create_sensor_info(n_channels=10, sfreq=1000.0):
    """
    Create sensor information with channels arranged in a circle.
    
    Parameters:
        n_channels (int): Number of MEG channels.
        sfreq (float): Sampling frequency.
    
    Returns:
        info (mne.Info): MEG sensor info.
        ch_names (list): List of channel names.
    """
    ch_names = [f'MEG{i:03d}' for i in range(n_channels)]
    ch_types = ['mag'] * n_channels

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Place sensors evenly on a circle (radius = 0.1 m) in the xy-plane.
    angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    sensor_locs = {
        ch: [0.1 * np.cos(ang), 0.1 * np.sin(ang), 0.0]
        for ch, ang in zip(ch_names, angles)
    }
    montage = mne.channels.make_dig_montage(ch_pos=sensor_locs)
    info.set_montage(montage)
    
    return info, ch_names

def create_sphere_model(r0=(0., 0., 0.), head_radius=0.09):
    """
    Create a spherical head model.
    
    Parameters:
        r0 (tuple): Center of the sphere.
        head_radius (float): Radius of the head model.
    
    Returns:
        sphere (mne.models.SphereModel): The spherical head model.
    """
    sphere = mne.make_sphere_model(r0=r0, head_radius=head_radius)
    return sphere

def simulate_dipole(dipole_pos, dipole_moment, times):
    """
    Create a dipole object with a time-varying amplitude.
    
    Parameters:
        dipole_pos (array-like): Dipole location (meters).
        dipole_moment (array-like): Dipole moment (A·m).
        times (numpy.ndarray): Time vector for the simulation.
    
    Returns:
        dip (mne.Dipole): The simulated dipole object.
    """
    # Create a 10 Hz sine wave as the time course
    dipole_ts = np.sin(2 * np.pi * 10 * times)
    
    # The dipole remains at the same location over time
    dipole_positions = np.tile(dipole_pos, (len(times), 1))
    # Compute the amplitude for each time point (outer product: time course * dipole moment)
    dipole_amplitudes = np.outer(dipole_ts, dipole_moment)
    
    # Orientation is taken from the dipole moment (normalized)
    ori = dipole_moment / np.linalg.norm(dipole_moment)
    dipole_oris = np.tile(ori, (len(times), 1))
    
    # Assign a constant goodness-of-fit for simplicity
    gof = np.full(len(times), 100.0)
    
    from mne import Dipole
    dip = Dipole(times=times, pos=dipole_positions, amplitude=dipole_amplitudes,
                 gof=gof, ori=dipole_oris)
    return dip

def compute_forward_fields(dip, sphere, info):
    """
    Compute the forward fields for the dipole using a spherical head model.
    
    Parameters:
        dip (mne.Dipole): The dipole object.
        sphere (mne.models.SphereModel): Spherical head model.
        info (mne.Info): Sensor information.
    
    Returns:
        fwd_fields (mne.EvokedArray): Simulated sensor data.
    """
    # This function uses MNE's built-in forward_dipole function (based on the Sarvas formula)
    fwd_fields = mne.forward.forward_dipole(dip, sphere, info)
    return fwd_fields

def plot_simulated_data(times, data, ch_names):
    """
    Plot simulated MEG sensor data.
    
    Parameters:
        times (numpy.ndarray): Time vector.
        data (numpy.ndarray): Simulated sensor data (channels x time points).
        ch_names (list): List of channel names.
    """
    plt.figure(figsize=(10, 5))
    n_channels = data.shape[0]
    for idx in range(n_channels):
        # Offset each channel for clarity
        plt.plot(times, data[idx] + idx * 1e-11, label=ch_names[idx])
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic Field (T)')
    plt.title('Simulated MEG Data from a Dipole (Spherical Head Model)')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()

def main():
    # Simulation parameters
    sfreq = 1000.0  # Sampling frequency in Hz
    duration = 0.5  # seconds
    times = np.linspace(0, duration, int(sfreq * duration))
    
    # Create sensor info and get channel names
    info, ch_names = create_sensor_info(n_channels=10, sfreq=sfreq)
    
    # Create the spherical head model
    sphere = create_sphere_model(r0=(0., 0., 0.), head_radius=0.09)
    
    # Define dipole parameters
    dipole_pos = np.array([0.02, 0.0, 0.03])       # Dipole location in meters
    dipole_moment = np.array([1e-8, 0.0, 0.0])       # Dipole moment (A·m)
    
    # Simulate the dipole
    dip = simulate_dipole(dipole_pos, dipole_moment, times)
    
    # Compute the forward solution (simulate sensor data)
    fwd_fields = compute_forward_fields(dip, sphere, info)
    
    print("Simulated sensor data shape:", fwd_fields.data.shape)
    
    # Visualize the simulated MEG data
    plot_simulated_data(times, fwd_fields.data, ch_names)

if __name__ == '__main__':
    main()
