"""
Compute SVDs of the leadfields of the OpenMEEG sample data.
"""
# %% 
# Import modules
import numpy as np
import h5py
import matplotlib.pyplot as plt

#%%
# Load the data
def load_mat73(file_path):
    data = {}
    with h5py.File(file_path, 'r') as f:
        # Iterate through all variables in the .mat file
        for key in f.keys():
            # Get the variable data
            var = f[key][()]
            # If the variable is a reference to another dataset, follow the reference
            if isinstance(var, h5py.Reference):
                var = f[var][()]
            # Store in dictionary
            data[key] = var
    return data

G_meg = load_mat73('openmeeg_sample_data/leadfields/meg_leadfield.mat')['linop']
G_eeg = load_mat73('openmeeg_sample_data/leadfields/eeg_leadfield.mat')['linop']
G_eit = load_mat73('openmeeg_sample_data/leadfields/eit_leadfield.mat')['linop']
G_ip = load_mat73('openmeeg_sample_data/leadfields/ip_leadfield.mat')['linop']
G_ecog = load_mat73('openmeeg_sample_data/leadfields/ecog_leadfield.mat')['linop']

# # print shape (source-detector pairs, voxels)
# print(G_meg['G'].shape)
# print(G_eeg['G'].shape)
# print(G_eit['G'].shape)
# print(G_ip['G'].shape)
# print(G_ecog['G'].shape)

#%%
# Compute SVDs
s_meg = np.linalg.svdvals(G_meg)
s_eeg = np.linalg.svdvals(G_eeg)
s_eit = np.linalg.svdvals(G_eit)
s_ip = np.linalg.svdvals(G_ip)
s_ecog = np.linalg.svdvals(G_ecog)

# # normalize by the first singular value
# s_meg = s_meg / s_meg[0]
# s_eeg = s_eeg / s_eeg[0]
# s_eit = s_eit / s_eit[0]
# s_ip = s_ip / s_ip[0]
# s_ecog = s_ecog / s_ecog[0]

#%%
# Plot SVDs
plt.semilogy(s_meg, label='MEG')
plt.semilogy(s_eeg, label='EEG')
plt.semilogy(s_eit, label='EIT')
plt.semilogy(s_ip, label='IP')
plt.semilogy(s_ecog, label='ECoG')
plt.xlabel('Singular value index')
plt.ylabel('Singular value')
plt.grid(True)
plt.legend()
plt.show()

#%%
# Save the SVDs
import sys

from guti.data_utils import save_svd

save_svd(s_meg, 'meg_openmeeg')
save_svd(s_eeg, 'eeg_openmeeg')
save_svd(s_eit, 'eit_openmeeg')
save_svd(s_ip, 'ip_openmeeg')
save_svd(s_ecog, 'ecog_openmeeg')









# %%
