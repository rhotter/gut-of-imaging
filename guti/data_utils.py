import numpy as np
import os

# Get the absolute path to the results directory at the root level
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_svd(s, modality_name, params=None):
    """
    Save the singular value spectrum and optional parameters to a file.

    Parameters
    ----------
    s : ndarray
        Singular values from SVD
    modality_name : str
        Name of modality, e.g. 'fnirs_cw' or 'eeg'
    params : dict, optional
        Dictionary of parameters to save alongside singular values
    """
    filepath = os.path.join(RESULTS_DIR, f"{modality_name}_svd_spectrum.npz")
    if params is None:
        np.savez(filepath, singular_values=s)
    else:
        np.savez(filepath, singular_values=s, parameters=params)
    print(f"Saved singular value spectrum to {filepath}")


def load_svd(modality_name):
    """
    Load the singular value spectrum from a file.

    Parameters
    ----------
    modality_name : str
        Name of modality, e.g. 'fnirs_cw' or 'eeg'

    Returns
    -------
    tuple
        (singular_values, parameters) where parameters is None if not saved
    """
    filepath = os.path.join(RESULTS_DIR, f"{modality_name}_svd_spectrum.npz")
    data = np.load(filepath)
    try:
        return data["singular_values"], data["parameters"]
    except KeyError:
        return data["singular_values"], None


def load_all_svds():
    """
    Load all SVD spectrums from the results folder.

    Returns
    -------
    dict
        Dictionary mapping modality names to tuples of (singular_values, parameters)
    """
    results = {}
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith("_svd_spectrum.npz"):
            modality_name = filename.replace("_svd_spectrum.npz", "")
            results[modality_name] = load_svd(modality_name)
    return results
