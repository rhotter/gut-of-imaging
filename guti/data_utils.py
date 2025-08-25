import numpy as np
import os
from dataclasses import asdict
from typing import Optional, Tuple, Dict
from numpy.typing import NDArray
from guti.parameters import Parameters

# Get the absolute path to the results directory at the root level
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
VARIANTS_DIR = os.path.join(RESULTS_DIR, "variants")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VARIANTS_DIR, exist_ok=True)



def save_svd(
    s: NDArray,
    modality_name: str,
    params: Optional[Parameters] = None,
    subdir: Optional[str] = None,
) -> None:
    """
    Save the singular value spectrum and optional parameters to a file.

    Parameters
    ----------
    s : ndarray
        Singular values from SVD
    modality_name : str
        Name of modality, e.g. 'fnirs_cw' or 'eeg'
    params : Parameters, optional
        Parameters object with the following structure:
        Parameters(
            num_sensors: int,
            grid_resolution: float,
            num_brain_grid_points: int,
            time_resolution: float,
            comment: str,
            noise_full_brain: float
        )
    subdir : str, optional
        Subdirectory within variants/[modality_name]/ to organize parameter sweeps.

    """
    if subdir is None:
        # Save as default configuration in main results directory
        filepath = os.path.join(RESULTS_DIR, f"{modality_name}_svd_spectrum.npz")
        if params is None:
            np.savez(filepath, singular_values=s)
        else:
            structured_params = asdict(params)
            np.savez(filepath, singular_values=s, parameters=structured_params)  # type: ignore
        print(f"Saved default SVD spectrum to {filepath}")
    else:
        # Save as variant in variants directory with hash
        if params is None:
            # If no params provided but default=False, create empty params for hashing
            params = Parameters()
        params_hash = params.get_hash()

        # Determine the target directory: variants/[modality_name]/[subdir]/
        target_dir = os.path.join(VARIANTS_DIR, modality_name, subdir)
        os.makedirs(target_dir, exist_ok=True)

        filepath = os.path.join(target_dir, f"{params_hash}.npz")
        structured_params = asdict(params)
        np.savez(filepath, singular_values=s, parameters=structured_params)  # type: ignore
        print(f"Saved variant SVD spectrum to {filepath}")
        print(f"Parameter hash: {params_hash}")
        print(f"Parameters: {structured_params}")


def load_svd(modality_name: str) -> Tuple[NDArray, Optional[Parameters]]:
    """
    Load the singular value spectrum from a file.

    Parameters
    ----------
    modality_name : str
        Name of modality, e.g. 'fnirs_cw' or 'eeg'

    Returns
    -------
    tuple
        (singular_values, parameters) where parameters is a Parameters object
        or None if no parameters were saved
    """
    filepath = os.path.join(RESULTS_DIR, f"{modality_name}_svd_spectrum.npz")
    data = np.load(filepath, allow_pickle=True)
    try:
        params_dict = data["parameters"].item()  # Use .item() to get the dictionary
        return data["singular_values"], Parameters.from_dict(params_dict)
    except KeyError:
        return data["singular_values"], None


def load_all_svds() -> Dict[str, Tuple[NDArray, Optional[Parameters]]]:
    """
    Load all SVD spectrums from the results folder.

    Returns
    -------
    dict
        Dictionary mapping modality names to tuples of (singular_values, Parameters)
        where Parameters is a Parameters object or None if no parameters were saved
    """
    results = {}
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith("_svd_spectrum.npz"):
            modality_name = filename.replace("_svd_spectrum.npz", "")
            results[modality_name] = load_svd(modality_name)
    return results


def load_svd_variant(
    modality_name: str, params_hash: str, subdir: Optional[str] = None
) -> Tuple[NDArray, Optional[Parameters]]:
    """
    Load a specific parameter variant of the SVD spectrum.

    Parameters
    ----------
    modality_name : str
        Name of modality, e.g. 'fnirs_cw' or 'eeg'
    params_hash : str
        Hash of the parameter configuration
    subdir : str, optional
        Subdirectory within variants/[modality_name]/ where the file is located

    Returns
    -------
    tuple
        (singular_values, parameters) where parameters is a Parameters object
    """
    if subdir is not None:
        filepath = os.path.join(
            VARIANTS_DIR, modality_name, subdir, f"{params_hash}.npz"
        )
    else:
        filepath = os.path.join(VARIANTS_DIR, modality_name, f"{params_hash}.npz")
    data = np.load(filepath, allow_pickle=True)
    params_dict = data["parameters"].item()
    return data["singular_values"], Parameters.from_dict(params_dict)


def list_svd_variants(
    modality_name: str, subdir: Optional[str] = None
) -> Dict[str, Parameters]:
    """
    List all available parameter variants for a modality.

    Parameters
    ----------
    modality_name : str
        Name of modality, e.g. 'fnirs_cw' or 'eeg'
    subdir : str, optional
        Subdirectory within variants/[modality_name]/ to search in

    Returns
    -------
    dict
        Dictionary mapping parameter hashes to Parameters objects
    """
    variants = {}

    # Determine the directory to search in
    if subdir is not None:
        search_dir = os.path.join(VARIANTS_DIR, modality_name, subdir)
        if not os.path.exists(search_dir):
            return variants
    else:
        search_dir = os.path.join(VARIANTS_DIR, modality_name)

    if not os.path.exists(search_dir):
        return variants

    for filename in os.listdir(search_dir):
        if filename.endswith(".npz"):
            # Extract hash from filename (hash is the filename without .npz)
            hash_part = filename[:-4]  # Remove .npz suffix
            if len(hash_part) == 8:  # Our hashes are 8 characters
                try:
                    s, params = load_svd_variant(modality_name, hash_part, subdir)
                    if params is not None:
                        variants[hash_part] = dict(s=s, params=params)
                except FileNotFoundError:
                    continue

    return variants
