import numpy as np
import os
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional, Union, Tuple, Dict
from numpy.typing import NDArray

# Get the absolute path to the results directory at the root level
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
VARIANTS_DIR = os.path.join(RESULTS_DIR, "variants")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VARIANTS_DIR, exist_ok=True)


@dataclass
class Parameters:
    """
    Parameters structure for SVD analysis.
    
    Attributes
    ----------
    num_sensors : int, optional
        Number of sensors used in the measurement
    grid_resolution : float, optional
        Resolution of the computational grid
    num_brain_grid_points : int, optional
        Number of grid points in the brain model
    time_resolution : float, optional
        Temporal resolution of the measurement
    comment : str, optional
        Additional comment or description
    """
    num_sensors: Optional[int] = None
    grid_resolution: Optional[float] = None
    num_brain_grid_points: Optional[int] = None
    time_resolution: Optional[float] = None
    comment: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Parameters':
        """Create Parameters object from dictionary."""
        return cls(
            num_sensors=data.get('num_sensors'),
            grid_resolution=data.get('grid_resolution'),
            num_brain_grid_points=data.get('num_brain_grid_points'),
            time_resolution=data.get('time_resolution'),
            comment=data.get('comment')
        )



def _generate_params_hash(params: Parameters) -> str:
    """Generate a short hash for parameter configuration."""
    params_dict = asdict(params)
    # Sort keys for consistent hashing
    params_str = json.dumps(params_dict, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]


def save_svd(s: NDArray, modality_name: str, params: Optional[Parameters] = None, default: bool = True) -> None:
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
            comment: str
        )
    default : bool, optional
        If True, save as default configuration in results/.
        If False, save as variant in results/variants/ with parameter hash.
        Default is True.
        
    Notes
    -----
    - When default=True: saved as '{modality_name}_svd_spectrum.npz' in results/
    - When default=False: saved as '{modality_name}_svd_spectrum_{hash}.npz' in results/variants/
    """
    if default:
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
        params_hash = _generate_params_hash(params)
        filepath = os.path.join(VARIANTS_DIR, f"{modality_name}_svd_spectrum_{params_hash}.npz")
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


def load_svd_variant(modality_name: str, params_hash: str) -> Tuple[NDArray, Optional[Parameters]]:
    """
    Load a specific parameter variant of the SVD spectrum.

    Parameters
    ----------
    modality_name : str
        Name of modality, e.g. 'fnirs_cw' or 'eeg'
    params_hash : str
        Hash of the parameter configuration

    Returns
    -------
    tuple
        (singular_values, parameters) where parameters is a Parameters object
    """
    filepath = os.path.join(VARIANTS_DIR, f"{modality_name}_svd_spectrum_{params_hash}.npz")
    data = np.load(filepath, allow_pickle=True)
    params_dict = data["parameters"].item()
    return data["singular_values"], Parameters.from_dict(params_dict)


def list_svd_variants(modality_name: str) -> Dict[str, Parameters]:
    """
    List all available parameter variants for a modality.

    Parameters
    ----------
    modality_name : str
        Name of modality, e.g. 'fnirs_cw' or 'eeg'

    Returns
    -------
    dict
        Dictionary mapping parameter hashes to Parameters objects
    """
    variants = {}
    pattern = f"{modality_name}_svd_spectrum_"
    
    for filename in os.listdir(VARIANTS_DIR):
        if filename.startswith(pattern) and filename.endswith(".npz"):
            # Extract hash from filename
            hash_part = filename[len(pattern):-4]  # Remove prefix and .npz suffix
            if len(hash_part) == 8:  # Our hashes are 8 characters
                try:
                    _, params = load_svd_variant(modality_name, hash_part)
                    if params is not None:
                        variants[hash_part] = params
                except FileNotFoundError:
                    continue
    
    return variants
