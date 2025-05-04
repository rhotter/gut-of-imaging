import numpy as np
from dataclasses import dataclass
import numpy as np

"""
This class defines the source and detector geometry.
"""


@dataclass
class SensorGeometry:
    """
    This class defines the source and detector geometry.
    """

    src_pos: np.ndarray
    det_pos: np.ndarray
    src_dirs: np.ndarray

    def __post_init__(self):
        self.nsrc = self.src_pos.shape[0]
        self.ndet = self.det_pos.shape[0]
