from medium import Medium
from sensor_geometry import SensorGeometry
import pmcx
import numpy as np


class Solver:
    def __init__(
        self, medium: Medium, sensors: SensorGeometry, tstart=0, tend=1e-8, tstep=5e-10
    ):
        self.medium = medium
        self.sensors = sensors
        self.tstart = tstart
        self.tend = tend
        self.tstep = tstep
        self.nt = int((tend - tstart) / tstep)

    def forward(self, src_idx, nphoton=1e7, random_seed=1):
        """
        Implements the forward monte carlo solver.
        """
        config = {
            "seed": random_seed,
            "nphoton": nphoton,
            "vol": self.medium.volume,
            "tstart": self.tstart,
            "tend": self.tend,
            "tstep": self.tstep,
            "srcpos": self.sensors.src_pos[src_idx],
            "srcdir": self.sensors.src_dirs[src_idx],
            "prop": self.medium.optical_properties,
            "detpos": self.sensors.det_pos,
            "replaydet": -1,
            "issavedet": 1,
            "issrcfrom0": 1,
            "issaveseed": 1,
            # 'unitinmm': 1.8,
            "maxdetphoton": nphoton,
        }

        result = pmcx.mcxlab(config)

        return result, config

    def get_td_data(self, res: dict, optical_properties: np.ndarray = None):
        """
        Get time domain data from pmcx.mcxlab() output.

        Parameters
        ----------
        res : dict
            output of pmcx.run()
        optical_properties : np.ndarray, optional
            The optical properties of each medium. If None, use the optical properties
            from the medium object.

        Returns
        -------
        data : (ntimebins, ndetectors) np.ndarray
        """

        if optical_properties is None:
            optical_properties = self.medium.optical_properties

        detp = res["detp"]
        weights = detweight(detp, optical_properties)
        tof = dettime(detp, optical_properties, self.medium.grid_resolution_mm)

        data = np.zeros((self.nt, self.sensors.ndet))

        for i in range(self.sensors.ndet):
            hist, bin_edges = np.histogram(
                tof[detp["detid"] == i + 1],
                bins=self.nt,
                weights=weights[detp["detid"] == i + 1],
                range=(self.tstart, self.tend),
            )
            t = bin_edges[:-1]
            data[:, i] = hist

        return data


def jacobian(forward_result, cfg):
    # one must define cfg['seed'] using the returned seeds
    cfg["seed"] = forward_result["seeds"]

    # one must define cfg['detphotons'] using the returned detp data
    cfg["detphotons"] = forward_result["detp"]["data"]

    # tell mcx to output absorption (μ_a) Jacobian
    cfg["outputtype"] = "jacobian"

    result = pmcx.mcxlab(cfg)

    J = result["flux"]  # Jacobian of shape (nz, ny, nx, nt, ndetectors)

    # Flip sign of jacobian (since dphi = -J @ dmua)
    J = -J
    return J


def dettime(detp, prop, unitinmm=1):
    """
    Recalculate the detected photon time using partial path data and
    optical properties (for perturbation Monte Carlo or detector readings).

    Parameters:
    detp (dict): The second output from mcxlab. detp must be a dictionary.
    prop (list): Optical property list, as defined in the cfg.prop field of mcxlab's input.
    unitinmm (float): Voxel edge-length in mm. If ignored, assume to be 1 (mm).

    Returns:
    dett (numpy.ndarray): Recalculated detected photon time based on the partial path data and optical property table.
    """

    R_C0 = 3.335640951981520e-12  # inverse of light speed in vacuum

    # Check the number of media
    medianum = len(prop)

    dett = np.zeros(detp["ppath"].shape[0])
    for i in range(medianum - 1):
        refractive_index = prop[i + 1][3]  # refractive index
        dett += refractive_index * detp["ppath"][:, i] * R_C0 * unitinmm
    return dett


def detweight(detp, prop):
    """
    Calculates the detector weights for each measurement.

    Parameters
    ----------
    detp : numpy.ndarray
        The detector path information from pmcx.run().
    prop : numpy.ndarray
        The optical properties of each medium.

    Returns
    -------
    numpy.ndarray
        The detector weights for each measurement.
    """
    # For each measurement, we multiply the path length in each medium by the absorption coefficient, and exponentiate
    # to get the intensity.

    pathlengths = detp["ppath"]
    absorption_coefficients = np.array(prop)[1:, 0]  # exclude background

    weights = np.exp(
        -pathlengths @ absorption_coefficients
    )  # * cfg['unitinmm']) # (nmeas,)

    return weights
