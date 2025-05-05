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

    def forward(self, src_idx, nphoton=1e7, random_seed=1, voxel_size_mm=1.0):
        """
        Implements the forward monte carlo solver.
        """
        # need to append the radii column to the det_pos
        det_pos = np.hstack((self.sensors.det_pos, 5 * self.sensors.det_radii))

        volume = self.medium.volume

        config = {
            "seed": random_seed,
            "nphoton": nphoton,
            "vol": volume,
            "tstart": self.tstart,
            "tend": self.tend,
            "tstep": self.tstep,
            "srcpos": self.sensors.src_pos[src_idx],
            "srcdir": self.sensors.src_dirs[src_idx],
            "prop": self.medium.optical_properties,
            "detpos": det_pos,
            "replaydet": -1,
            "issavedet": 1,
            "issrcfrom0": 1,
            "issaveseed": 1,
            "unitinmm": voxel_size_mm,
            "maxdetphoton": nphoton,
        }

        result = pmcx.mcxlab(config)

        return result, config


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
