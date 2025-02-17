from medium import Medium
from sensor_geometry import SensorGeometry
import pmcx
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, device_put
import optax
from jax.numpy.linalg import eigvalsh

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


def compute_dphi(data_bg, data_true):
    """
    Compute the normalized difference in time domain data between the background and true data.

    Parameters
    ----------
    data_bg : numpy.ndarray
        The time domain data for the background medium.
    data_true : numpy.ndarray
        The time domain data for the true medium.

    Returns
    -------
    numpy.ndarray
        The difference in time domain data between the background and true data.
    """
    cw_data_bg = np.sum(data_bg, axis=0)[np.newaxis, :]
    dphi = (data_true - data_bg) / cw_data_bg
    return dphi


def ridge_regression(A, y, alpha_frac=1e-4):
    # Transfer data to GPU
    A = device_put(A)
    y = device_put(y)

    # Compute AA^T and its maximum eigenvalue
    ATA = A.T @ A
    eigenvalues = eigvalsh(ATA)
    max_eigenvalue = jnp.max(eigenvalues)

    # Calculate alpha
    alpha = alpha_frac * max_eigenvalue

    # Compute ridge regression coefficients
    I = jnp.eye(A.shape[1])
    beta = jnp.linalg.inv(ATA + alpha * I) @ A.T @ y

    # Compute the error
    error = jnp.sum((A @ beta - y) ** 2)

    return beta, error


@jit
def total_variation(x):
    """Compute the total variation of x."""
    # return jnp.sum(jnp.abs(jnp.diff(x)))
    return (
        jnp.sum(jnp.abs(x[1:, :, :] - x[:-1, :, :]))
        + jnp.sum(jnp.abs(x[:, 1:, :] - x[:, :-1, :]))
        + jnp.sum(jnp.abs(x[:, :, 1:] - x[:, :, :-1]))
    )


@jit
def loss_fn(x, A, b, weights, lambda_tv, lambda_l1, lambda_l2):
    """
    Compute the loss function (least squares + TV regularization).
    """
    Ax = jnp.tensordot(A, x, axes=([0, 1, 2], [0, 1, 2]))
    return (
        jnp.sum(weights * (Ax - b) ** 2)
        + lambda_tv * total_variation(x)
        + lambda_l2 * jnp.sum(x**2)
        + lambda_l1 * jnp.sum(jnp.abs(x))
    )


def error_fn(x, A, b):
    """
    Compute the least squares error.
    """
    Ax = jnp.tensordot(A, x, axes=([0, 1, 2], [0, 1, 2]))
    return jnp.sum((Ax - b) ** 2)


# @jit


def reshape_and_sum(A, grouping_size):
    """
    Reshape the array A and sum over the pixels in the group.

    Parameters:
    A (numpy.ndarray): The input array with shape (nz, ny, nx, nt, ndetectors, nsources).
    grouping_size (int): The size of the grouping for pixels.

    Returns:
    numpy.ndarray: The reshaped and summed array.
    """

    nz, ny, nx, nt, ndetectors, nsources = A.shape

    # Ensure that the dimensions are divisible by grouping_size
    if nz % grouping_size != 0 or ny % grouping_size != 0 or nx % grouping_size != 0:
        raise ValueError(
            "The dimensions of the array are not divisible by the grouping size."
        )

    # Reshape the array to group the pixels
    new_shape = (
        nz // grouping_size,
        grouping_size,
        ny // grouping_size,
        grouping_size,
        nx // grouping_size,
        grouping_size,
        nt,
        ndetectors,
        nsources,
    )

    A_grouped = A.reshape(new_shape)

    # Sum over the grouped pixels (2nd, 4th, and 6th dimensions)
    A_summed = A_grouped.sum(axis=(1, 3, 5))

    return A_summed


def get_weights(data_bg):
    """
    Get the weights for the least squares problem. We trust the data with more photons more.

    We assume poisson noise, so the weights (1/sigma^2) are the background data
    """
    return data_bg


def invert(
    J,
    dphi,
    data_bg,
    lambda_tv=0.1,
    max_iter=1000,
    learning_rate=0.01,
    lambda_l2=1,
    lambda_l1=1,
    voxel_grouping_size=1,
):
    """
    Solves the least squares problem with TV regularization using gradient descent.

    Parameters:
    ----------
    J : array
        Jacobian matrix. Shape: (nz, ny, nx, nt, ndetectors, nsources)
    dphi : array
        Normalized difference between starting and observed data. Shape: (nt, ndetectors)
    data_bg : array
        Background data. Shape: (nt, ndetectors, nsources)

    Optional Parameters:
    -------------------
    lambda_tv : float
        Regularization parameter for the TV term.
    max_iter : int
        Maximum number of iterations for the optimization.
    learning_rate : float
        Learning rate for the optimizer.
    lambda_l2 : float
        Regularization parameter for the L2 term.
    lambda_l1 : float
        Regularization parameter for the L1 term.

    Returns:
    x : array
        Solution to the regularized least squares problem.
    """
    # sum the data over voxel_grouping_size voxels
    # J = reshape_and_sum(J, voxel_grouping_size)

    nz, ny, nx, nt, ndetectors, nsources = J.shape

    J = device_put(J)
    dphi = device_put(dphi)
    weights = get_weights(data_bg)

    # Initial guess
    x = jnp.zeros((nz, ny, nx))

    # Gradient of the loss function
    grad_loss = grad(loss_fn)

    # Setup the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(x)

    # Optimization loop
    errors = []
    errors.append(loss_fn(x, J, dphi, weights, lambda_tv, lambda_l1, lambda_l2))
    for _ in range(max_iter):
        grads = grad_loss(x, J, dphi, weights, lambda_tv, lambda_l1, lambda_l2)
        updates, opt_state = optimizer.update(grads, opt_state)
        x = optax.apply_updates(x, updates)

        # Compute the error
        error = loss_fn(x, J, dphi, weights, lambda_tv, lambda_l1, lambda_l2)
        errors.append(error)

    print(f"Error: {errors[-1]:.2e}")

    return x, errors


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
