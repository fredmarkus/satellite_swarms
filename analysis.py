"""
Information theoretic analysis of the satellite network.
"""

import numpy as np

from sat.core import satellite

def fpost_sanity_check(f_post: np.ndarray, sat: satellite, verbose: bool, state_dim: int):
    """
    Perform a sanity check on the Posterior FIM of the satellite. The FIM should be positive definite.\

    Args:
        f_post (np.ndarray): The Posterior FIM of the satellite.
        sat (satellite): The satellite object.
        verbose (bool): Whether to print the FIM and its properties.
        state_dim (int): The dimension of the state vector.

    Raises:
        ValueError: If the FIM of the satellite is not positive definite. 
    """

    if np.linalg.matrix_rank(f_post) == state_dim:
        if not np.all(np.linalg.eigvals(sat.cov_m - np.linalg.inv(f_post)) > 0):
            raise ValueError(f"FIM of satellite {sat.id} is not positive definite")

    # # Check if f_post is invertible
    if np.linalg.matrix_rank(f_post) != state_dim:
        if verbose:
            print(f_post)
            print(f"Satellite {sat.id} FIM is not invertible")

    else:
        eig_val, eig_vec = np.linalg.eig(f_post)
        if verbose:
            print(f"Eigenvalues of satellite {sat.id} FIM: ", eig_val)
            print("Condition number of FIM: ", np.linalg.cond(f_post))
            print("Eigenvectors of FIM: ", eig_vec)


def get_cov_trace(timesteps: int, cov_hist: np.ndarray, n_sats: int) -> np.ndarray:
    """
    Calculate the trace of the covariance matrix over all time steps 
    and normalize it by the number of satellites.

    Args:
        timesteps (int): The number of time steps.
        cov_hist (np.ndarray): The covariance matrix over all time steps.
        n_sats (int): The number of satellites.

    Returns:
        np.ndarray: The satellite-normalized trace of the covariance matrix.
    """
    cov_trace = np.zeros((timesteps))
    for i in range(timesteps):
        cov_trace[i] = np.trace(cov_hist[i, :, :]) / n_sats
        # pos_error[i,:] = pos_error[i,:]/n_sats

    return cov_trace


def get_crb_trace(timesteps: int, pfim: np.ndarray, n_sats: int) -> np.ndarray:
    """
    Get the Cramer-Rao Bound (CRB) trace of the Posterior FIM over all time steps
    and normalize it by the number of satellites.

    Args:
        timesteps (int): The number of time steps.
        pfim (np.ndarray): The Posterior Fisher Information Matrix (PFIM).
        n_sats (int): The number of satellites.

    Returns:
        np.ndarray: The satellite-normalized trace of the CRB.
    """

    crb = np.zeros_like(pfim)
    crb_trace = np.zeros((timesteps))
    for i in range(timesteps):
        crb[i, :, :] = np.linalg.inv(pfim[i, :, :])
        crb_trace[i] = np.trace(crb[i, :, :]) / n_sats

    return crb_trace
