"""
Construct the Jacobian matrix/matrices for the desired problem.
"""

import numpy as np

# pylint: disable=import-error
# pylint: disable=invalid-name
from sat.core import satellite

def combined_state_H(sat: satellite,  meas_dim: int, state_dim: int) -> np.ndarray:

    """
    Calculate the Jacobian matrix for a single satellite using the combined measurement model.

    Args:
        sat (satellite): The satellite instance.
        meas_dim (int): The dimension of the measurement vector.
        state_dim (int): The dimension of the state vector.

    Returns:
        np.ndarray: The Jacobian matrix for the satellite.
    """

    H = np.zeros((meas_dim, state_dim * sat.n_sats))
    H[0 : sat.bearing_dim, sat.id * state_dim : (sat.id + 1) * state_dim] = (
        sat.H_landmark(sat.x_p)
    )

    dist_J = sat.H_inter_range(sat.x_p)
    H[sat.bearing_dim:meas_dim,sat.id * state_dim : (sat.id + 1) * state_dim] = dist_J
    for i , other_sat in enumerate(sat.curr_visible_sats):
        H[sat.bearing_dim + i : sat.bearing_dim + i + 1, other_sat.id * state_dim : (other_sat.id + 1) * state_dim] = -dist_J[i, :]

    return H
