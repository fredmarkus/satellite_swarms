"""
Construct the Jacobian matrix/matrices for the desired problem.
"""

import numpy as np

# pylint: disable=import-error
# pylint: disable=invalid-name
from sat.core import satellite

def combined_state_H(sat: satellite,  meas_dim: int, state_dim: int, timestep: int, k: int) -> np.ndarray:

    """
    Calculate the Jacobian matrix for a single satellite using the combined measurement model.

    Args:
        sat (satellite): The satellite instance.
        meas_dim (int): The dimension of the measurement vector.
        state_dim (int): The dimension of the state vector.
        timestep (int): The current timestep.
        k (int): The current satellite index.

    Returns:
        np.ndarray: The Jacobian matrix for the satellite.
    """

    H = np.zeros((meas_dim, state_dim * sat.n_sats))
    H[0 : sat.bearing_dim, k * state_dim : (k + 1) * state_dim] = (
        sat.H_landmark(sat.x_p)
    )

    opp_sat_id = 0
    for j in range(sat.bearing_dim, meas_dim):
        dist = sat.H_inter_range(timestep + 1, j, sat.x_p)
        H[j, sat.id * state_dim : (sat.id + 1) * state_dim] = dist

        if opp_sat_id == sat.id:
            opp_sat_id += 1
            H[j, opp_sat_id * state_dim : (opp_sat_id + 1) * state_dim] = -dist
        else:
            H[j, opp_sat_id * state_dim : (opp_sat_id + 1) * state_dim] = -dist
            opp_sat_id += 1

    return H
