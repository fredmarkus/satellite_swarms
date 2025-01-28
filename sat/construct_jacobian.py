"""
Construct the Jacobian matrix/matrices for the desired problem.
"""

import numpy as np

# pylint: disable=import-error
# pylint: disable=invalid-name
from sat.core import satellite

def combined_state_H_landmark_range(sat: satellite,  meas_dim: int, state_dim: int, timestep: int) -> np.ndarray:

    """
    Calculate the Jacobian matrix for a single satellite using the combined measurement model.

    Args:
        sat (satellite): The satellite instance.
        meas_dim (int): The dimension of the measurement vector.
        state_dim (int): The dimension of the state vector.
        timestep (int): The current timestep.

    Returns:
        np.ndarray: The Jacobian matrix for the satellite.
    """

    H = np.zeros((meas_dim, state_dim * sat.n_sats))
    H[0 : sat.bearing_dim, sat.id * state_dim : (sat.id + 1) * state_dim] = (
        sat.H_landmark(sat.x_p)
    )

    opp_sat_id = 0
    for j in range(sat.bearing_dim, meas_dim):
        dist = sat.H_inter_range(timestep + 1, j, sat.x_p)
        H[j, sat.id * state_dim : (sat.id + 1) * state_dim] = dist

        if opp_sat_id == sat.id:
            opp_sat_id += 1
            H[j, opp_sat_id * state_dim : (opp_sat_id + 1) * state_dim] = -dist
            opp_sat_id += 1
        else:
            H[j, opp_sat_id * state_dim : (opp_sat_id + 1) * state_dim] = -dist
            opp_sat_id += 1

    return H


def combined_state_H_bearing_range(sat: satellite,  meas_dim: int, state_dim: int, timestep: int) -> np.ndarray:

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

    opp_sat_id = 0
    for j in range(0, int(sat.bearing_dim/3)):
        bearing = sat.H_sat_bearing(timestep + 1, j, sat.x_p)
        H[j*3 : (j+1)*3, sat.id * state_dim : (sat.id + 1) * state_dim] = (
            bearing
        )

        if opp_sat_id == sat.id:
            opp_sat_id += 1
            H[j*3:(j+1)*3, opp_sat_id * state_dim : (opp_sat_id + 1) * state_dim] = -bearing
            opp_sat_id += 1

        else:
            H[j*3:(j+1)*3, opp_sat_id * state_dim : (opp_sat_id + 1) * state_dim] = -bearing
            opp_sat_id += 1

    opp_sat_id = 0
    for j in range(sat.bearing_dim, meas_dim):
        dist = sat.H_inter_range(timestep + 1, j, sat.x_p)
        H[j, sat.id * state_dim : (sat.id + 1) * state_dim] = dist

        if opp_sat_id == sat.id:
            opp_sat_id += 1
            H[j, opp_sat_id * state_dim : (opp_sat_id + 1) * state_dim] = -dist
            opp_sat_id += 1
        else:
            H[j, opp_sat_id * state_dim : (opp_sat_id + 1) * state_dim] = -dist
            opp_sat_id += 1

    return H
