"""
Construct the Jacobian matrix/matrices for the desired problem.
"""
from typing import List

import numpy as np
from brahe import Epoch
import jax.numpy as jnp

from gnc_payload.sensors.camera_model import CameraModelManager

# pylint: disable=import-error
# pylint: disable=invalid-name
from sat.core import satellite

def combined_H(sat: satellite, state_dim: int, meas_type: List[str], epoch: Epoch) -> np.ndarray:

    """
    Calculate the Jacobian matrix for a single satellite using the combined measurement model for various measurement types.

    Args:
        sat (satellite): The satellite instance.
        meas_dim (int): The dimension of the measurement vector.
        state_dim (int): The dimension of the state vector.
        meas_type (List[str]): The list of measurement types to be considered.

    Returns:
        np.ndarray: The Jacobian matrix for the satellite.
    """
    H = np.zeros((sat.meas_dim, state_dim * sat.n_sats))

    if "land" in meas_type and sat.land_bearing_dim > 0:
        H[0 : sat.land_bearing_dim, sat.id * state_dim : (sat.id + 1) * state_dim] = (
            sat.H_landmark_actual(sat.z1, sat.camera_model_manager, sat.measurement_camera_names, sat.x_m, epoch)
        )

    if "sat_bearing" in meas_type and sat.sat_bearing_dim > 0:
        bearing_J = sat.H_sat_bearing(sat.x_m)
        H[sat.land_bearing_dim:sat.sat_bearing_dim + sat.land_bearing_dim,sat.id * state_dim : (sat.id + 1) * state_dim] = bearing_J
        for i , other_sat in enumerate(sat.curr_visible_sats):
            H[sat.land_bearing_dim + i*3 : sat.land_bearing_dim + (i+1)*3, other_sat.id * state_dim : (other_sat.id + 1) * state_dim] = -bearing_J[i*3:(i+1)*3, :]

    if "range" in meas_type and sat.range_dim > 0:
        dist_J = sat.H_inter_range(sat.x_m)
        H[sat.sat_bearing_dim + sat.land_bearing_dim:sat.meas_dim,sat.id * state_dim : (sat.id + 1) * state_dim] = dist_J
        for i , other_sat in enumerate(sat.curr_visible_sats):
            H[sat.sat_bearing_dim + sat.land_bearing_dim + i : sat.sat_bearing_dim + sat.land_bearing_dim + i + 1, other_sat.id * state_dim : (other_sat.id + 1) * state_dim] = -dist_J[i, :]

    return H
