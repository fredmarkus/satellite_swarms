"""
Simulation setup for satellite formation flying using recursive filter.
"""

import argparse
import copy
import os
import sys

import brahe
from brahe.epoch import Epoch
import jax.numpy as jnp

submodule_path = os.path.join(os.path.dirname(__file__), './gnc_payload')
sys.path.append(submodule_path)

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from tqdm import tqdm
import yaml

from analysis import fpost_sanity_check
from analysis import get_cov_trace
from analysis import get_crb_trace
from sat.construct_jacobian import combined_H
from utils.core_config_utils import load_sat_config
from utils.data_io_utils import import_landmarks
from utils.data_io_utils import store_all_data
from utils.data_io_utils import setup_data_dir
from utils.math_utils import M_Jac
from utils.math_utils import transform_eci_to_lvlh
from utils.plotting_utils import all_sat_position_error
from utils.plotting_utils import plot_all_sat_crb_trace
from utils.plotting_utils import plot_trajectory
from utils.yaml_autogen_utils import generate_satellites_yaml

from utils.config_utils import load_config
from gnc_payload.utils.orbit_utils import get_max_sso_latitude
from gnc_payload.sensors.camera_model import CameraModelManager
from gnc_payload.utils.brahe_utils import load_brahe_data_files


import pickle
from time import time

import brahe
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from brahe.epoch import Epoch

from gnc_payload.dynamics.orbital_dynamics import Dynamics
from gnc_payload.sensors.camera_model import CameraModelManager
from gnc_payload.utils.brahe_utils import load_brahe_data_files_if_needed


def run_simulation(args):
    """
    Run a Monte Carlo simulation for satellite formation flying.
    Calculate the Fisher Information Matrix (FIM) and Cramer-Rao Bound (CRB).
    The function calculates the FIM and CRB for each timestep and trial, and averages the results over all trials.
    It also calculates the positional error for each satellite and saves the results to files.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - N (int): Number of timesteps.
            - f (float): Frequency in Hz.
            - n_sats (int): Number of satellites.
            - state_dim (int): Dimension of the state vector.
            - num_trials (int): Number of Monte Carlo trials.
            - landmark_objects (list): List of landmark objects.
            - random_yaml (bool): Flag to indicate whether to use a random YAML configuration.
            - verbose (bool): Flag to enable verbose output.
            - ignore_earth (bool): Flag to ignore Earth in the simulation.
            - measurement_type (list): List of measurement types to be used.
            - anchor (bool): Flag to indicate whether to use an anchor satellite (will be satellite with id 0)

    Raises:
        ValueError: If the number of satellites specified is greater than the number of satellites in the YAML file.

    Returns:
        None

    This function performs the following steps:
        1. Load satellite configuration from a YAML file and create satellite instances.
        2. Generate nominal trajectories for the satellites.
        3. Perform Monte Carlo simulations using recursive filter to generate FIM.
        4. Save and plot the results
    """

    # N = args.N
    # dt = 1 / args.f  # Hz
    n_sats = args.n_sats
    state_dim = args.state_dim
    num_trials = args.num_trials
    meas_type = args.measurement_type
    anchor = args.anchor

    # Process noise covariance matrix based on paper "Autonomous orbit determination and observability analysis for formation satellites"
    # by OU Yangwei, ZHANG Hongbo, XING Jianjun page 6
    Q = np.eye(15) * 1e-12
    # Unmodelled acceleration has larger uncertainty
    Q[6:9, 6:9] = np.eye(3) * 1e-9
    # Bias uncertainty also larger
    Q[12:15, 12:15] = np.eye(3) * 1e-9  
  
    Q_block = block_diag(*[Q for _ in range(n_sats)])
    # Individual covariance matrix for each satellite
    ind_cov = np.diag(
        np.array([5, 5, 5, 5, 5, 5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    )  

    ### Satellite Initialization ###
    # TODO: Improve orbits to use actual sso orbits
    sats = load_sat_config(args=args, Q_noise=Q)

    config = load_config()
    config["solver"]["world_update_rate"] = 2  # Hz
    config["mission"]["duration"] = 3 * 90 * 20  # s

    dt = 1 / config["solver"]["world_update_rate"]
    # starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
    N = int(np.ceil(config["mission"]["duration"] / dt))  # number of time steps in the simulation

    ground_truth_dynamics = Dynamics(
        config=config,
        use_drag=True,
        use_j2=True,
    )

    # Number of iterations on the IEKF
    num_iterations = 1

    # Angular velocity
    w = np.array([0, 0, np.pi / 18])

    ## Calculate FIM in recursive fashion.
    fim = np.zeros((num_trials, N, state_dim * n_sats, state_dim * n_sats))
    cov_hist = np.zeros((num_trials, N, state_dim * n_sats, state_dim * n_sats))

    sats_copy = copy.deepcopy(sats)

    filter_position = np.zeros((num_trials, N, 3 * n_sats))
    pos_error = np.zeros((num_trials, N, 3 * n_sats))

    # Conditioning threshold for the S matrix where regularization is applied
    cond_threshold = 1e15

    for trial in tqdm(range(num_trials), desc=f"Monte Carlo for {n_sats} sat"):            

        f_prior = np.zeros((state_dim * n_sats, state_dim * n_sats))
        f_post = np.zeros((state_dim * n_sats, state_dim * n_sats))

        # Initialize the combined state vector and covariance matrices
        # TODO: Complexify this function to handle different variance weights for different satellites
        cov_m = block_diag(*[ind_cov for _ in range(n_sats)])
        cov_p = block_diag(*[ind_cov for _ in range(n_sats)])

        # Set the initial covariance of the first satellite to be very small when using anchor mode
        if anchor:
            Q_block[0:6,0:6] = 1e-20*np.eye(6)

        total_x_m = np.zeros((state_dim * n_sats))
        total_x_p = np.zeros((state_dim * n_sats))

        comb_curr_pos = np.zeros((3 * n_sats))
        lvlh_curr_pos = np.zeros((3 * n_sats))

        A = np.zeros((n_sats * state_dim, n_sats * state_dim))

        # Initialize the measurement states using satellites initial measurement state
        for i in range(n_sats):
            total_x_m[i * state_dim : (i + 1) * state_dim] = np.concatenate([sats[i].r_m, sats[i].v_m, sats[i].ua, quaternion.as_rotation_vector(quaternion.as_quat_array(sats[i].q_m)), sats[i].w_b])

        # Looping for timesteps
        for i in tqdm(range(N), desc="Timesteps"):
            if args.verbose:   
                print("Timestep", i)
            
            latest_epoch = sats[0].data_manager.latest_epoch
            rot = w  + 0.05 * np.array([np.cos(2*np.pi * i / (10 / dt)), np.sin(2*np.pi * i / (10 / dt)), 0])

            for k, sat in enumerate(sats_copy):

                if sat.id == 0 and anchor:
                    #TODO: Replace the x_m for the anchor case
                    sat.x_m = ground_truth_dynamics.perturbed_f(sat[0].curr_pos, dt, epoch=latest_epoch)
                    cov_m[0:6,0:6] = 1e-20*np.eye(6)
                    cov_p[0:6,0:6] = 1e-20*np.eye(6)

                # Provide the underlying groundtruth position to the satellite for bearing and ranging measurements
                # x_gt = np.concatenate([sat.r_gt, sat.v_gt])
                x_gt = sat.data_manager.latest_state[0:6]
                q = sat.data_manager.latest_attitude

                next_state = ground_truth_dynamics.perturbed_f(x=x_gt,dt=dt, epoch=latest_epoch)
                next_quat = quaternion.from_rotation_matrix(q) * quaternion.from_rotation_vector(w * dt)
                sat.data_manager.push_next_state(next_state[0:6], quaternion.as_rotation_matrix(next_quat))

                # Get an IMU measurement for the satellite
                gyro_meas, _ = sat.imu.update(rot, np.zeros((3,)))
                # imu_gyro_bias = sat.imu.get_bias()[0]

                sat.predict(u=gyro_meas,epoch=latest_epoch)

                # Assign the state transition matrix to the correct block in the A matrix
                A[k * state_dim : (k + 1) * state_dim,
                  k * state_dim : (k + 1) * state_dim,
                ] = sat.cov_p

                # Update the combined state vector and underlying groundtruth
                total_x_p[k * state_dim : (k + 1) * state_dim] = sat.x_p
                comb_curr_pos[k * 3 : (k + 1) * 3] = sat.data_manager.latest_state[0:3]

            # FIM Calculations
            D11 = A.T @ np.linalg.inv(Q_block) @ A
            D12 = -A.T @ np.linalg.inv(Q_block)

            f_prior = D12.T @ np.linalg.inv(f_post + D11) @ D12

            # Update the combined covariance matrix
            cov_p = A @ cov_m @ A.T + Q_block

            comb_y_m = []  # Combined measurement vector
            R_vec = np.array([])  # Combined measurement noise vector
            # M_vec = [] # Combined Jacobian matrix for the process noise

            if i % 120 == 0:
                for sat in sats_copy:
                    
                    # Skip the first satellite as we assume it has perfect knowledge
                    if sat.id == 0 and anchor:
                        continue

                    # Get visible landmarks using actual current position of other satellites
                    # visible_landmarks = sat.visible_landmarks_list()
                    visible_sats = sat.visible_sats_list(sats_copy)

                    if "land" in meas_type:
                        for camera_name in CameraModelManager.CAMERA_NAMES:
                            sat.data_manager.take_measurement(
                                sat.landmark_bearing_sensor,sat.camera_model_manager[camera_name]
                            )

                        measurement_camera_names, *z = sat.data_manager.latest_measurements

                        # # Apply mask for fraction of used measurements
                        mask = np.random.choice([True, False], size=z[0].shape[0], p=[0.04, 0.96])
                        z0 = z[0][mask]
                        sat.z1 = z[1][mask]
                        sat.measurement_camera_names = measurement_camera_names[mask]
                        # Flatten the measurement vector
                        z0 = np.array(z0.reshape(-1))

                        sat.land_bearing_dim = z0.shape[0]

                    if "sat_bearing" in meas_type:
                        sat.sat_bearing_dim = len(visible_sats) * 3

                    if "range" in meas_type:
                        sat.range_dim = len(visible_sats)
                    
                    sat.meas_dim = sat.land_bearing_dim + sat.sat_bearing_dim + sat.range_dim

                    # Re-initialize the measurement matrices for each satellite with the correct dimensions
                    if sat.meas_dim > 0:
                        y_m = []

                        if "land" in meas_type and sat.land_bearing_dim > 0:
                            # h.extend(sat.h_landmark(sat.x_p[0:3]).tolist())
                            # y_m.extend(sat.measure_z_landmark().tolist())
                            y_m.extend(z0.tolist())
                            R_vec = np.append(R_vec, [sat.R_weight_land_bearing] * sat.land_bearing_dim)
                            # M_vec.append(M_Jac(y_m[0 : sat.land_bearing_dim]))

                        if "sat_bearing" in meas_type and sat.sat_bearing_dim > 0:
                            # h[sat.land_bearing_dim : sat.sat_bearing_dim + sat.land_bearing_dim] = sat.h_sat_bearing(sat.x_p[0:3])
                            y_m[sat.land_bearing_dim : sat.sat_bearing_dim + sat.land_bearing_dim] = sat.measure_z_sat_bearing()
                            R_vec = np.append(R_vec, [sat.R_weight_sat_bearing] * sat.sat_bearing_dim)
                            # M_vec.append(M_Jac(y_m[sat.land_bearing_dim : sat.sat_bearing_dim + sat.land_bearing_dim]))

                        if "range" in meas_type and sat.range_dim > 0:
                            # h.extend(sat.h_inter_range(sat.x_p[0:3]).tolist())
                            y_m.extend(sat.measure_z_range().tolist())
                            R_vec = np.append(R_vec, [sat.R_weight_range] * sat.range_dim)
                            # M_vec.append(np.eye((sat.range_dim)))


                        # Append vectors and matrices to combined form
                        comb_y_m.extend(y_m)
                        # comb_h.extend(h)

                        # if comb_H.size == 0:
                        #     comb_H = H
                        # else:
                        #     comb_H = np.append(comb_H, H, axis=0)

                    elif sat.meas_dim == 0:
                        continue

                # Create R based on the number of measurements of all satellites
                R = np.diag(R_vec)
                # M_val = block_diag(*M_vec)
                comb_y_m = np.array(comb_y_m).reshape(-1)

                for j in range(num_iterations):
                    # Reset estimation arrays
                    comb_h = [] 
                    comb_H = np.array([])
                    for sat in sats_copy:
                        if j == 0:
                            sat.x_p = jnp.array(
                                np.concatenate(
                                    [
                                        sat.r_p,
                                        sat.v_p,
                                        sat.ua,
                                        quaternion.as_rotation_vector(quaternion.as_quat_array(sat.q_p)),
                                        sat.w_b,
                                    ]
                                )
                            )
                        else:
                            # set the updated state from the previous iteration
                            sat.x_p = jnp.array(
                                np.concatenate(
                                    [
                                        sat.r_m,
                                        sat.v_m,
                                        sat.ua,
                                        quaternion.as_rotation_vector(quaternion.as_quat_array(sat.q_m)),
                                        sat.w_b,
                                    ]
                                )
                            )

                        # Update the satellite estimates and use those in the combined state
                        h = []
                        if "land" in meas_type and sat.land_bearing_dim > 0:
                            # h.extend(sat.h_landmark(sat.x_p[0:3]).tolist())
                            h.extend(sat.h_landmark_actual(sat.z1, sat.camera_model_manager, sat.measurement_camera_names, sat.x_p, epoch=latest_epoch).tolist())
                            # M_vec.append(M_Jac(y_m[0 : sat.land_bearing_dim]))

                        if "sat_bearing" in meas_type and sat.sat_bearing_dim > 0:
                            h.extend(sat.h_sat_bearing(sat.x_p[0:3].tolist()))
                            # M_vec.append(M_Jac(y_m[sat.land_bearing_dim : sat.sat_bearing_dim + sat.land_bearing_dim]))

                        if "range" in meas_type and sat.range_dim > 0:
                            h.extend(sat.h_inter_range(sat.x_p[0:3]).tolist())
                            # M_vec.append(np.eye((sat.range_dim)))

                        comb_h.extend(h)
                        # Calculate Jacobian matrix H for combined state (still just one satellite H)
                        H = combined_H(sat, sat.meas_dim, state_dim, meas_type, sat.z1, sat.camera_model_manager, sat.measurement_camera_names, sat.x_p, epoch=latest_epoch)
                        if comb_H.size == 0:
                                comb_H = H
                        else:
                            comb_H = np.append(comb_H, H, axis=0)

                    # Kalman Gain TODO: Consider the M matrix to include with the process noise depends on how the noise is modelled. 
                    # K2 = cov_p @ comb_H.T @ np.linalg.inv(comb_H @ cov_p @ comb_H.T + M_val @ R @ M_val.T)

                    S = comb_H @ cov_p @ comb_H.T + R
                    if j == 0:
                        cond = np.linalg.cond(S)
                        print(cond)
                        if cond > cond_threshold:
                            S += np.eye(S.shape[0]) * 1e-6
                            print("Ill-conditioned matrix detected. Regularization applied.")

                    K = cov_p @ comb_H.T @ np.linalg.inv(S)
                    delta = comb_y_m - comb_h

                    for k, sat in enumerate(sats_copy):
                        # Update the combined state vector
                        sat.r_m = sat.x_p[0:3] + K[k * state_dim : (k * state_dim) + 3, :] @ delta
                        sat.v_m = sat.x_p[3:6] + K[(k * state_dim) + 3 : (k * state_dim) + 6, :] @ delta
                        sat.ua = sat.x_p[6:9] + K[(k * state_dim) + 6 : (k * state_dim) + 9, :] @ delta
                        sat.q_m = quaternion.as_rotation_vector(
                        quaternion.from_rotation_vector(np.array(sat.x_p[9:12]))
                        * quaternion.from_rotation_vector(K[(k * state_dim) + 9 : (k * state_dim) + 12, :] @ delta))
                        sat.w_b = sat.x_p[12:15] + K[(k * state_dim) + 12 : (k * state_dim) + 15, :] @ delta

                        # total_x_m[i * state_dim : (i + 1) * state_dim] = (
                    
                    
                # Update final "total" state vector and covariance matrix cov_m
                cov_m = (np.eye(state_dim * n_sats) - K @ comb_H) @ cov_p @ (
                    (np.eye(state_dim * n_sats) - K @ comb_H).T
                ) + K @ R @ K.T

                f_post = (
                    f_prior + comb_H.T @ np.linalg.inv(R) @ comb_H + np.linalg.inv(Q_block)
                )

                # else: 
                #     # No measurements so just set the prior as the posterior
                #     total_x_m = total_x_p
                #     cov_m = cov_p
                #     f_post = f_prior

                # Set sat's x_m so that they can be used for the next prior update x_p state.
                # Individual covariances of sats don't matter for this because we use the full covariances
                for sat in sats_copy:
                    total_x_m[sat.id * state_dim : sat.id * state_dim + 3] = sat.r_m
                    total_x_m[sat.id * state_dim + 3 : sat.id * state_dim + 6] = sat.v_m
                    total_x_m[sat.id * state_dim + 6 : sat.id * state_dim + 9] = sat.ua
                    total_x_m[sat.id * state_dim + 9 : sat.id * state_dim + 12] = sat.q_m
                    total_x_m[sat.id * state_dim + 12 : sat.id * state_dim + 15] = sat.w_b

                    sat.cov_m = cov_m[sat.id * state_dim : (sat.id + 1) * state_dim, sat.id * state_dim : (sat.id + 1) * state_dim]
                    
                    # Convert final iterated rotation vector to quaternion
                    sat.q_m = quaternion.as_float_array(quaternion.from_rotation_vector(sat.q_m))
            
            # Set current prior as the current posterior
            else:
                for sat in sats_copy:
                    sat.r_m = sat.r_p
                    sat.v_m = sat.v_p
                    sat.q_m = sat.q_p    
                    # Ua and gyro bias are not updated in the prior step     
                
            # FIM Calculation

            # Assign Posterior Covariance
            cov_hist[trial, i, :, :] = cov_m

            filter_position[trial, i, :] = (
                np.array([total_x_m[0::state_dim], total_x_m[1::state_dim], total_x_m[2::state_dim]])
                .transpose()
                .reshape(-1)
            )

            # Convert to LVLH frame
            for j in range(n_sats):
                # R_curr_pos = transform_eci_to_lvlh(x_traj[i + 1, 0:3, j],x_traj[i + 1, 3:6, j])
                R_filter = transform_eci_to_lvlh(total_x_m[j*state_dim : j*state_dim+3],total_x_m[j*state_dim + 3 : j*state_dim + 6])

                filter_position[trial, i, j*3:j*3+3] = R_filter.T @ filter_position[trial, i, j*3:j*3+3]
                lvlh_curr_pos[j*3:j*3+3] = R_filter.T @ comb_curr_pos[j*3:j*3+3]

            pos_error[trial, i, :] = filter_position[trial, i, :] - lvlh_curr_pos

            # # Sanity check that Cov - FIM is positive definite (Should always be true)
            fpost_sanity_check(f_post, cov_m, args.verbose, state_dim)

            fim[trial, i, :, :] = f_post

        # Reset the satellites to initial condition for the next trial
        sats_copy = copy.deepcopy(
            sats
        )  

    # Average history of relevant variables
    fim = np.mean(fim, axis=0)
    cov_hist = np.mean(cov_hist, axis=0)
    pos_error = np.mean(pos_error, axis=0)

    # Calculate Covariance and CRB trace
    cov_trace = get_cov_trace(N, cov_hist, n_sats)
    crb_trace = get_crb_trace(N, fim, n_sats)


    # Save the results to files
    store_all_data(
        n_sats=n_sats,
        cov_trace=cov_trace,
        crb_trace=crb_trace,
        pos_error=pos_error,
        sats=sats,
    )

    # Plotting of errors
    all_sat_position_error(pos_error, n_sats, meas_type, cov_hist)

    # Plot filter trajectories
    # plot_trajectory(x_traj[:,:,0], filter_position[0,:,0:3], N)



if __name__ == "__main__":

    #Setup data directory
    setup_data_dir()
    
    ### Landmark Initialization ###
    landmark_objects = import_landmarks()

    # General Parameters
    parser = argparse.ArgumentParser(
        description="Nonlinear Recursive Monte Carlo Simulation"
    )
    parser.add_argument(
        "--N", 
        type=int,
        default=100,
        help="Number of timesteps")
    parser.add_argument(
        "--f", 
        type=float,
        default=1,
        help="Frequency of the simulation"
    )
    parser.add_argument(
        "--ignore_earth",
        action="store_true",
        default=False,
        help="Ignore the Earth from blocking measurements. Only applies to range measurements. \
                        Bearing measurements always consider the earth.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of Monte Carlo trials"
    )
    parser.add_argument(
        "--n_sats",
        type=int,
        default=1,
        help="Number of satellites")
    parser.add_argument(
        "--random_yaml",
        action="store_true",
        default=False,
        help="Use random satellite configuration",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        default=False,
        help="Run simulations for all number of satellites from 1 to n_sats",
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=15,
        help="Dimension of the state vector"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print information"
    )
    parser.add_argument(
        "--measurement_type",
        nargs="+",
        help="Type of measurements to be used \
            Options are 'range', 'land', 'sat_bearing'"
    )
    parser.add_argument(
        "--anchor",
        action="store_true",
        default=False,
        help="Use first satellite as anchor",
    )
    args = parser.parse_args()

    if args.random_yaml:
        if not os.path.exists("config"):
            os.makedirs("config")
        generate_satellites_yaml(filename="config/sat_autogen.yaml", n_sats=args.n_sats)

    args.landmark_objects = landmark_objects

    # Check if random_yaml not set that the number of satellites specified is less than or equal to the number of satellites in the yaml config file
    if not args.random_yaml:
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

            if len(config["satellites"]) < args.n_sats:
                raise ValueError(
                    """Number of satellites specified is greater than the number of satellites in the yaml file. 
                    Add --random_yaml flag to generate random satellite configuration for the provided number of 
                    satellites or create custom satellites in config/config.yaml"""
                )

    # load_brahe_data_files()

    if args.run_all:
        for i in range(1, args.n_sats + 1):
            args.n_sats = i
            run_simulation(args)
    else:
        run_simulation(args)

    plot_all_sat_crb_trace()
    plt.show()
