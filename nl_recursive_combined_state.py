"""
Simulation setup for satellite formation flying using recursive filter.
"""

import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from tqdm import tqdm
import yaml

from analysis import fpost_sanity_check
from analysis import get_cov_trace
from analysis import get_crb_trace
from sat.construct_jacobian import combined_H
from sat.dynamics import exchange_trajectories
from sat.dynamics import f
from sat.dynamics import f_jac
from sat.dynamics import simulate_nominal_trajectories
from utils.config_utils import load_sat_config
from utils.data_io_utils import import_landmarks
from utils.data_io_utils import store_all_data
from utils.data_io_utils import setup_data_dir
from utils.math_utils import M_Jac
from utils.plotting_utils import all_sat_position_error
from utils.plotting_utils import plot_all_sat_crb_trace
from utils.plotting_utils import plot_trajectory
from utils.yaml_autogen_utils import generate_satellites_yaml


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

    N = args.N
    dt = 1 / args.f  # Hz
    n_sats = args.n_sats
    state_dim = args.state_dim
    num_trials = args.num_trials
    meas_type = args.measurement_type

    # Process noise covariance matrix based on paper "Autonomous orbit determination and observability analysis for formation satellites"
    # by OU Yangwei, ZHANG Hongbo, XING Jianjun page 6
    Q = np.diag(np.array([1e-9, 1e-9, 1e-9, 1e-12, 1e-12, 1e-12]))
    Q_block = block_diag(*[Q for _ in range(n_sats)])
    ind_cov = np.diag(
        np.array([10, 10, 10, 1e-4, 1e-4, 1e-4])
    )  # Individual covariance matrix for each satellite

    ### Satellite Initialization ###
    sats = load_sat_config(args=args)

    x_traj = simulate_nominal_trajectories(N, dt, sats, state_dim)

    # Transfer the positions of the other satellites for each satellite at N+1 timesteps
    sats = exchange_trajectories(sats, x_traj)

    ## Calculate FIM in recursive fashion.
    fim = np.zeros((num_trials, N, state_dim * n_sats, state_dim * n_sats))
    cov_hist = np.zeros((num_trials, N, state_dim * n_sats, state_dim * n_sats))

    sats_copy = copy.deepcopy(sats)

    filter_position = np.zeros((num_trials, N, 3 * n_sats))
    pos_error = np.zeros((num_trials, N, 3 * n_sats))

    for trial in tqdm(range(num_trials), desc=f"Monte Carlo for {n_sats} sat"):            

        f_prior = np.zeros((state_dim * n_sats, state_dim * n_sats))
        f_post = np.zeros((state_dim * n_sats, state_dim * n_sats))

        # Initialize the combined state vector and covariance matrices
        # TODO: Complexify this function to handle different variance weights for different satellites
        cov_m = block_diag(*[ind_cov for _ in range(n_sats)])
        cov_p = block_diag(*[ind_cov for _ in range(n_sats)])

        ### USE SAT 0 AS ANCHOR ###
        # Set the initial covariance of the first satellite to be very small
        # Q_block[0:6,0:6] = 1e-20*np.eye(6)

        total_x_m = np.zeros((state_dim * n_sats))
        total_x_p = np.zeros((state_dim * n_sats))

        comb_curr_pos = np.zeros((3 * n_sats))

        A = np.zeros((n_sats * state_dim, n_sats * state_dim))

        # Initialize the measurement states using satellites initial measurement state
        for i in range(n_sats):
            total_x_m[i * state_dim : (i + 1) * state_dim] = sats[i].x_m

        # Looping for timesteps
        for i in tqdm(range(N), desc="Timesteps"):   
            print("Timestep", i)    

            for k, sat in enumerate(sats_copy):
                # Provide the underlying groundtruth position to the satellite for bearing and ranging measurements

                ### USE SAT 0 AS ANCHOR ###
                # if sat.id == 0:
                #     sat.x_m = x_traj[i, :, 0]
                #     cov_m[0:6,0:6] = 1e-20*np.eye(6)
                #     cov_p[0:6,0:6] = 1e-20*np.eye(6)

                sat.curr_pos = x_traj[i + 1, 0:3, k]
                sat.x_p = f(sat.x_m, dt)

                # Assign the state transition matrix to the correct block in the A matrix
                A[k * state_dim : (k + 1) * state_dim,
                  k * state_dim : (k + 1) * state_dim,
                ] = f_jac(sat.x_m, dt)

                # Update the combined state vector and underlying groundtruth
                total_x_p[k * state_dim : (k + 1) * state_dim] = sat.x_p
                comb_curr_pos[k * 3 : (k + 1) * 3] = sat.curr_pos

            # FIM Calculations
            D11 = A.T @ np.linalg.inv(Q_block) @ A
            D12 = -A.T @ np.linalg.inv(Q_block)

            f_prior = D12.T @ np.linalg.inv(f_post + D11) @ D12

            # Update the combined covariance matrix
            cov_p = A @ cov_m @ A.T + Q_block

            comb_y_m = []  # Combined measurement vector
            comb_h = []  # Combined estimation vector
            comb_H = np.array([])  # Combined Jacobian matrix

            R_vec = np.array([])  # Combined measurement noise vector
            # M_vec = [] # Combined Jacobian matrix for the process noise

            for sat in sats_copy:
                
                ### USE SAT 0 AS ANCHOR ###
                # Skip the first satellite as we assume it has perfect knowledge
                # if sat.id == 0:
                #     continue

                # Get visible landmarks using actual current position of other satellites
                visible_landmarks = sat.visible_landmarks_list()
                visible_sats = sat.visible_sats_list(sats_copy)

                if "land" in meas_type:
                    sat.land_bearing_dim = len(visible_landmarks) * 3

                if "sat_bearing" in meas_type:
                    sat.sat_bearing_dim = len(visible_sats) * 3

                if "range" in meas_type:
                    sat.range_dim = len(visible_sats)
                
                meas_dim = sat.land_bearing_dim + sat.sat_bearing_dim + sat.range_dim

                # Re-initialize the measurement matrices for each satellite with the correct dimensions
                if meas_dim > 0:
                    y_m = []
                    h = []

                    # Calculate Jacobian matrix H for combined state (still just one satellite H)
                    H = combined_H(sat, meas_dim, state_dim, meas_type)
                    
                    if "land" in meas_type and sat.land_bearing_dim > 0:
                        h.extend(sat.h_landmark(sat.x_p[0:3]).tolist())
                        y_m.extend(sat.measure_z_landmark().tolist())
                        R_vec = np.append(R_vec, [sat.R_weight_land_bearing] * sat.land_bearing_dim)
                        # M_vec.append(M_Jac(y_m[0 : sat.land_bearing_dim]))

                    if "sat_bearing" in meas_type and sat.sat_bearing_dim > 0:
                        h[sat.land_bearing_dim : sat.sat_bearing_dim + sat.land_bearing_dim] = sat.h_sat_bearing(sat.x_p[0:3])
                        y_m[sat.land_bearing_dim : sat.sat_bearing_dim + sat.land_bearing_dim] = sat.measure_z_sat_bearing()
                        R_vec = np.append(R_vec, [sat.R_weight_sat_bearing] * sat.sat_bearing_dim)
                        # M_vec.append(M_Jac(y_m[sat.land_bearing_dim : sat.sat_bearing_dim + sat.land_bearing_dim]))

                    if "range" in meas_type and sat.range_dim > 0:
                        h.extend(sat.h_inter_range(sat.x_p[0:3]).tolist())
                        y_m.extend(sat.measure_z_range().tolist())
                        R_vec = np.append(R_vec, [  sat.R_weight_range] * sat.range_dim)
                        # M_vec.append(np.eye((sat.range_dim)))

                    # Append vectors and matrices to combined form
                    comb_y_m.extend(y_m)
                    comb_h.extend(h)

                    if comb_H.size == 0:
                        comb_H = H
                    else:
                        comb_H = np.append(comb_H, H, axis=0)

                elif meas_dim == 0:
                    continue

            # Create R based on the number of measurements of all satellites
            R = np.diag(R_vec)
            # M_val = block_diag(*M_vec)
            comb_y_m = np.array(comb_y_m).reshape(-1)
            comb_h = np.array(comb_h).reshape(-1)


            # Kalman Gain TODO: Consider the M matrix to include with the process noise depends on how the noise is modelled. 
            # K2 = cov_p @ comb_H.T @ np.linalg.inv(comb_H @ cov_p @ comb_H.T + M_val @ R @ M_val.T)

            if np.any(np.abs(comb_H) > 1e-18):
                K = cov_p @ comb_H.T @ np.linalg.inv(comb_H @ cov_p @ comb_H.T + R)
                tmp = comb_y_m - comb_h
                
                total_x_m = total_x_p + K @ tmp
                cov_m = (np.eye(state_dim * n_sats) - K @ comb_H) @ cov_p @ (
                    (np.eye(state_dim * n_sats) - K @ comb_H).T
                ) + K @ R @ K.T

                f_post = (
                    f_prior + comb_H.T @ np.linalg.inv(R) @ comb_H + np.linalg.inv(Q_block)
                )

            else: 
                # No measurements so just set the prior as the posterior
                total_x_m = total_x_p
                cov_m = cov_p
                f_post = f_prior

            # Posterior Update

            # Set sat's x_m so that they can be used for the next prior update x_p state.
            # Individual covariances of sats don't matter for this because we use the full covariances
            for sat in sats_copy:
                sat.x_m = total_x_m[sat.id * state_dim : (sat.id + 1) * state_dim]
                sat.cov_m = cov_m[sat.id * state_dim : (sat.id + 1) * state_dim, sat.id * state_dim : (sat.id + 1) * state_dim]

            # FIM Calculation
            print(np.linalg.cond(f_post))
            # Assign Posterior Covariance
            cov_hist[trial, i, :, :] = cov_m

            filter_position[trial, i, :] = (
                np.array([total_x_m[0::state_dim], total_x_m[1::state_dim], total_x_m[2::state_dim]])
                .transpose()
                .reshape(-1)
            )
            pos_error[trial, i, :] = filter_position[trial, i, :] - comb_curr_pos

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
    parser.add_argument("--N", type=int, default=100, help="Number of timesteps")
    parser.add_argument(
        "--f", type=float, default=1, help="Frequency of the simulation"
    )
    parser.add_argument(
        "--ignore_earth",
        action="store_true",
        default=False,
        help="Ignore the Earth from blocking measurements. Only applies to range measurements. \
                        Bearing measurements always consider the earth.",
    )
    parser.add_argument(
        "--num_trials", type=int, default=1, help="Number of Monte Carlo trials"
    )
    parser.add_argument("--n_sats", type=int, default=1, help="Number of satellites")
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
        "--state_dim", type=int, default=6, help="Dimension of the state vector"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Print information"
    )
    parser.add_argument(
        "--measurement_type", nargs="+", help="Type of measurements to be used \
            Options are 'range', 'land', 'sat_bearing'"
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

    if args.run_all:
        for i in range(1, args.n_sats + 1):
            args.n_sats = i
            run_simulation(args)
    else:
        run_simulation(args)

    plot_all_sat_crb_trace()
    plt.show()
