import copy
from typing import Dict

import numpy as np
from scipy.linalg import block_diag
from tqdm import tqdm

from analysis import fpost_sanity_check
from analysis import get_cov_trace
from analysis import get_crb_trace
from sat.construct_jacobian import combined_state_H
from sat.dynamics import exchange_trajectories
from sat.dynamics import rk4_discretization
from sat.dynamics import state_transition
from sat.dynamics import simulate_nominal_trajectories
from utils.data_io_utils import store_all_data
from utils.plotting_utils import all_sat_position_error

def run_combined_state_sim(args: Dict) -> None:

    dt = args['dt']
    ind_cov = args['ind_cov']
    N = args['N']
    num_trials = args['num_trials']
    n_sats = args['n_sats']
    Q = args['Q']
    sats = args['sats']
    state_dim = args['state_dim']
    verbose = args['verbose']

    Q_block = block_diag(*[Q for _ in range(n_sats)])
    filter_position = np.zeros((num_trials, N, 3 * n_sats))
    pos_error = np.zeros((num_trials, N, 3 * n_sats))
    
    sats_copy = copy.deepcopy(sats)

    fim = np.zeros((num_trials, N, state_dim * n_sats, state_dim * n_sats))
    cov_hist = np.zeros((num_trials, N, state_dim * n_sats, state_dim * n_sats))

    # Generate ground-truth synthetic data for the next N timesteps
    x_traj = simulate_nominal_trajectories(N, dt, sats, state_dim)

    # Transfer the positions of the other satellites for each satellite at N+1 timesteps
    sats = exchange_trajectories(sats, x_traj)

    for trial in tqdm(range(num_trials), desc=f"Monte Carlo for {n_sats} sat"):

        f_prior = np.zeros((state_dim * n_sats, state_dim * n_sats))
        f_post = np.zeros((state_dim * n_sats, state_dim * n_sats))

        # Initialize the combined state vector and covariance matrices
        # TODO: Complexify this function to handle different variance weights for different satellites
        cov_m = block_diag(*[ind_cov for _ in range(n_sats)])
        cov_p = block_diag(*[ind_cov for _ in range(n_sats)])
        x_m = np.zeros((state_dim * n_sats))
        x_p = np.zeros((state_dim * n_sats))

        comb_curr_pos = np.zeros((3 * n_sats))

        A = np.zeros((n_sats * state_dim, n_sats * state_dim))

        # Initialize the measurement states using satellites initial measurement state
        for i in range(n_sats):
            x_m[i * state_dim : (i + 1) * state_dim] = sats[i].x_m

        # Looping for timesteps
        for i in tqdm(range(N), desc="Timesteps"):

            for k, sat in enumerate(sats_copy):
                sat.curr_pos = x_traj[
                    i + 1, 0:3, sat.id
                ]  # Provide the underlying groundtruth position to the satellite for bearing and ranging measurements
                sat.x_p = rk4_discretization(sat.x_m, dt)

                # Assign the state transition matrix to the correct block in the A matrix
                A[
                    k * state_dim : (k + 1) * state_dim,
                    k * state_dim : (k + 1) * state_dim,
                ] = state_transition(sat.x_m)

                # Update the combined state vector and underlying groundtruth
                x_p[k * state_dim : (k + 1) * state_dim] = sat.x_p
                comb_curr_pos[k * 3 : (k + 1) * 3] = sat.curr_pos

            # FIM Calculations
            D11 = A.T @ np.linalg.inv(Q_block) @ A
            D12 = -A.T @ np.linalg.inv(Q_block)

            f_prior = D12.T @ np.linalg.inv(f_post + D11) @ D12

            # Update the combined covariance matrix
            cov_p = A @ cov_m @ A.T + Q_block

            comb_y_m = np.array([])  # Combined measurement vector
            comb_h = np.array([])  # Combined estimation vector
            comb_H = np.array([[]])  # Combined Jacobian matrix

            R_vec = np.array([])  # Combined measurement noise vector

            for k, sat in enumerate(sats_copy):

                # Get visible landmarks
                visible_landmarks = sat.visible_landmarks_list(sat.x_p)
                sat.bearing_dim = len(visible_landmarks) * 3
                meas_dim = n_sats - 1 + sat.bearing_dim

                # Re-initialize the measurement matrices for each satellite with the correct dimensions
                # based on the number of visible landmarks
                y_m = np.zeros((meas_dim))
                h = np.zeros((meas_dim))
                R_vec = np.append(R_vec, [sat.R_weight_bearing] * sat.bearing_dim)

                # Calculate Jacobian matrix H for combined state (still just one satellite H)
                H = combined_state_H(sat, meas_dim, state_dim, i, k)

                # Calculate h
                h[0 : sat.bearing_dim] = sat.h_landmark(sat.x_p[0:3])
                for j in range(sat.bearing_dim, meas_dim):
                    h[j] = sat.h_inter_range(i + 1, j, sat.x_p[0:3])
                    R_vec = np.append(R_vec, sat.R_weight_range)

                y_m[0 : sat.bearing_dim] = sat.measure_z_landmark()
                y_m[sat.bearing_dim : meas_dim] = sat.measure_z_range(sats_copy)

                # Append vectors and matrices to combined form
                comb_y_m = np.append(comb_y_m, y_m, axis=0)
                comb_h = np.append(comb_h, h, axis=0)
                if k == 0:
                    comb_H = H
                else:
                    comb_H = np.append(comb_H, H, axis=0)

            # Create R based on the number of measurements of all satellites
            R = np.diag(R_vec)

            # Calculate K
            K = cov_p @ comb_H.T @ np.linalg.inv(comb_H @ cov_p @ comb_H.T + R)
            x_m = x_p + K @ (comb_y_m - comb_h)
            cov_m = (np.eye(state_dim * n_sats) - K @ comb_H) @ cov_p @ (
                (np.eye(state_dim * n_sats) - K @ comb_H).T
            ) + K @ R @ K.T

            # Set sat's x_m so that they can be used for the next prior update x_p state.
            # Individual covariances of sats don't matter for this because we use the full covariances
            for sat in sats_copy:
                sat.x_m = x_m[sat.id * state_dim : (sat.id + 1) * state_dim]

            # FIM Calculation
            f_post = (
                f_prior + comb_H.T @ np.linalg.inv(R) @ comb_H + np.linalg.inv(Q_block)
            )

            # Assign Posterior Covariance
            cov_hist[trial, i, :, :] = cov_m

            filter_position[trial, i, :] = (
                np.array([x_m[0::state_dim], x_m[1::state_dim], x_m[2::state_dim]])
                .transpose()
                .reshape(-1)
            )
            pos_error[trial, i, :] = filter_position[trial, i, :] - comb_curr_pos

            # # Sanity check that Cov - FIM is positive definite (Should always be true)
            fpost_sanity_check(f_post, sat, verbose, state_dim)

            fim[trial, i, :, :] = f_post

        sats_copy = copy.deepcopy(
            sats
        )  # Reset the satellites to initial condition for the next trial

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
    all_sat_position_error(pos_error, n_sats)

    # Plotting

    # Plot the trajectory of the satellite
    # plot_trajectory(x_traj, filter_position, N)

    # Plot the positional error
    # plot_position_error(pos_error)

    # Plot crb and cov trace
    # plot_covariance_crb_trace(crb_trace, cov_trace)

    # plt.show()
