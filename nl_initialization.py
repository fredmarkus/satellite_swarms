"""
Simulation setup for satellite formation flying using recursive filter.
"""

import argparse
from collections import deque
import cyipopt
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
import yaml

from analysis import fpost_sanity_check
from analysis import get_cov_trace
from analysis import get_crb_trace
from sat.dynamics import exchange_trajectories
from sat.dynamics import f
from sat.dynamics import f_jac
from sat.dynamics import simulate_nominal_trajectories
from traj_solver import trajSolver_v2
from utils.config_utils import load_sat_config
from utils.data_io_utils import import_landmarks
from utils.data_io_utils import store_all_data
from utils.data_io_utils import setup_data_dir
from utils.math_utils import M_Jac
from utils.plotting_utils import all_sat_position_error
from utils.plotting_utils import plot_all_sat_crb_trace
from utils.plotting_utils import plot_trajectory
from utils.yaml_autogen_utils import generate_satellites_yaml


# Function to solve the non-linear least squares problem
def solve_nls(x_traj, nlp, N,state_dim):
    # Randomize initia; guess
    # TODO: Fix this to make the initial guess noise a function of the error between ground truth and last guess
    glob_x0 = x_traj[:-1,:,0] + np.random.normal(loc=0,scale=1,size=(N,state_dim))
    glob_x0 = glob_x0.flatten()

    nlp.add_option('max_iter', 400)
    nlp.add_option('tol', 1e-4)
    nlp.add_option('print_level', 3)
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('hessian_approximation', 'limited-memory') # 'exact' or 'limited-memory'
    nlp.add_option('linear_solver', 'ma57')
    # nlp.add_option('limited_memory_max_history', 10)
    # nlp.add_option('limited_memory_max_skipping', 2)
    # nlp.add_option('bound_push',1e-6)
    # nlp.add_option('output_file', 'output.log')


    x, _ = nlp.solve(glob_x0)

    return x

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
    # Individual covariance matrix for each satellite
    # ind_cov = np.diag(
    #     np.array([1, 1, 1, 1, 1, 1])
    # )  
    nls_estimates = []

    ### Satellite Initialization ###
    sats = load_sat_config(args=args)

    x_traj = simulate_nominal_trajectories(N, dt, sats, state_dim)
    np.save(f"data/nominal_trajectory.npy", x_traj)

    # Transfer the positions of the other satellites for each satellite at N+1 timesteps
    sats = exchange_trajectories(sats, x_traj)

    ## Calculate FIM in recursive fashion.
    # fim = np.zeros((num_trials, N, state_dim * n_sats, state_dim * n_sats))
    # cov_hist = np.zeros((num_trials, N, state_dim * n_sats, state_dim * n_sats))


        #Initialize the NLP
    solver = trajSolver_v2(
        x_traj=x_traj,
        N=N,
        n_sats=n_sats,
        state_dim=state_dim,
        dt=dt,
        is_initial=False,
        sats_collection=sats, 
        meas_type=meas_type,
    )

    # TODO: Parameterize m and cl,cu as parameter of is_initial (size changes depending)
    # FOR NOW USE SOLVING FOR INITIAL CONDITIONS INCLUSIVE
    nlp = cyipopt.Problem(
        n = N * state_dim,
        m = N * state_dim,
        problem_obj=solver,
        lb = jnp.full(N * state_dim, -jnp.inf),
        ub = jnp.full(N * state_dim, jnp.inf),
        cl = [0] * (N * state_dim),
        cu = [0] * (N * state_dim),
    )

    x = solve_nls(x_traj, nlp, N, state_dim)
    nls_estimates.append(x)

    error = x - x_traj[:-1,:,0].reshape(-1)

    print(error)
    np.save(f"data/states_least_squares_error.npy", error)

    # # Set sat's x_m so that they can be used for the next prior update x_p state.
    # # Individual covariances of sats don't matter for this because we use the full covariances
    # for sat in sats_copy:
    #     sat.x_m = total_x_m[sat.id * state_dim : (sat.id + 1) * state_dim]
    #     # sat.cov_m = cov_m[sat.id * state_dim : (sat.id + 1) * state_dim, sat.id * state_dim : (sat.id + 1) * state_dim]

    # # FIM Calculation
    # print(np.linalg.cond(f_post))
    # # Assign Posterior Covariance
    # cov_hist[trial, i, :, :] = cov_m

    # filter_position[trial, i, :] = (
    #     np.array([total_x_m[0::state_dim], total_x_m[1::state_dim], total_x_m[2::state_dim]])
    #     .transpose()
    #     .reshape(-1)
    # )
    # pos_error[trial, i, :] = filter_position[trial, i, :] - comb_curr_pos

    # # # Sanity check that Cov - FIM is positive definite (Should always be true)
    # fpost_sanity_check(f_post, cov_m, args.verbose, state_dim)

    # fim[trial, i, :, :] = f_post

    # # Reset the satellites to initial condition for the next trial
    # sats_copy = copy.deepcopy(
    #     sats
    # )  

    # Average history of relevant variables
    # fim = np.mean(fim, axis=0)
    # cov_hist = np.mean(cov_hist, axis=0)
    # pos_error = np.mean(pos_error, axis=0)

    # # Calculate Covariance and CRB trace
    # cov_trace = get_cov_trace(N, cov_hist, n_sats)
    # crb_trace = get_crb_trace(N, fim, n_sats)


    # Save the results to files
    # store_all_data(
    #     n_sats=n_sats,
    #     cov_trace=cov_trace,
    #     crb_trace=crb_trace,
    #     pos_error=pos_error,
    #     sats=sats,
    # )

    # Plotting of errors
    # all_sat_position_error(pos_error, n_sats, meas_type, cov_hist)

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
