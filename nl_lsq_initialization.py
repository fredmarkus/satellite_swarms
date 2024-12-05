# This file will consider the initial state estimation problem when no accurate initial estimates are available. 
import argparse
import copy
import csv
import cyipopt
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
import yaml

from landmark import landmark, latlon2ecef
from satellite import satellite
from sat_dynamics import rk4_discretization
from traj_solver import trajSolver
from utils.plotting_utils import plot_covariance_crb, plot_trajectory, plot_position_error

# Constants
MU = 3.986004418 * 10**5 # km^3/s^2 # Gravitational parameter of the Earth

# Function to solve the non-linear least squares problem
def solve_nls(x_traj, nlp, sat_id):
    # Randomize initia; guess
    # TODO: FIx this to make the initial guess noise a function of the error between ground truth and last guess
    glob_x0 = x_traj[:,:,sat_id] + np.random.normal(loc=0,scale=100_000,size=(N,state_dim))
    glob_x0 = glob_x0.flatten()

    nlp.add_option('max_iter', 100)
    nlp.add_option('tol', 1e-6)
    nlp.add_option('print_level', 5)
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('hessian_approximation', 'limited-memory') # 'exact' or 'limited-memory'
    nlp.add_option('linear_solver', 'mumps')
    # nlp.add_option('limited_memory_max_history', 10)
    # nlp.add_option('limited_memory_max_skipping', 2)
    # nlp.add_option('bound_push',1e-6)
    # nlp.add_option('output_file', 'output.log')


    x, _ = nlp.solve(glob_x0)

    return x


if __name__ == "__main__":

    ### Landmark Initialization ###
    # Import csv data for the lanldmarks
    landmarks = []
    with open('landmark_coordinates.csv', newline='',) as csvfile:
        reader = csv.reader(csvfile, delimiter=',',)
        for row in reader:
            landmarks.append(np.array([row[0], row[1], row[2], row[3]]))


    landmarks_ecef = latlon2ecef(landmarks)
    landmark_objects = []

    # Initialize the landmark objects with their correct name and the ECEF coordinates
    for landmark_obj in landmarks_ecef:
        landmark_objects.append(landmark(x=float(landmark_obj[1]), y=float(landmark_obj[2]), z=float(landmark_obj[3]), name=(landmark_obj[0])))


    #General Parameters
    parser = argparse.ArgumentParser(description='Nonlinear Recursive Monte Carlo Simulation')
    parser.add_argument('--N', type=int, default=5, help='Number of timesteps')
    parser.add_argument('--f', type=float, default=0.1, help='Frequency of the simulation')
    parser.add_argument('--n_sats', type=int, default=1, help='Number of satellites')
    parser.add_argument('--R_weight', type=float, default=1000, help='Measurement noise weight')
    parser.add_argument('--state_dim', type=int, default=6, help='Dimension of the state vector')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of Monte Carlo trials')
    args = parser.parse_args()

    N = args.N
    f = args.f #Hz
    n_sats = args.n_sats
    R_weight = args.R_weight
    state_dim = args.state_dim

    bearing_dim = len(landmark_objects)*3
    dt = 1/f
    meas_dim = n_sats-1 + bearing_dim   
    R = np.eye(meas_dim)*R_weight
    # Process noise covariance matrix based on paper "Autonomous orbit determination and observability analysis for formation satellites" by OU Yangwei, ZHANG Hongbo, XING Jianjun
    # page 6
    Q = np.diag(np.array([10e-6,10e-6,10e-6,10e-12,10e-12,10e-12]))

    #MC Parameters
    num_trials = args.num_trials
    nls_estimates = []


    # Do not seed in order for Monte-Carlo simulations to actually produce different outputs!
    # np.random.seed(42)        #Set seed for reproducibility

    ### Satellite Initialization ###
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    sats = []

    for sat_config in config["satellites"]:

        # Overwrite the following yaml file parameters with values provided in this script
        sat_config["N"] = N
        sat_config["landmarks"] = landmark_objects
        sat_config["meas_dim"] = meas_dim
        sat_config["n_sats"] = n_sats
        sat_config["R_weight"] = R_weight
        sat_config["bearing_dim"] = bearing_dim

        satellite_inst = satellite(**sat_config)
        sats.append(satellite_inst)

    ## Calculate FIM directly in recursive fashion.
    fim = np.zeros((num_trials,N*state_dim, N*state_dim))

    J = np.zeros((meas_dim, state_dim))
    R_inv = np.linalg.inv(R)

    cov_hist = np.zeros((N,n_sats,state_dim,state_dim))
    sats_copy = copy.deepcopy(sats)

    filter_position = np.zeros((num_trials, N, 3, n_sats))
    pos_error = np.zeros((num_trials, N , 3, n_sats))

    non_zero_meas = False

    while True:

        y_m = np.zeros((N,meas_dim,n_sats))
        # f_prior = np.zeros((state_dim, state_dim))
        # f_post = np.zeros((state_dim, state_dim))

        x_traj = np.zeros((N, state_dim, n_sats))
        for sat in sats: 
            x = sat.x_0
            for i in range(N):
                x_traj[i,:,sat.id] = x
                x = rk4_discretization(x, dt)

        for sat in sats:
            sat_i = 0 #iterator variable
            for other_sat in sats:
                if sat.id != other_sat.id:
                    sat.other_sats_pos[:,:,sat_i] = x_traj[:,0:3,other_sat.id] # Transfer all N+1 3D positions of the other satellites from x_traj
                    sat_i += 1

            sat.x_0 = x_traj[0,:,sat.id]
 

        for i in range(N-1):
            print("Timestep: ", i)

            for sat in sats_copy:
                sat.curr_pos = x_traj[i+1,0:3,sat.id] #Provide the underlying groundtruth position to the satellite for bearing and ranging measurements

            for sat in sats_copy:
                y_m[i,0:bearing_dim,sat.id] = sat.measure_z_landmark()
                y_m[i,bearing_dim:meas_dim,sat.id] = sat.measure_z_range(sats_copy)
                if y_m[i,:,sat.id].any() != 0:
                    # print(f"Satellite {sat.id} at time {i} has measurements {y_m[i,:,sat.id]}")
                    non_zero_meas = True
        
        if non_zero_meas:

        ## Initialize the NLP
            solver = trajSolver(
                x_traj=x_traj,
                y_m=y_m,
                sat=sats_copy[0],
                N=N,
                meas_dim=meas_dim, 
                bearing_dim=bearing_dim, 
                n_sats=n_sats,
                MU=MU, 
                state_dim=state_dim,
                dt=dt)
            
            nlp = cyipopt.Problem(
                n = N*state_dim,
                m = N*state_dim,
                problem_obj=solver,
                lb = jnp.full(N*state_dim, -jnp.inf),
                ub = jnp.full(N*state_dim, jnp.inf),
                cl = jnp.zeros(N*state_dim),
                cu = jnp.zeros(N*state_dim)
            )

            x = solve_nls(x_traj, nlp, 0)
            nls_estimates.append(x)

            error = np.linalg.norm(x - x_traj[:,:,0].flatten())
            print(f"Error: {error}")
            
            non_zero_meas = False

            # for sat in sats_copy:
            #     #Calculate H
            #     H[0:bearing_dim,0:state_dim] = sat.H_landmark(sat.x_p)
            #     for j in range(bearing_dim,meas_dim):
            #         H[j,:] = sat.H_inter_range(i+1, j, sat.x_p)
            #     #Calculate K
            #     K = sat.cov_p@H.T@np.linalg.pinv(H@sat.cov_p@H.T + R)

            #     #Calculate h
            #     h[0:bearing_dim] = sat.h_landmark(sat.x_p[0:3])
            #     for j in range(bearing_dim,meas_dim):
            #         h[j] = sat.h_inter_range(i+1, j, sat.x_p[0:3])

            #     y_m[i,0:bearing_dim,sat.id] = sat.measure_z_landmark()
            #     y_m[i,bearing_dim:meas_dim,sat.id] = sat.measure_z_range(sats_copy)

            #     sat.x_m = sat.x_p + K@(y_m[i,:,sat.id] - h)
            #     sat.cov_m = (np.eye(state_dim) - K@H)@sat.cov_p@((np.eye(state_dim) - K@H).T) + K@R@K.T

            #     cov_hist[i,sat.id,:,:] += sat.cov_m

            #     filter_position[trial,i,:,sat.id] = sat.x_m[0:3]
            #     pos_error[trial, i,:,sat.id] = filter_position[trial,i,:,sat.id] - sat.curr_pos[0:3]


            # Calculate the FIM for satellite 0
            # A = state_transition(sats_copy[0].x_m)
            # f_prior = A@f_post@A.T + np.linalg.inv(Q) # with process noise
            # J[0:bearing_dim,0:state_dim] = sats_copy[0].H_landmark(sats_copy[0].x_m)
            # for j in range(bearing_dim,meas_dim): ## Consider checks for nan values
            #     J[j,0:state_dim] = sats_copy[0].H_inter_range(i+1, j, sats_copy[0].x_m)
            # f_post = f_prior + J.T@R_inv@J

            # # Check if f_post is invertible
            # if np.linalg.matrix_rank(f_post) != state_dim:
            #     print(f_post)
            #     print("FIM is not invertible")

            # else:
            #     eig_val, eig_vec = np.linalg.eig(f_post)
            #     # print("Eigenvalues of FIM: ", eig_val)
            #     # print("Condition number of FIM: ", np.linalg.cond(f_post))
            #     # print("Eigenvectors of FIM: ", eig_vec)
                
            # #Assume no knowledge at initial state so we don't place any information in the first state_dim x state_dim block
            # if i > 0:
            #     start_i = i * state_dim
            #     fim[trial, start_i:start_i+state_dim,start_i:start_i+state_dim] = f_post

                # print(f"Satellite {sat.id} at time {i} has covariance {sat.cov_m}")
        
        # sats_copy = copy.deepcopy(sats) # Reset the satellites for the next trial

    # Average FIM
    # fim = np.mean(fim, axis=0)

    #Get the average covariance matrix for each satellite for each timestep
    for i in range(N):
        for sat in sats:
            cov_hist[i,sat.id,:,:] = cov_hist[i,sat.id,:,:]/num_trials

    sat1_cov_hist = cov_hist[:,0,:,:]

    # The FIM should be the inverse of the Cramer-Rao Bound. Isolating first step as it is full of 0 values and nor invertible.
    # fim_acc = fim[state_dim:,state_dim:]
    # crb = np.linalg.inv(fim_acc)
    # crb_final = np.zeros((N*state_dim,N*state_dim))
    # crb_final[state_dim:,state_dim:] = crb

    # Using pseudo-inverse to invert the matrix
    # crb = np.linalg.pinv(fim)
    # crb_diag = np.diag(crb)

    # Plot the covariance matrix and the FIM diagonal entries.
    # plot_covariance_crb(crb_diag, state_dim, sat1_cov_hist)

    # Plot the trajectory of the satellite
    plot_trajectory(x_traj, filter_position, N)

    # Plot the positional error
    plot_position_error(pos_error)

    plt.show()
