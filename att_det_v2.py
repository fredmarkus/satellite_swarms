import argparse
import copy
import csv
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
import yaml

from landmark import landmark, latlon2ecef
from satellite import satellite
from sat_dynamics import rk4_discretization, state_transition, attitude_state_propagation
from utils.math_utils import hat, hat_stack, quaternion_to_rotation_matrix, quaternion_to_left_matrix
from utils.plotting_utils import plot_covariance_crb, plot_trajectory, plot_position_error

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
    parser.add_argument('--N', type=int, default=100, help='Number of timesteps')
    parser.add_argument('--f', type=float, default=1, help='Frequency of the simulation')
    parser.add_argument('--n_sats', type=int, default=1, help='Number of satellites')
    parser.add_argument('--R_weight', type=float, default=10e-5, help='Measurement noise weight')
    parser.add_argument('--state_dim', type=int, default=6, help='Dimension of the state vector')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of Monte Carlo trials')
    parser.add_argument('--verbose', type=bool, default=False, help='Print information')
    args = parser.parse_args()

    N = args.N
    f = args.f #Hz
    n_sats = args.n_sats
    R_weight = args.R_weight
    state_dim = args.state_dim
    verbose = args.verbose

    bearing_dim = len(landmark_objects)*3
    dt = 1/f
    meas_dim = n_sats-1 + bearing_dim   
    R = np.eye(meas_dim)*R_weight
    R_att = np.eye(bearing_dim)*R_weight
    # Process noise covariance matrix based on paper "Autonomous orbit determination and observability analysis for formation satellites" by OU Yangwei, ZHANG Hongbo, XING Jianjun
    # page 6
    Q = np.diag(np.array([10e-6,10e-6,10e-6,10e-12,10e-12,10e-12]))
    Q_att = np.diag(np.array([10e-6,10e-6,10e-6,10e-12,10e-12,10e-12]))

    #MC Parameters
    num_trials = args.num_trials

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
        sat_config["verbose"] = verbose

        satellite_inst = satellite(**sat_config)
        sats.append(satellite_inst)


    # Generate synthetic for N+1 timesteps so that we can calculate the FIM for N timesteps;
    x_traj = np.zeros((N+1, state_dim, n_sats)) # Discretized trajectory of satellite states over time period
    # This loop is deterministic as we are always doing the same discretization so we do not need to regenerate the trajectories.
    for sat in sats: 
        x = sat.x_0
        for i in range(N+1):
            x_traj[i,:,sat.id] = x
            x = rk4_discretization(x, dt)

    # Get the positions of the other satellites for each satellite at all N+1 timesteps
    for sat in sats:
            sat_i = 0 #iterator variable
            for other_sat in sats:
                if sat.id != other_sat.id:
                    sat.other_sats_pos[:,:,sat_i] = x_traj[:,0:3,other_sat.id] # Transfer all N+1 3D positions of the other satellites from x_traj
                    sat_i += 1

    # NOTE: We are only doing these trials for one satellites (for now) 

    ## Calculate FIM directly in recursive fashion.
    fim = np.zeros((num_trials,N*state_dim, N*state_dim))

    J = np.zeros((meas_dim, state_dim))
    R_inv = np.linalg.inv(R)

    cov_hist = np.zeros((N,n_sats,state_dim,state_dim))
    sats_copy = copy.deepcopy(sats)

    filter_position = np.zeros((num_trials, N, 3, n_sats))
    pos_error = np.zeros((num_trials, N , 3, n_sats))


    for trial in range(num_trials):
        print("Monte Carlo Trial: ", trial)

        # y_m = np.zeros((N,meas_dim,n_sats))
        # H = np.zeros((meas_dim,state_dim))
        # h = np.zeros((meas_dim))
        # f_prior = np.zeros((state_dim, state_dim, n_sats))
        # f_post = np.zeros((state_dim, state_dim, n_sats))

        for i in range(N):
            print("Timestep: ", i)

            for sat in sats_copy:
                sat.q_p = attitude_state_propagation(sat.q_m, sat.beta_m, sat.w, dt)
                v = -(sat.w - sat.beta_m)
                mag = np.linalg.norm(v)
                v_hat = hat(v/mag) if mag != 0 else np.zeros(3)
                R1 = np.eye(3) + np.sin(mag*dt)*v_hat + (1 - np.cos(mag*dt))*v_hat@v_hat
                A = np.block([
                    [R1, -dt*np.eye(3)],
                    [np.zeros((3,3)), np.eye(3)]])
                
                sat.att_cov_p = A@sat.att_cov_m@A.T + Q_att

            for sat in sats_copy:
                # TODO: Fix this using correct quaternion initialization and propagation
                br_landmark = sat.measure_z_landmark()
                # inertial2body = quaternion_to_rotation_matrix(sat.q_p).T
                # inertial_measurement = -sat.x_p[0:3]/ np.linalg.norm(sat.x_p[0:3])
                Z = br_landmark# - inertial2body@inertial_measurement 
                C = np.block([[hat_stack(br_landmark), np.zeros((bearing_dim, 3))]])
                S = C@sat.att_cov_p@C.T + R_att

                #Kalman Gain
                L = sat.att_cov_p@C.T@np.linalg.inv(S)

                #Update
                dx = L@Z
                psi = dx[0:3]
                d_beta = dx[3:]
                theta = np.linalg.norm(psi)
                r = psi/theta if theta != 0 else np.zeros(3)
                
                # Posterior updates for quaternion, bias and covariance matrix
                sat.q_m = quaternion_to_left_matrix(sat.q_p)@np.concatenate(([np.cos(theta/2)], np.sin(theta/2)*r))
                # Normalize the quaternion  
                sat.q_m = sat.q_m/np.linalg.norm(sat.q_m)
                sat.beta_m = sat.beta_m + d_beta
                sat.att_cov_m = (np.eye(6) - L@C)@sat.att_cov_p@(np.eye(6) - L@C).T + L@R_att@L.T


                #Calculate H
                # H[0:bearing_dim,0:state_dim] = sat.H_landmark(sat.x_p)

                # Range measurements do not aid the attitude determination problem and should thus be ignored.

                # FIM Calculation
                #f_post[:,:,sat.id] = f_prior[:,:,sat.id] + H.T@R_inv@H + np.linalg.inv(Q)

                #Calculate K
                # K = sat.cov_p@H.T@np.linalg.pinv(H@sat.cov_p@H.T + R)

                #Calculate h
                # h[0:bearing_dim] = sat.h_landmark(sat.x_p[0:3])
                # for j in range(bearing_dim,meas_dim):
                #     h[j] = sat.h_inter_range(i+1, j, sat.x_p[0:3])

                # y_m[i,0:bearing_dim,sat.id] = sat.measure_z_landmark()
                # y_m[i,bearing_dim:meas_dim,sat.id] = sat.measure_z_range(sats_copy)

                # sat.x_m = sat.x_p + K@(y_m[i,:,sat.id] - h)
                # sat.cov_m = (np.eye(state_dim) - K@H)@sat.cov_p@((np.eye(state_dim) - K@H).T) + K@R@K.T

                # cov_hist[i,sat.id,:,:] += sat.cov_m

                # filter_position[trial,i,:,sat.id] = sat.x_m[0:3]
                # pos_error[trial, i,:,sat.id] = filter_position[trial,i,:,sat.id] - sat.curr_pos[0:3]

                # # Sanity check that Cov - FIM is positive definite (Should always be true)
                # if np.linalg.matrix_rank(f_post[:,:,sat.id]) == state_dim:
                #     if not np.all(np.linalg.eigvals(sat.cov_m - np.linalg.inv(f_post[:,:,sat.id])) > 0):
                #         print(f"Satellite {sat.id} FIM is NOT positive definite. Something is went wrong!!!")

                # # Check if f_post is invertible
                # if np.linalg.matrix_rank(f_post[:,:,sat.id]) != state_dim:
                #     print(f_post[:,:,sat.id])
                #     print(f"Satellite {sat.id} FIM is not invertible")

                # else:
                #     eig_val, eig_vec = np.linalg.eig(f_post[:,:,sat.id])
                #     print(f"Eigenvalues of satellite {sat.id} FIM: ", eig_val)
                #     # print("Condition number of FIM: ", np.linalg.cond(f_post))
                #     # print("Eigenvectors of FIM: ", eig_vec)
                
            #Assume no knowledge at initial state so we don't place any information in the first state_dim x state_dim block
            # if i > 0:
            #     start_i = i * state_dim
            #     fim[trial, start_i:start_i+state_dim,start_i:start_i+state_dim] = f_post[:,:,0]

            #     # print(f"Satellite {sat.id} at time {i} has covariance {sat.cov_m}")
        
        sats_copy = copy.deepcopy(sats) # Reset the satellites for the next trial

    # Average FIM
    # fim = np.mean(fim, axis=0)

    #Get the average covariance matrix for each satellite for each timestep
    # for i in range(N):
    #     for sat in sats:
    #         cov_hist[i,sat.id,:,:] = cov_hist[i,sat.id,:,:]/num_trials

    # sat1_cov_hist = cov_hist[:,0,:,:]

    # The FIM should be the inverse of the Cramer-Rao Bound. Isolating first step as it is full of 0 values and nor invertible.
    # fim_acc = fim[state_dim:,state_dim:]
    # crb = np.linalg.inv(fim_acc)
    # crb_final = np.zeros((N*state_dim,N*state_dim))
    # crb_final[state_dim:,state_dim:] = crb

    # Using pseudo-inverse to invert the matrix
    # crb = np.linalg.pinv(fim)
    # crb_diag = np.diag(crb)

    # # Plot the covariance matrix and the FIM diagonal entries.
    # plot_covariance_crb(crb_diag, state_dim, sat1_cov_hist)

    # # Plot the trajectory of the satellite
    # plot_trajectory(x_traj, filter_position, N)

    # # Plot the positional error
    # plot_position_error(pos_error)

    # plt.show()