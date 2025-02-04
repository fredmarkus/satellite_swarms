import argparse
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import yaml

from landmarks.landmark import landmark, latlon2ecef
from sat.core import satellite
from sat.dynamics import rk4_discretization, state_transition
from utils.plotting_utils import plot_covariance_crb, plot_trajectory, plot_position_error

if __name__ == "__main__":

    ### Landmark Initialization ###
    # Import csv data for the lanldmarks
    landmarks = []
    with open('landmarks/landmark_coordinates.csv', newline='',) as csvfile:
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
    dt = 1/args.f #Hz
    n_sats = args.n_sats
    R_weight = args.R_weight
    state_dim = args.state_dim
    verbose = args.verbose
    
    #MC Parameters
    num_trials = args.num_trials

    bearing_dim = len(landmark_objects)*3
    meas_dim = n_sats-1 + bearing_dim   

    # Process noise covariance matrix based on paper "Autonomous orbit determination and observability analysis for formation satellites" by OU Yangwei, ZHANG Hongbo, XING Jianjun
    # page 6
    Q = np.diag(np.array([10e-6,10e-6,10e-6,10e-12,10e-12,10e-12]))


    # Do not seed in order for Monte-Carlo simulations to actually produce different outputs!
    # np.random.seed(42)        #Set seed for reproducibility

    ### Satellite Initialization ###
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    sats = []

    for i, sat_config in enumerate(config["satellites"]):
    
        # Only create the number of satellites specified in the argument. The rest of the yaml file is ignored
        if i < n_sats:
        # Overwrite the following yaml file parameters with values provided in this script
            sat_config["N"] = N
            sat_config["landmarks"] = landmark_objects
            sat_config["n_sats"] = n_sats
            # sat_config["R_weight"] = R_weight
            sat_config["bearing_dim"] = bearing_dim
            sat_config["verbose"] = verbose

            satellite_inst = satellite(**sat_config)
            sats.append(satellite_inst)
            
        else:
            break


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

    cov_hist = np.zeros((N,n_sats,state_dim,state_dim))
    sats_copy = copy.deepcopy(sats)

    filter_position = np.zeros((num_trials, N, 3, n_sats))
    pos_error = np.zeros((num_trials, N , 3, n_sats))


    for trial in range(num_trials):
        print("Monte Carlo Trial: ", trial)

        f_prior = np.zeros((state_dim, state_dim, n_sats))
        f_post = np.zeros((state_dim, state_dim, n_sats))

        for i in range(N):
            print("Timestep: ", i)

            for sat in sats_copy:
                sat.curr_pos = x_traj[i+1,0:3,sat.id] #Provide the underlying groundtruth position to the satellite for bearing and ranging measurements
                A = state_transition(sat.x_m)
                sat.x_p = rk4_discretization(sat.x_m, dt)
                sat.cov_p = A@sat.cov_m@A.T + Q # Update the covariance matrix as a P_p = A*P*A^T + LQL^T

                # FIM Calculations
                D11 = A.T@np.linalg.inv(Q)@A
                D12 = -A.T@np.linalg.inv(Q)
                
                f_prior[:,:,sat.id] = D12.T@np.linalg.inv(f_post[:,:,sat.id] + D11)@D12

            for sat in sats_copy:

                #Get visible landmarks
                visible_landmarks = sat.visible_landmarks_list(sat.x_p)
                sat.bearing_dim = len(visible_landmarks)*3
                meas_dim = n_sats-1 + sat.bearing_dim

                # Re-initialize the measurement matrices for each satellite with the correct dimensions
                # based on the number of visible landmarks
                y_m = np.zeros((meas_dim))
                H = np.zeros((meas_dim,state_dim))
                h = np.zeros((meas_dim))

                # Re-initialize the measurement noise covariance matrix with correct dimensions
                R = np.eye(meas_dim)*R_weight

                #Calculate H
                H[0:sat.bearing_dim,0:state_dim] = sat.H_landmark(sat.x_p)
                for j in range(sat.bearing_dim,meas_dim):
                    H[j,:] = sat.H_inter_range(i+1, j, sat.x_p)

                # FIM Calculation
                f_post[:,:,sat.id] = f_prior[:,:,sat.id] + H.T@np.linalg.inv(R)@H + np.linalg.inv(Q)

                #Calculate K
                K = sat.cov_p@H.T@np.linalg.inv(H@sat.cov_p@H.T + R)

                #Calculate h
                h[0:sat.bearing_dim] = sat.h_landmark(sat.x_p[0:3])
                for j in range(sat.bearing_dim,meas_dim):
                    h[j] = sat.h_inter_range(i+1, j, sat.x_p[0:3])

                y_m[0:sat.bearing_dim] = sat.measure_z_landmark()
                y_m[sat.bearing_dim:meas_dim] = sat.measure_z_range(sats_copy)

                sat.x_m = sat.x_p + K@(y_m - h)
                sat.cov_m = (np.eye(state_dim) - K@H)@sat.cov_p@((np.eye(state_dim) - K@H).T) + K@R@K.T

                cov_hist[i,sat.id,:,:] += sat.cov_m

                filter_position[trial,i,:,sat.id] = sat.x_m[0:3]
                pos_error[trial, i,:,sat.id] = filter_position[trial,i,:,sat.id] - sat.curr_pos[0:3]

                # Sanity check that Cov - FIM is positive definite (Should always be true)
                if np.linalg.matrix_rank(f_post[:,:,sat.id]) == state_dim:
                    if not np.all(np.linalg.eigvals(sat.cov_m - np.linalg.inv(f_post[:,:,sat.id])) > 0):
                        print(f"Satellite {sat.id} FIM is NOT positive definite. Something went wrong!!!")

                # Check if f_post is invertible
                if np.linalg.matrix_rank(f_post[:,:,sat.id]) != state_dim:
                    if verbose:
                        print(f_post[:,:,sat.id])
                        print(f"Satellite {sat.id} FIM is not invertible")

                else:
                    eig_val, eig_vec = np.linalg.eig(f_post[:,:,sat.id])
                    if verbose:
                        print(f"Eigenvalues of satellite {sat.id} FIM: ", eig_val)
                    # print("Condition number of FIM: ", np.linalg.cond(f_post))
                    # print("Eigenvectors of FIM: ", eig_vec)
                
            #Assume no knowledge at initial state so we don't place any information in the first state_dim x state_dim block
            if i > 0:
                start_i = i * state_dim
                fim[trial, start_i:start_i+state_dim,start_i:start_i+state_dim] = f_post[:,:,0]
        
        sats_copy = copy.deepcopy(sats) # Reset the satellites for the next trial

    # Average FIM
    fim = np.mean(fim, axis=0)

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
    crb = np.linalg.pinv(fim)
    crb_diag = np.diag(crb)

    # Plot the covariance matrix and the FIM diagonal entries.
    plot_covariance_crb(crb_diag, state_dim, sat1_cov_hist)

    # Plot the trajectory of the satellite
    plot_trajectory(x_traj, filter_position, N)

    # Plot the positional error
    plot_position_error(pos_error)

    plt.show()
