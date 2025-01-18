import argparse
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import block_diag
from tqdm import tqdm
import yaml

from landmark import landmark, latlon2ecef
from sat.sat_core import satellite
from sat.sat_dynamics import rk4_discretization, state_transition
from utils.plotting_utils import plot_covariance_crb, plot_trajectory, plot_position_error, plot_covariance_crb_trace, plot_all_sat_crb_trace
from utils.yaml_autogen_utils import generate_satellites_yaml


def run_simulation(args):

    N = args.N
    dt = 1/args.f #Hz
    n_sats = args.n_sats
    R_weight = args.R_weight
    state_dim = args.state_dim
    
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
    if args.random_yaml:
        with open("config/sat_autogen.yaml", "r") as file:
            config = yaml.safe_load(file)

    else:
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

            if len(config["satellites"]) < n_sats:
                raise ValueError("""Number of satellites specified is greater than the number of satellites in the yaml file. 
                                 Add --random_yaml flag to generate random satellite configuration for the provided number of 
                                 satellites or create custom satellites in config/config.yaml""")


    sats = []

    for i, sat_config in enumerate(config["satellites"]):
        
        # Only create the number of satellites specified in the argument. The rest of the yaml file is ignored
        if i < n_sats:
        # Overwrite the following yaml file parameters with values provided in this script
            sat_config["N"] = N
            sat_config["landmarks"] = landmark_objects
            sat_config["n_sats"] = n_sats
            sat_config["R_weight"] = R_weight
            sat_config["bearing_dim"] = bearing_dim
            sat_config["verbose"] = args.verbose
            sat_config["ignore_earth"] = args.ignore_earth

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


    ## Calculate FIM directly in recursive fashion.
    fim = np.zeros((num_trials, N, state_dim*n_sats, state_dim*n_sats))
    cov_hist = np.zeros((num_trials, N, state_dim*n_sats, state_dim*n_sats))

    sats_copy = copy.deepcopy(sats)

    filter_position = np.zeros((num_trials, N, 3*n_sats))
    pos_error = np.zeros((num_trials, N , 3*n_sats))

    Q = np.diag(np.array([10e-6,10e-6,10e-6,10e-12,10e-12,10e-12]))
    Q_block = block_diag(*[Q for _ in range(n_sats)])
    ind_cov = np.diag(np.array([1,1,1,0.1,0.1,0.1])) # Individual covariance matrix for each satellite


    for trial in tqdm(range(num_trials), desc=f"Monte Carlo for {n_sats} sat"):

        f_prior = np.zeros((state_dim*n_sats, state_dim*n_sats))
        f_post = np.zeros((state_dim*n_sats, state_dim*n_sats))

        # Initialize the combined state vector and covariance matrices
        # TODO: Complexify this function to handle different variance weights for different satellites
        cov_m = block_diag(*[ind_cov for _ in range(n_sats)])
        cov_p = block_diag(*[ind_cov for _ in range(n_sats)])
        x_m = np.zeros((state_dim*n_sats))
        x_p = np.zeros((state_dim*n_sats))

        comb_curr_pos = np.zeros((3*n_sats))

        A = np.zeros((n_sats*state_dim, n_sats*state_dim))

        #Initialize the measurement states using satellites initial measurement state
        for i in range(n_sats):
            x_m[i*state_dim:(i+1)*state_dim] = sats[i].x_m

        # Looping for timesteps
        for i in tqdm(range(N), desc="Timesteps"):

            for k, sat in enumerate(sats_copy):
                sat.curr_pos = x_traj[i+1,0:3,sat.id] #Provide the underlying groundtruth position to the satellite for bearing and ranging measurements
                sat.x_p = rk4_discretization(sat.x_m, dt)
                
                #Assign the state transition matrix to the correct block in the A matrix
                A[k*state_dim:(k+1)*state_dim,k*state_dim:(k+1)*state_dim] = state_transition(sat.x_m)

                #Update the combined state vector and underlying groundtruth
                x_p[k*state_dim:(k+1)*state_dim] = sat.x_p
                comb_curr_pos[k*3:(k+1)*3] = sat.curr_pos


            # FIM Calculations
            D11 = A.T@np.linalg.inv(Q_block)@A
            D12 = -A.T@np.linalg.inv(Q_block)
            
            f_prior = D12.T@np.linalg.inv(f_post + D11)@D12

            #Update the combined covariance matrix 
            cov_p = A@cov_m@A.T + Q_block

            comb_y_m = np.array([])
            comb_h = np.array([])
            comb_H = np.array([[]])

            for k, sat in enumerate(sats_copy):

                #Get visible landmarks
                visible_landmarks = sat.visible_landmarks_list(sat.x_p)
                sat.bearing_dim = len(visible_landmarks)*3
                meas_dim = n_sats-1 + sat.bearing_dim

                # Re-initialize the measurement matrices for each satellite with the correct dimensions
                # based on the number of visible landmarks
                y_m = np.zeros((meas_dim))
                H = np.zeros((meas_dim,state_dim*n_sats))
                h = np.zeros((meas_dim))

                #Calculate H
                H[0:sat.bearing_dim,k*state_dim:(k+1)*state_dim] = sat.H_landmark(sat.x_p)

                other_sat_id = 0
                for j in range(sat.bearing_dim,meas_dim):
                    # other_sat_id = j-sat.bearing_dim + 1
                    range_dist = sat.H_inter_range(i+1, j, sat.x_p)
                    H[j,sat.id*state_dim:(sat.id+1)*state_dim] = range_dist
                    if other_sat_id == sat.id:
                        other_sat_id += 1
                        H[j,other_sat_id*state_dim:(other_sat_id+1)*state_dim] = -range_dist
                    else:
                        H[j,other_sat_id*state_dim:(other_sat_id+1)*state_dim] = -range_dist
                        other_sat_id += 1


                #Calculate h
                h[0:sat.bearing_dim] = sat.h_landmark(sat.x_p[0:3])
                for j in range(sat.bearing_dim,meas_dim):
                    h[j] = sat.h_inter_range(i+1, j, sat.x_p[0:3])

                y_m[0:sat.bearing_dim] = sat.measure_z_landmark()
                y_m[sat.bearing_dim:meas_dim] = sat.measure_z_range(sats_copy)

                #Append vectors and matrices to combined form
                comb_y_m = np.append(comb_y_m, y_m, axis=0)
                comb_h = np.append(comb_h, h, axis=0)
                if k == 0: 
                    comb_H = H
                else:
                    comb_H = np.append(comb_H, H, axis=0)
            
            #Create R based on the number of measurements of all satellites
            R = np.eye(comb_y_m.shape[0])*R_weight

            #Calculate K
            K = cov_p@comb_H.T@np.linalg.inv(comb_H@cov_p@comb_H.T + R)
            x_m = x_p + K@(comb_y_m - comb_h)
            cov_m = (np.eye(state_dim*n_sats) - K@comb_H)@cov_p@((np.eye(state_dim*n_sats) - K@comb_H).T) + K@R@K.T

            # Set sat's x_m so that they can be used for the next prior update x_p state.
            # Individual covariances of sats don't matter for this because we use the full covariances
            for sat in sats_copy:
                sat.x_m = x_m[sat.id*state_dim:(sat.id+1)*state_dim]
            
            # FIM Calculation
            f_post = f_prior + comb_H.T@np.linalg.inv(R)@comb_H + np.linalg.inv(Q_block)

            # Assign Posterior Covariance
            cov_hist[trial,i,:,:] = cov_m

            filter_position[trial,i,:] = np.array([x_m[0::6], x_m[1::6], x_m[2::6]]).transpose().reshape(-1)
            pos_error[trial,i,:] = filter_position[trial,i,:] - comb_curr_pos

            # Sanity check that Cov - FIM is positive definite (Should always be true)
            if np.linalg.matrix_rank(f_post) == state_dim:
                if not np.all(np.linalg.eigvals(sat.cov_m - np.linalg.inv(f_post)) > 0):
                    print(f"Satellite {sat.id} FIM is NOT positive definite. Something went wrong!!!")

            # # Check if f_post is invertible
            if np.linalg.matrix_rank(f_post) != state_dim:
                if args.verbose:
                    print(f_post)
                    print(f"Satellite {sat.id} FIM is not invertible")

            else:
                eig_val, eig_vec = np.linalg.eig(f_post)
                if args.verbose:
                    print(f"Eigenvalues of satellite {sat.id} FIM: ", eig_val)
                    print("Condition number of FIM: ", np.linalg.cond(f_post))
                    print("Eigenvectors of FIM: ", eig_vec)
                
            fim[trial,i,:,:] = f_post
        
        sats_copy = copy.deepcopy(sats) # Reset the satellites to initial condition for the next trial

    # Average FIM and covariance history
    fim = np.mean(fim, axis=0)
    cov_hist = np.mean(cov_hist,axis=0)

    # With the given covariance matrix over all time steps, we can calculate the trace of the covariance matrix
    # We divide this by the number of satellites to get satellite-normalize trace of the covariance matrix
    cov_trace = np.zeros((N))
    for i in range(N):
        cov_trace[i] = np.trace(cov_hist[i,:,:])/n_sats

    # Invert FIM to get crb
    crb = np.zeros_like(fim)
    crb_trace = np.zeros((N))
    for i in range(N):
        crb[i,:,:] = np.linalg.inv(fim[i,:,:])
        crb_trace[i] = np.trace(crb[i,:,:])/n_sats

    if not os.path.exists("data"):
        os.makedirs("data")

    np.save(f'data/{n_sats}_cov_trace.npy', cov_trace)
    np.save(f'data/{n_sats}_crb_trace.npy', crb_trace)

    # Plotting

    # Plot the trajectory of the satellite
    # plot_trajectory(x_traj, filter_position, N)

    # Plot the positional error
    # plot_position_error(pos_error)

    # Plot crb and cov trace
    # plot_covariance_crb_trace(crb_trace, cov_trace)

    # plt.show()


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
    parser.add_argument('--R_weight', type=float, default=10e-5, help='Measurement noise weight')
    parser.add_argument('--f', type=float, default=1, help='Frequency of the simulation')
    parser.add_argument('--ignore_earth', action="store_true", default=False, help='Ignore the Earth from blocking measurements. Only applies to range measurements. \
                        Bearing measurements always consider the earth.')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of Monte Carlo trials')
    parser.add_argument('--n_sats', type=int, default=1, help='Number of satellites')
    parser.add_argument('--random_yaml',action="store_true", default=False, help='Use random satellite configuration')
    parser.add_argument('--run_all', action="store_true", default=False, help='Run simulations for all number of satellites from 1 to n_sats')
    parser.add_argument('--state_dim', type=int, default=6, help='Dimension of the state vector')
    parser.add_argument('--verbose', action="store_true", default=False, help='Print information')
    args = parser.parse_args()

    os.system(f"rm -r data")

    if args.random_yaml:
        if not os.path.exists("config"):
            os.makedirs("config")
        generate_satellites_yaml(filename="config/sat_autogen.yaml", n_sats=args.n_sats)

    if args.run_all:
        for i in range(1, args.n_sats+1):
            args.n_sats = i
            run_simulation(args)
    else:
        run_simulation(args)


    plot_all_sat_crb_trace()
    plt.show()
    
