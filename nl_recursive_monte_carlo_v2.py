import argparse
import copy
import csv
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils.plotting_utils import plot_covariance_crb, plot_trajectory, plot_position_error
from landmark import landmark
from satellite import satellite

# Constants
MU = 3.986004418 * 10**5 # km^3/s^2 # Gravitational parameter of the Earth
HEIGHT = 550 # km # Height of the satellite
ROH_0 = 1.225e9 # kg/km^3 # Density of air at sea level
H_0 = 0 # km # Height of sea level
MASS = 2 # kg # Mass of the satellite
AREA = 1e-9 # km^2 # Cross-sectional area of the satellite
SCALE_HEIGHT = 8.4 # km # Scale height of the atmosphere
C_D = 2.2 # Drag coefficient of the satellite
EQ_RADIUS = 6378.1370 # km # Equatorial radius of the Earth
POLAR_RADIUS = 6356.7523 # km # Polar radius of the Earth
J2 = 1.08263e-3 # J2 perturbation coefficient

DENSITY = ROH_0*np.exp(-(HEIGHT-H_0)/SCALE_HEIGHT) # kg/km^3 # Density of the atmosphere at the height of the satellite


def latlon2ecef(landmarks: list) -> np.ndarray:
    """
    Convert latitude, longitude, and altitude coordinates to Earth-Centered, Earth-Fixed (ECEF) coordinates.

    Args:
        landmarks (list of tuples): A list of tuples where each tuple contains:
            - landmark[0] (any): An identifier for the landmark.
            - landmark[1] (float): Latitude in radians.
            - landmark[2] (float): Longitude in radians.
            - landmark[3] (float): Altitude in kilometers.
    Returns:
        numpy.ndarray: A 2D array where each row corresponds to a landmark and contains:
            - landmark[0] (any): The identifier for the landmark.
            - X (float): The ECEF X coordinate in kilometers.
            - Y (float): The ECEF Y coordinate in kilometers.
            - Z (float): The ECEF Z coordinate in kilometers.
    """
    ecef = np.array([])
    a = 6378.137
    b = 6356.7523
    e = 1 - (b**2/a**2)
    
    # helper function
    def N(a,b,lat):
        return a**2/np.sqrt(a**2*np.cos(lat)**2 + b**2*np.sin(lat)**2)
    
    for landmark in landmarks:
        X = (N(a,b,float(landmark[1])) + float(landmark[3]))*np.cos(float(landmark[1]))*np.cos(float(landmark[2]))
        Y = (N(a,b,float(landmark[1])) + float(landmark[3]))*np.cos(float(landmark[1]))*np.sin(float(landmark[2]))
        Z = (N(a,b,float(landmark[1]))*(1-e) + float(landmark[3]))*np.sin(float(landmark[1]))
        ecef = np.append(ecef, np.array([landmark[0],X,Y,Z]), axis=0)
    
    return ecef.reshape(-1,4)

def gravitational_acceleration(r):
    return (-MU / (jnp.linalg.norm(r)**3)) * r

# TODO: Refactor to make this dependent on the satellite object specifically. 
# Reason: We assume that all satellites have the same mass and area values which is not necessarily true
def atmospheric_drag(v):
    drag = (-0.5*C_D*DENSITY*AREA*v*np.linalg.norm(v)**2)/MASS
    return drag

def j2_dynamics(r):
    r_norm = jnp.linalg.norm(r)

    F = 3*MU*J2*EQ_RADIUS**2/(2*r_norm**5)
    a_x = F*(r[0])*(5*(r[2]/r_norm)**2 - 1)
    a_y = F*(r[1])*(5*(r[2]/r_norm)**2 - 1)
    a_z = F*(r[2])*(5*(r[2]/r_norm)**2 - 3)

    return jnp.array([a_x, a_y, a_z])

def j2_jacobian(r):
    jac = jax.jacobian(j2_dynamics)(r)
    return jac

# Numerical solution using RK4 method
def rk4_discretization(x, dt: float):
    r = x[0:3]
    v = x[3:6]

    def dr_dt(v):
        """Derivative of r with respect to time is velocity v."""
        return v

    def dv_dt(r, v):
        """Derivative of v with respect to time is gravitational acceleration."""
        return gravitational_acceleration(r) + atmospheric_drag(v) + j2_dynamics(r)

    # Calculate k1 for r and v
    k1_r = dr_dt(v)
    k1_v = dv_dt(r, v)

    # Calculate k2 for r and v
    k2_r = dr_dt(v + 0.5 * dt * k1_v)
    k2_v = dv_dt(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v)

    # Calculate k3 for r and v
    k3_r = dr_dt(v + 0.5 * dt * k2_v)
    k3_v = dv_dt(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v)

    # Calculate k4 for r and v
    k4_r = dr_dt(v + dt * k3_v)
    k4_v = dv_dt(r + dt * k3_r, v + dt * k3_v)

    # Combine the k terms to get the next position and velocity
    r_new = r + (dt / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    v_new = v + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    # Return the updated state vector
    return jnp.concatenate((r_new, v_new))


# Numerical solution of the dynamics using Euler's method
def euler_discretization(x: np.ndarray, dt: float) -> np.ndarray:
    r = x[0:3]
    v = x[3:6]
    r_new = r + v*dt
    v_new = v + (-MU/(np.linalg.norm(r)**3))*r*dt
    return np.concatenate((r_new, v_new))


def state_transition(x):
    I = np.eye(3)
    # Account for j2 dynamics
    j2_jac = j2_jacobian(x[0:3])
    
    # Account for gravitational acceleration
    grav_jac = -MU/(np.linalg.norm(x[0:3])**3)*I + 3*MU*np.outer(x[0:3],x[0:3])/(np.linalg.norm(x[0:3])**5)
    
    # Account for atmospheric drag
    drag_jac = -DENSITY*C_D*AREA/(2*MASS)*(np.outer(x[3:6],x[3:6])/np.linalg.norm(x[3:6]) + I*np.linalg.norm(x[3:6]))
    
    A = np.block([[np.zeros((3,3)), I], [grav_jac+j2_jac, drag_jac]])
    return A

if __name__ == "__main__":
    ## MAIN LOOP START; TODO: Implement this into a main loop and make argparse callable
    #General Parameters
    parser = argparse.ArgumentParser(description='Nonlinear Recursive Monte Carlo Simulation')
    parser.add_argument('--N', type=int, default=400, help='Number of timesteps')
    parser.add_argument('--f', type=float, default=1, help='Frequency of the simulation')
    parser.add_argument('--n_sats', type=int, default=1, help='Number of satellites')
    parser.add_argument('--R_weight', type=float, default=10e-4, help='Measurement noise weight')
    parser.add_argument('--bearing_dim', type=int, default=3, help='Dimension of the bearing measurement')
    parser.add_argument('--state_dim', type=int, default=6, help='Dimension of the state vector')
    parser.add_argument('--num_trials', type=int, default=2, help='Number of Monte Carlo trials')
    args = parser.parse_args()

    N = args.N
    f = args.f #Hz
    n_sats = args.n_sats
    R_weight = args.R_weight
    bearing_dim = args.bearing_dim
    state_dim = args.state_dim

    dt = 1/f
    meas_dim = n_sats-1 + bearing_dim   
    R = np.eye(meas_dim)*R_weight
    # Process noise covariance matrix based on paper "Autonomous orbit determination and observability analysis for formation satellites" by OU Yangwei, ZHANG Hongbo, XING Jianjun
    # page 6
    Q = np.diag(np.array([10e-6,10e-6,10e-6,10e-12,10e-12,10e-12]))

    #MC Parameters
    num_trials = args.num_trials

    # Do not seed in order for Monte-Carlo simulations to actually produce different outputs!
    # np.random.seed(42)        #Set seed for reproducibility

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
    f_prior = np.zeros((state_dim, state_dim))
    f_post = np.zeros((state_dim, state_dim))

    J = np.zeros((meas_dim, state_dim))
    R_inv = np.linalg.inv(R)

    cov_hist = np.zeros((N,n_sats,state_dim,state_dim))
    sats_copy = copy.deepcopy(sats)

    filter_position = np.zeros((num_trials, N, 3, n_sats))
    pos_error = np.zeros((num_trials, N , 3, n_sats))


    for trial in range(num_trials):
        print("Monte Carlo Trial: ", trial)

        y_m = jnp.zeros((N,meas_dim,n_sats))
        H = np.zeros((meas_dim,state_dim))
        h = np.zeros((meas_dim))

        for i in range(N):
            print(i)

            for sat in sats_copy:
                sat.curr_pos = x_traj[i+1,0:3,sat.id] #Provide the underlying groundtruth position to the satellite for bearing and ranging measurements
                A = state_transition(sat.x_m)
                sat.x_p = rk4_discretization(sat.x_m, dt)
                sat.cov_p = A@sat.cov_m@A.T + Q # Update the covariance matrix as a P_p = A*P*A^T + LQL^T


            for sat in sats_copy:
                H[0:bearing_dim,0:state_dim] = sat.H_landmark(sat.x_p)
                for j in range(bearing_dim,meas_dim):
                    H[j,:] = sat.H_inter_range(i+1, j, sat.x_p)

                K = sat.cov_p@H.T@np.linalg.pinv(H@sat.cov_p@H.T + R)
                
                # Check if the camera is on for the satellite to take bearing measurements to landmarks
                if sat.camera_exists:
                    y_m = y_m.at[i,0:bearing_dim,sat.id].set(sat.measure_z_landmark(tuple(landmark_objects)))
                    h[0:bearing_dim] = sat.h_landmark(sat.x_p[0:3])
                else :
                    y_m = y_m.at[i,0:bearing_dim,sat.id].set(np.zeros((bearing_dim,))) # Set to zero if camera is off
                    h[0:bearing_dim] = np.zeros((bearing_dim,))

                # Range measurements always take place regardless of camera status
                y_m = y_m.at[i,bearing_dim:meas_dim,sat.id].set(sat.measure_z_range(sats_copy)) # This sets the range measurement

                for j in range(bearing_dim,meas_dim):
                    h[j] = sat.h_inter_range(i+1, j, sat.x_p[0:3])
                
                sat.x_m = sat.x_p + K@(y_m[i,:,sat.id] - h)
                sat.cov_m = (np.eye(state_dim) - K@H)@sat.cov_p@((np.eye(state_dim) - K@H).T) + K@R@K.T

                cov_hist[i,sat.id,:,:] += sat.cov_m

                filter_position[trial,i,:,sat.id] = sat.x_m[0:3]
                pos_error[trial, i,:,sat.id] = filter_position[trial,i,:,sat.id] - sat.curr_pos[0:3]


            #Assume no knowledge at initial state so we don't place any information in the first state_dim x state_dim block
            if i > 0:
                start_i = i * state_dim
                A = state_transition(sats_copy[0].x_m)
                f_prior = A@f_post@A.T + np.linalg.inv(Q) # with process noise
                J[0:bearing_dim,0:3] = sats_copy[0].H_landmark(sats_copy[0].x_m[0:3])
                for j in range(3,meas_dim): ## Consider checks for nan values
                    J[j,0:3] = sats_copy[0].H_inter_range(i+1, j, sats_copy[0].x_m[0:3])
                f_post = f_prior + J.T@R_inv@J

                # Check if f_post is invertible
                # if np.linalg.matrix_rank(f_post) == state_dim:
                #     print("FIM is invertible")
                # else: 
                #     print("FIM is not invertible")
                #     print(f_post)
                fim[trial, start_i:start_i+state_dim,start_i:start_i+state_dim] = f_post

                # print(f"Satellite {sat.id} at time {i} has covariance {sat.cov_m}")
        
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
