import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import yaml
import cyipopt
import jax
import jax.numpy as jnp
# from functools import partial

from satellite import satellite
from sat_dynamics import rk4_discretization, state_transition
from landmark import latlon2ecef, landmark
from traj_solver import trajSolver

# Constants
MU = 3.986004418 * 10**5 # km^3/s^2
R_EARTH = 6378
R_SAT = 550 + R_EARTH

class landmark: # Class for the landmark object. Coordinates in ECEF (TODO: Check with Paulo or Zac if this is correct)
    def __init__(self, x: float, y: float, z: float, name: str) -> None:
        self.pos = np.array([x,y,z])
        self.name = name


# Helper function to determine closest landmark to the satellite at the current state position based on Euclidean distance
def closest_landmark(pos, landmarks: list) -> object:
    """
    Find the closest landmark position to the current position.

    Args:
        pos (np.ndarray): The current position of the satellite.
        landmarks (list): A list of landmark objects, each having a 'pos' attribute.

    Returns:
        The position of the landmark object that is closest to the current position. 
        If no landmarks are provided, returns None.
    """
    min_dist = np.inf
    closest_landmark = None

    for landmark in landmarks:
        dist = jnp.linalg.norm(pos - landmark.pos)
        if dist < min_dist:
            min_dist = dist
            closest_landmark = landmark

    # print(f"Closest landmark to satellite at position {pos} is {closest_landmark.name} at position {closest_landmark.pos}")
    return closest_landmark.pos

# Numerical solution of the dynamics using Euler's method
def euler_discretization(x: np.ndarray, dt: float) -> np.ndarray:
    r = x[0:3]
    v = x[3:6]
    r_new = r + v*dt
    v_new = v + (-MU/(np.linalg.norm(r)**3))*r*dt
    return np.concatenate((r_new, v_new))

# Function to solve the non-linear least squares problem
def solve_nls(x_traj, nlp, sat_id):
    # Randomize initia; guess
    glob_x0 = x_traj[:,:,sat_id] + np.random.normal(loc=0,scale=math.sqrt(R_weight),size=(N,state_dim))
    # glob_x0[:,3] = 0.0
    # glob_x0[:,4] = 0.0
    # glob_x0[:,5] = np.sqrt(MU/R_SAT)
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


## MAIN LOOP START; TODO: Implement this into a main loop and make argparse callable
#General Parameters
N = 10
f = 1/60 #Hz
dt = 1/f
n_sats = 1
R_weight = 0.1
state_dim = 6

#MC Parameters
num_trials = 1
nls_estimates = np.zeros((num_trials, state_dim * N))

# Do not seed in order for Monte-Carlo simulations to actually produce different outputs!
# np.random.seed(42)        #Set seed for reproducibility

### Landmark Initialization ###
# Import csv data for the lanldmarks
landmarks = []
with open('landmark_coordinates.csv', newline='',) as csvfile:
    reader = csv.reader(csvfile, delimiter=',',)
    for row in reader:
        landmarks.append(np.array([row[0],row[1], row[2], row[3]]))


landmarks_ecef = latlon2ecef(landmarks)
landmark_objects = []

# Initialize the landmark objects with their correct name and the ECEF coordinates
for landmark_obj in landmarks_ecef:
    landmark_objects.append(landmark(x=float(landmark_obj[1]), y=float(landmark_obj[2]), z=float(landmark_obj[3]), name=(landmark_obj[0])))

bearing_dim = len(landmark_objects)*3
meas_dim = n_sats-1 + bearing_dim
R = np.eye(meas_dim)*R_weight

### Satellite Initialization ###
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

sats = []

for sat_config in config["satellites"]:
    # sat_config["state"] = np.array(sat_config["state"])
    # sat_config["state"][5] = float(np.sqrt(MU/R_SAT)) # for now assign velocity using vis-viva equation
    sat_config["N"] = N
    sat_config["landmarks"] = landmark_objects
    sat_config["meas_dim"] = meas_dim
    sat_config["n_sats"] = n_sats
    sat_config["R_weight"] = R_weight
    sat_config["bearing_dim"] = bearing_dim

    satellite_inst = satellite(**sat_config)
    sats.append(satellite_inst)


# Generate synthetic data
x_traj = np.zeros((N, state_dim, n_sats)) # Discretized trajectory of satellite states over time period
# This loop is deterministic as we are always doing the same discretization so we do not need to regenerate the trajectories.
for sat in sats: 
    x = sat.x_0
    for i in range(N):
        x_traj[i,:,sat.id] = x
        x = rk4_discretization(x, dt)

# Get the positions of the other satellites for each satellite at all N timesteps
for sat in sats:
        sat_i = 0 #iterator variable
        for other_sat in sats:
            if sat.id != other_sat.id:
                sat.other_sats_pos[:,:,sat_i] = x_traj[:,0:3,other_sat.id] # Transfer all N 3D positions of the other satellites from x_traj
                sat_i += 1

# NOTE: We are only doing these trials for one satellites (for now)

# Generate the Jacobian for FIM
x_true = x_traj[:,:,0]


## Calculate FIM directly
fim = np.zeros((N*state_dim, N*state_dim))
f_prior = np.zeros((state_dim, state_dim))
f_post = np.zeros((state_dim, state_dim))

J = np.zeros((meas_dim, state_dim))
R_inv = np.linalg.inv(R)

for i in range(1,N):
    #Assume no knowledge at initial state so we don't place any information in the first state_dim x state_dim block
    start_i = i * state_dim
    A = state_transition(x_true[i-1,:])
    f_prior = A@f_post@A.T # No process noise
    J[0:bearing_dim,0:state_dim] = sats[0].H_landmark(x_true[i])
    for j in range(bearing_dim,meas_dim): ## Consider checks for nan values
        J[j,0:state_dim] = sats[0].H_inter_range(i, j, x_true[i])
    f_post = f_prior + J.T@R_inv@J
    fim[start_i:start_i+state_dim,start_i:start_i+state_dim] = f_post

fim = np.diag(fim).reshape(-1,6)
print(np.diag(fim).reshape(-1,6))
plt.plot(fim[:,0], label='x', color='green')
plt.plot(fim[:,1], label='y', color='red',linestyle='dashed')
plt.plot(fim[:,2], label='z', color='blue',linestyle='dotted')
plt.show()

for trial in range(num_trials):

    y_m = np.zeros((N,meas_dim,n_sats))

    for i in range(N):

        for sat in sats:
            sat.curr_pos = x_traj[i,0:3,sat.id]

        for sat in sats:
            y_m[i,0:bearing_dim,sat.id] = (sat.measure_z_landmark()) # This sets the bearing measurement
            y_m[i,bearing_dim:meas_dim,sat.id] = sat.measure_z_range(sats) # This sets the range measurement


    # Initialize the solver 
    # SOLVING ALWAYS WITH FIRST SATELLITE!!!
    solver = trajSolver(
        x_traj=x_traj, 
        y_m=y_m, 
        sat=sats[0], 
        N=N, 
        meas_dim=meas_dim, 
        bearing_dim=bearing_dim, 
        n_sats=n_sats, 
        MU=MU, 
        state_dim=state_dim,
        dt=dt
        )
    
    nlp = cyipopt.Problem(
        n = N*state_dim,
        m = N*state_dim,
        problem_obj = solver,
        lb = jnp.full(N*state_dim,-np.inf),
        ub = jnp.full(N*state_dim,np.inf),
        cu = jnp.zeros(N*state_dim),
        cl = jnp.zeros(N*state_dim),
    )


    x = solve_nls(x_traj, nlp, sat_id=0)
    nls_estimates[trial] = x



#Compute sample covariance matrix for all parameters
# sample_var = np.var(nls_estimates, axis=0)

# print("sample var", sample_var)
