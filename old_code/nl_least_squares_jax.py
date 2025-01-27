from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import yaml
from pyomo.environ import *
from pyomo.dae import *
import cyipopt
import jax
import jax.numpy as jnp
# import sympy as sp

# Constants
MU = 3.986004418 * 10**5 # km^3/s^2
R_EARTH = 6378
R_SAT = 550 + R_EARTH

class landmark: # Class for the landmark object. Coordinates in ECEF (TODO: Check with Paulo or Zac if this is correct)
    def __init__(self, x: float, y: float, z: float, name: str) -> None:
        self.pos = np.array([x,y,z])
        self.name = name

class satellite:

    def __init__(self, state: np.ndarray, rob_cov_init: float, robot_id: int, dim: int, meas_dim: int, Q_weight: float, R_weight: float, N: int, n_sats: int, landmarks: object) -> None:
        self.x_0 = state
        self.cov_m = rob_cov_init*np.eye(dim)
        self.id = robot_id # Unique identifier for the satellite
        self.dim = dim # State dimension of the satellite (currently 3 position + 3 velocity)
        self.meas_dim = meas_dim
        self.R_weight = R_weight # This is the variance weight for the measurement noise
        self.Q_weight = Q_weight # This is the variance weight for the process noise
        self.N = N # Number of time steps
        self.n_sats = n_sats # Number of satellites
        self.landmarks = landmarks
        
        # self.info_matrix = np.linalg.inv(self.cov_m)
        # self.info_vector = self.info_matrix@self.x_0
        # self.x_p = self.x_0 # Initialize the prior vector exactly the same as the measurement vector
        # self.cov_p = self.cov_m # Initialize the prior covariance exactly the same as the measurement covariance

        self.min_landmark_list = None # Initialize an array of the closest landmark to the satellite t
        self.curr_pos = self.x_0[0:3] #Determines the current position of the satellite (Necessary for landmark bearing and satellite ranging)
        self.other_sats_pos = np.zeros((N, 3, int(n_sats-1))) # Provides the position of the other satellites for all N timesteps

    def h_landmark(self, x):
        min_landmark = closest_landmark(x,self.landmarks)
        norm = jnp.sqrt((x[0] - min_landmark[0])**2 + (x[1] - min_landmark[1])**2 + (x[2] - min_landmark[2])**2)
        
        return (x - min_landmark)/norm
    
    def h_inter_range(self, i, j, x): # This function calculates the range measurement between the satellite and another satellite
        # timestep = math.floor((i-1)/3)
        sat_id = j-3 # j is the range measurement index startj g
        sat_pos = self.other_sats_pos[i,:,sat_id]
        norm = jnp.sqrt((x[0] - sat_pos[0])**2 + (x[1] - sat_pos[1])**2 + (x[2] - sat_pos[2])**2)
        return norm
    
    def measure_z_range(self, sats: list) -> np.ndarray:
        z = np.empty((0))
        for sat in sats:
            if sat.id != self.id:
                # TODO: Implement a check to see whether the earth is in the way. Right now we are just measuring the straight line distances, potentially through the earth.
                d = np.array([np.linalg.norm(self.curr_pos - sat.curr_pos)]) + np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(1))
                z = np.append(z,d,axis=0)
        return z
    
    def measure_z_landmark(self, landmarks: list) -> np.ndarray:
        # Determine the closest landmark to the satellite at the current state and calculate the bearing to that landmark
        if self.min_landmark_list is None:
            self.min_landmark_list = np.array([closest_landmark(self.curr_pos, landmarks)]).reshape(1,3)
        else:
            self.min_landmark_list = np.append(self.min_landmark_list,[closest_landmark(self.curr_pos, landmarks)],axis=0)
        
        vec = (self.curr_pos - self.min_landmark_list[-1]) + np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(int(self.dim/2)))
        return vec/np.linalg.norm(self.curr_pos - self.min_landmark_list[-1])
    

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

    return closest_landmark.pos

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

# Numerical solution using RK4 method
def rk4_discretization(x: np.ndarray, dt: float) -> np.ndarray:
    r = x[0:3]
    v = x[3:6]

    def dr_dt(v):
        """Derivative of r with respect to time is velocity v."""
        return v

    def dv_dt(r):
        """Derivative of v with respect to time is gravitational acceleration."""
        return (-MU / (np.linalg.norm(r)**3)) * r

    # Calculate k1 for r and v
    k1_r = dr_dt(v)
    k1_v = dv_dt(r)

    # Calculate k2 for r and v
    k2_r = dr_dt(v + 0.5 * dt * k1_v)
    k2_v = dv_dt(r + 0.5 * dt * k1_r)

    # Calculate k3 for r and v
    k3_r = dr_dt(v + 0.5 * dt * k2_v)
    k3_v = dv_dt(r + 0.5 * dt * k2_r)

    # Calculate k4 for r and v
    k4_r = dr_dt(v + dt * k3_v)
    k4_v = dv_dt(r + dt * k3_r)

    # Combine the k terms to get the next position and velocity
    r_new = r + (dt / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    v_new = v + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    # Return the updated state vector
    return np.concatenate((r_new, v_new))

# Numerical solution of the dynamics using Euler's method
def euler_discretization(x: np.ndarray, dt: float) -> np.ndarray:
    r = x[0:3]
    v = x[3:6]
    r_new = r + v*dt
    v_new = v + (-MU/(np.linalg.norm(r)**3))*r*dt
    return np.concatenate((r_new, v_new))

### Landmark Initialization ###
# Import csv data for the landmarks
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

#Parameters
N = 10
f = 1 #Hz
dt = 1/f
n_sats = 3
R_weight = 0.1
bearing_dim = 3
meas_dim = n_sats-1 + bearing_dim
R = np.eye(meas_dim)*R_weight
state_dim = 6


x_traj = np.zeros((N, 6, n_sats)) # Discretized trajectory of satellite states over time period 

### Satellite Initialization ###

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

sats = []

for sat_config in config["satellites"]:
    sat_config["state"] = np.array(sat_config["state"])
    sat_config["state"][5] = np.sqrt(MU/R_SAT) # for now assign velocity that generates a circular orbit
    sat_config["N"] = N
    sat_config["landmarks"] = landmark_objects
    sat_config["meas_dim"] = meas_dim
    sat_config["n_sats"] = n_sats
    sat_config["R_weight"] = R_weight
    satellite_inst = satellite(**sat_config)
    sats.append(satellite_inst)


# Generate synthetic data
np.random.seed(0)        # For reproducibility
time_data = np.linspace(0, N, N*f)  # Time points
#

for sat in sats: 
    x = sat.x_0
    for i in range(N):
        x_traj[i,:,sat.id] = x
        x = euler_discretization(x, dt)

y_m = jnp.zeros((N,meas_dim,n_sats))

for i in range(N):

    for sat in sats:
        sat.curr_pos = x_traj[i,0:3,sat.id]

    for sat in sats:
        y_m = y_m.at[i,0:bearing_dim,sat.id].set(sat.measure_z_landmark(landmark_objects)) # This sets the bearing measurement
        # y_m[i,0:bearing_dim,sat.id] = sat.measure_z_landmark(landmark_objects) # This sets the bearing measurement
        y_m = y_m.at[i,bearing_dim:meas_dim,sat.id].set(sat.measure_z_range(sats)) # This sets the range measurement
        # y_m[i,bearing_dim:meas_dim,sat.id] = sat.measure_z_range(sats) # This sets the range measurement

# Get the positions of the other satellites for each satellite
for sat in sats:
    sat_i = 0 #iterator variable
    for other_sat in sats:
        if sat.id != other_sat.id:
            # for i in range(N):
                # sat.other_sats_pos[i,:,sat_i] = x_traj[i,0:3,other_sat.id] # Transfer all N 3D positions of the other satellites from x_traj
            sat.other_sats_pos[:,:,sat_i] = x_traj[:,0:3,other_sat.id] # Transfer all N 3D positions of the other satellites from x_traj
            sat_i += 1

# Solve the problem using cyipopt

for sat in sats: 
    class trajSolver:
        def __init__(self, x_traj, y_m, time_data, sat, N, meas_dim, bearing_dim, n_sats, MU):
            self.x_traj = x_traj
            self.y_m = y_m
            self.time_data = time_data
            self.sat = sat
            self.N = N
            self.meas_dim = meas_dim
            self.bearing_dim = bearing_dim
            self.n_sats = n_sats
            self.MU = MU

        def objective(self, x):
            obj = 0
            for i in range(self.N):
                start_i = i*3
                # TODO Include covariance matrix in the objective function
                # print(y_m[i,0:3,sat.id])
                obj += (jnp.sum(y_m[i,0:3,self.sat.id] - self.sat.h_landmark(x[start_i:start_i+3])))**2
                for j in range(3,self.meas_dim):
                    # print(y_m[i,j,sat.id])
                    # index = start_i+j
                    term= (y_m[i,j,self.sat.id] - self.sat.h_inter_range(i, j, x[start_i:start_i+3]))**2
                    # print(term)
                    obj += term
                    
            return obj

        def gradient(self, x):
            grad = jax.grad(self.objective)(x)

            return grad

        def constraints(self, x):
            dim = 3
            g = jnp.zeros(self.N*dim)
            g = g.at[0].set(x[0] - self.sat.x_0[0]) 
            g = g.at[1].set(x[1] - self.sat.x_0[1])
            g = g.at[2].set(x[2] - self.sat.x_0[2])
            
            for i in range(1,self.N-2):
                norm_pos = x[i*dim]**2 + x[i*dim+1]**2 + x[i*dim+2]**2
                
                # constraints combining position and velocity using central difference
                g = g.at[i*dim + 0].set(x[(i+2)*dim+ 0] - 2*x[(i+1)*dim + 0] + x[i*dim + 0] + (dt**2)*self.MU*x[i*dim + 0]/norm_pos**3)
                g = g.at[i*dim + 1].set(x[(i+2)*dim+ 1] - 2*x[(i+1)*dim + 1] + x[i*dim + 1] + (dt**2)*self.MU*x[i*dim + 1]/norm_pos**3)
                g = g.at[i*dim + 2].set(x[(i+2)*dim+ 2] - 2*x[(i+1)*dim + 2] + x[i*dim + 2] + (dt**2)*self.MU*x[i*dim + 2]/norm_pos**3)

            # # Backward
            for i in range(self.N-2,self.N):
                norm_pos = x[i*dim]**2 + x[i*dim+1]**2 + x[i*dim+2]**2
                g = g.at[i*dim+0].set(x[(i)*dim+0] - 2*x[(i-1)*dim+0] + x[(i-2)*dim+0] + (dt**2)*self.MU*x[i*dim+0]/norm_pos**3)
                g = g.at[i*dim+1].set(x[(i)*dim+1] - 2*x[(i-1)*dim+1] + x[(i-2)*dim+1] + (dt**2)*self.MU*x[i*dim+1]/norm_pos**3)
                g = g.at[i*dim+2].set(x[(i)*dim+2] - 2*x[(i-1)*dim+2] + x[(i-2)*dim+2] + (dt**2)*self.MU*x[i*dim+2]/norm_pos**3)



            # g[0] = x[(i)*dim] - 2*x[(i-1)*dim] + x[(i-2)*dim] + (dt**2)*self.MU*x[i*dim]/norm_pos**3
            # g[1] = x[(i)*dim+1] - 2*x[(i-1)*dim+1] + x[(i-2)*dim+1] + (dt**2)*self.MU*x[i*dim+1]/norm_pos**3
            # g[2] = x[(i)*dim+2] - 2*x[(i-1)*dim+2] + x[(i-2)*dim+2] + (dt**2)*self.MU*x[i*dim+2]/norm_pos**3

            return g


        def jacobian(self, x):
            jacobian = jax.jacfwd(self.constraints)(x)
            jacobian = jacobian[self.jrow, self.jcol]
            
            return jacobian


        # TODO: Implement the hessian and the jacobian structure functions
        # def hessian(self, x, lagrange, obj_factor):
        #     pass

        def jacobianstructure(self):
            dim = 3
            row = []
            col = []

            # Initial conditions
            row.extend([0,1,2])
            col.extend([0,1,2])

            for i in range(1,N-2):

                row.append(i*dim)
                col.append(i*dim)
                
                row.append(i*dim)
                col.append(i*dim+1)

                row.append(i*dim)
                col.append(i*dim+2)

                row.append(i*dim)
                col.append(i*dim+3)
                
                row.append(i*dim)
                col.append(i*dim+6)

                row.append(i*dim+1)
                col.append(i*dim)

                row.append(i*dim+1)
                col.append(i*dim+1)

                row.append(i*dim+1)
                col.append(i*dim+2)

                row.append(i*dim+1)
                col.append(i*dim+4)

                row.append(i*dim+1)
                col.append(i*dim+7)

                row.append(i*dim+2)
                col.append(i*dim)

                row.append(i*dim+2)
                col.append(i*dim+1)

                row.append(i*dim+2)
                col.append(i*dim+2)

                row.append(i*dim+2)
                col.append(i*dim+5)

                row.append(i*dim+2)
                col.append(i*dim+8)


            # Columns will carry indices outside the range of N*dim Remove these indices in the row and column arrays
            for elem in col: 
                if elem >= N*dim:
                    index = col.index(elem)
                    # remove element at index
                    col = col[:index] + col[index+1:]
                    row = row[:index] + row[index+1:]

            # final two timesteps require backward difference
            for i in range(N-2,N):

                row.append(i*dim)
                col.append(i*dim-6)
                        
                row.append(i*dim)
                col.append(i*dim-3)
                
                row.append(i*dim)
                col.append(i*dim)

                row.append(i*dim)
                col.append(i*dim+1)

                row.append(i*dim)
                col.append(i*dim+2)

                row.append(i*dim+1)
                col.append(i*dim-5)
                
                row.append(i*dim+1)
                col.append(i*dim-2)

                row.append(i*dim+1)
                col.append(i*dim)

                row.append(i*dim+1)
                col.append(i*dim+1)

                row.append(i*dim+1)
                col.append(i*dim+2)

                row.append(i*dim+2)
                col.append(i*dim-4)

                row.append(i*dim+2)
                col.append(i*dim-1)

                row.append(i*dim+2)
                col.append(i*dim)

                row.append(i*dim+2)
                col.append(i*dim+1)

                row.append(i*dim+2)
                col.append(i*dim+2)
                
            # jac = np.zeros((self.N*3,self.N*3))
            # for i in range(len(row)):
            #     jac[row[i],col[i]] = 11

            # print(jac)
            self.jrow = np.array(row, dtype=int)
            self.jcol = np.array(col, dtype=int)

            return np.array(row, dtype=int), np.array(col, dtype=int)

        # def hessianstructure(self):
        #     pass

    solver = trajSolver(x_traj, y_m, time_data, sat, N, meas_dim, bearing_dim, n_sats, MU)
    nlp = cyipopt.Problem(
        n = N*3,
        m = N*3,
        problem_obj = solver,
        lb = jnp.full(N*3,-np.inf),
        ub = jnp.full(N*3,np.inf),
        cl = jnp.zeros(N*3),
        cu = jnp.zeros(N*3)
    )

    # glob_x0 = [1]*(N*3)
    # glob_x0 = np.array(sat.x_0[0:3])
    glob_x0 = np.array([10,10,10])
    glob_x0 = np.tile(glob_x0, N)
    nlp.add_option('max_iter', 500)
    nlp.add_option('tol', 1e-6)
    nlp.add_option('print_level', 5)
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('hessian_approximation', 'limited-memory')
    nlp.add_option('limited_memory_max_history', 10)
    nlp.add_option('limited_memory_max_skipping', 2)

    x, info = nlp.solve(glob_x0)
    print(x)


    # TODOs:
    # Exploit Jacobian sparsity in Jacobian structure function
    # Implement Hessian and its structure functions
    # Check correctness of results and indexing operations
    # RK4 rather than Euler
