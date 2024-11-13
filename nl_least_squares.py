# Satellite flies at an altitude of ca. 550km above the earth's surface
# The earth's radius is 6378km and the gravitational parameter is 3.986004418 x 10^5 km^3/s^2

from gekko import GEKKO
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

# Constants
MU = 3.986004418 * 10**5 # km^3/s^2
R_EARTH = 6378
R_SAT = 550 + R_EARTH

class landmark: # Class for the landmark object. Coordinates in ECEF (TODO: Check with Paulo or Zac if this is correct)
    def __init__(self, x: float, y: float, z: float, name: str) -> None:
        """
        Initialize a new instance of the class.

        Args:
            x (float): The x-coordinate of the position.
            y (float): The y-coordinate of the position.
            z (float): The z-coordinate of the position.
            name (str): The name associated with the position.

        Returns:
            None
        """
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

        self.info_matrix = np.linalg.inv(self.cov_m)
        self.info_vector = self.info_matrix@self.x_0
        self.x_p = self.x_0 # Initialize the prior vector exactly the same as the measurement vector
        self.cov_p = self.cov_m # Initialize the prior covariance exactly the same as the measurement covariance

        self.curr_pos = self.x_0[0:3] #Determines the current position of the satellite (Necessary for landmark bearing and satellite ranging)
        self.other_sats_pos = np.ndarray(shape=(n_sats-1,3)) # Determines the current position of the other satellites

        # Gekko optimization variables
        self.m = GEKKO(remote=False)
        self.m.time = np.linspace(0, 1, 2) # Time vector
        self.opt_state = self.m.Array(self.m.Var, (N, dim)) # State vector to be optimized
        # self.y_est_landmark = [[self.m.Intermediate(self.h_landmark_gekko(self.opt_state,i,landmarks=self.landmarks)) for _ in range(meas_dim)] for i in range(N)]
        # self.y_est_inter_range = [[self.m.Intermediate(self.h_inter_range_gekko(i, self.other_sats_pos[j,:])) for j in range(n_sats-1)] for i in range(N)]
        # self.y_est_landmark = [[None]*3]*N
        # self.y_est_landmark = self.y_est_landmark*meas_dim
        # self.y_est_inter_range = [[None]*(n_sats-1)]*N
        # self.y_est_inter_range = self.y_est_inter_range*(n_sats-1)
        self.y_est = [[None]*meas_dim]*N

        print("Satellite initialized")



    # def h_landmark(self, landmarks: list) -> np.ndarray:
    #     min_landmark = closest_landmark(self.curr_pos, landmarks)
    #     return (self.curr_pos - min_landmark.pos)/np.linalg.norm(self.curr_pos - min_landmark.pos) # This provides a normalized vector (the bearing)

    # def h_inter_range(self, sat_pos: np.ndarray) -> np.ndarray:
    #     return np.array([np.linalg.norm(self.curr_pos - sat_pos)])

    # def h_combined(self, landmarks: list, sats: list) -> np.ndarray:
    #     h = self.h_landmark(landmarks)
    #     for sat in sats:
    #         if sat.id != self.id:
    #             h = np.append(h, self.h_inter_range(sat.curr_pos),axis=0)

    #     return h
    
    # def H_landmark(self, landmark: object) -> np.ndarray:
    #     return -(self.curr_pos - landmark.pos)@(self.curr_pos - landmark.pos).T/np.linalg.norm(self.curr_pos - landmark.pos)**3 + np.eye(self.dim/2)/np.linalg.norm(self.curr_pos - landmark.pos)
    
    # def H_inter_range(self, sat_pos: np.ndarray) -> np.ndarray:
    #     dx = (sat_pos[0] - self.curr_pos[0])/np.linalg.norm(self.curr_pos[0:3] - sat_pos)
    #     dy = (sat_pos[1] - self.curr_pos[1])/np.linalg.norm(self.curr_pos[0:3] - sat_pos)
    #     dz = (sat_pos[2] - self.curr_pos[2])/np.linalg.norm(self.curr_pos[0:3] - sat_pos)
    #     return np.array([[dx, dy, dz]])

    # # This function combines the H matrices for the bearing and inter-satellite range measurements
    # def combined_H(self, landmark: object, sats: list) -> np.ndarray:
    #     H = self.H_landmark(landmark)
    #     for sat in sats:
    #         if sat.id != self.id:
    #             H = np.vstack((H, self.H_inter_range(sat.curr_pos)))

    #     return H
    

    def closest_landmark_gekko(self, landmarks: list):
        min_dist = math.inf
        closest_landmark = None

        for landmark in landmarks:
            dist = np.linalg.norm(self.curr_pos - landmark.pos)
            if dist < min_dist:
                min_dist = dist
                closest_landmark = landmark

        return closest_landmark

    # def h_combined_gekko(self, index: int, landmarks: list, sats: list):
    #     h = self.m.Array(self.m.Var, (self.meas_dim))
    #     h[0:3] = self.h_landmark_gekko(index, landmarks)
    #     k = 0
    #     for sat in sats:
    #         if sat.id != self.id:
    #             tmp = self.h_inter_range_gekko(index, sat.curr_pos)
    #             h[k+3] = tmp # 3 dimensions are already taken up by the bearing measurement
    #             k += 1
        
    #     return h

    def h_landmark_gekko(self, opt_state, i: int, landmarks: list):
        min_landmark = self.closest_landmark_gekko(landmarks)
        norm = self.m.sqrt((opt_state[i, 0] - min_landmark.pos[0])**2 + (opt_state[i, 1] - min_landmark.pos[1])**2 + (opt_state[i, 2] - min_landmark.pos[2])**2)
        return (opt_state[i,0:3] - min_landmark.pos)/norm # TODO: Check if this subtraction between a np array and a gekko array works
    
    def h_inter_range_gekko(self, i, sat_pos): # This function calculates the range measurement between the satellite and another satellite
        norm = self.m.sqrt((self.opt_state[i, 0] - sat_pos[0])**2 + (self.opt_state[i, 1] - sat_pos[1])**2 + (self.opt_state[i, 2] - sat_pos[2])**2)
        return norm


    def measure_z_landmark(self, landmarks: list) -> np.ndarray:
        # Determine the closest landmark to the satellite at the current state and calculate the bearing to that landmark
        min_landmark = closest_landmark(self.curr_pos, landmarks)

        if min_landmark is None:
            raise Exception("No landmarks found")
        
        vec = (self.curr_pos - min_landmark.pos) + np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(int(self.dim/2)))
        return vec/np.linalg.norm(self.curr_pos - min_landmark.pos)

    def measure_z_range(self, sats: list) -> np.ndarray:
        z = np.empty((0))
        for sat in sats:
            if sat.id != self.id:
                # TODO: Implement a check to see whether the earth is in the way. Right now we are just measuring the straight line distances, potentially through the earth.
                d = np.array([np.linalg.norm(self.curr_pos[0:3] - sat.curr_pos[0:3])]) + np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(1))
                z = np.append(z,d,axis=0)
        return z
    
    def measure_z_combined(self, landmarks: list, sats: list) -> np.ndarray:
        z = self.measure_z_landmark(landmarks) # Take bearing measurement to the closest landmark
        if len(sats) > 1:
            z = np.append(z, self.measure_z_range(sats),axis=0) # Append the range measurement

        return z


# Helper function to determine closest landmark to the satellite at the current state position based on Euclidean distance
def closest_landmark(pos, landmarks: list) -> object:
    """
    Find the closest landmark to the current position.

    Args:
        pos (np.ndarray): The current position of the satellite.
        landmarks (list): A list of landmark objects, each having a 'pos' attribute.

    Returns:
        Landmark: The landmark object that is closest to the current position. 
        If no landmarks are provided, returns None.
    """
    min_dist = np.inf
    closest_landmark = None

    for landmark in landmarks:
        dist = np.linalg.norm(np.array(pos) - landmark.pos)
        if dist < min_dist:
            min_dist = dist
            closest_landmark = landmark

    return closest_landmark


# Numerical solution of the dynamics using Euler's method
def euler_discretization(x: np.ndarray, dt: float) -> np.ndarray:
    r = x[0:3]
    v = x[3:6]
    r_new = r + v*dt
    v_new = v + (-MU/(np.linalg.norm(r)**3))*r*dt
    return np.concatenate((r_new, v_new))

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


# Parameters
N = 2
n_sats = 2
f = 1 # Hz
dt = 1/f
meas_dim = n_sats-1 + 3 # 1 bearing measurements that have dimension 3 and n_sats-1 range measurements each with dimension 1 (no measurement to yourself)
R_weight = 0.25
R = R_weight*np.eye(meas_dim)
state_dim = 6
bearing_dim = 3


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

### Satellite Initialization ###
x00 = np.array([R_SAT, 0, 0, 0, 0, np.sqrt(MU/R_SAT)]) # Initial state of satellite. For now assume a circular orbit
x01 = np.array([0, R_SAT, 0, 0, 0, np.sqrt(MU/R_SAT)]) # Initial state of satellite. For now assume a circular orbit

sat0 = satellite(
    state=x00,
    rob_cov_init=0.25,
    robot_id=0,
    dim=6,
    meas_dim=meas_dim,
    R_weight=R_weight,
    Q_weight=0.25,
    N=N,
    n_sats=n_sats,
    landmarks=landmark_objects
)

sat1 = satellite(
    state=x01,
    rob_cov_init=0.25,
    robot_id=1,
    dim=6,
    meas_dim=meas_dim,
    R_weight=R_weight,
    Q_weight=0.25,
    N=N,
    n_sats=n_sats,
    landmarks=landmark_objects

)

sats = [sat0, sat1]



'''
TODO: Scale this to multiple satellites and landmarks
'''

# t = np.linspace(0, N*dt, N) # Time vector
x_traj = np.zeros((N, 6, n_sats)) # Discretized trajectory of satellite states over time period 

for sat in sats:
    x = sat.x_0
    for i in range(N):
        x = rk4_discretization(x,dt) # RK4 discretization of satellite state dynamics
        x_traj[i,:,sat.id] = x  # Store the trajectory of the satellite states

y = np.zeros((N, meas_dim, n_sats)) # Measurements for each satellite over the time period

for i in range(N):

    for sat in sats:
        sat.curr_pos = x_traj[i,:,sat.id][0:3] # Update the current position of all satellites for the current measurement batch

    for sat in sats:
        y[i,:,sat.id] = sat.measure_z_combined(landmark_objects, sats) # Generate the faked measurements for each satellite

        for j in range(bearing_dim):
            # print(sat.y_est_landmark[i][j])
            # sat.y_est_landmark[i][j] = sat.m.Intermediate(sat.h_landmark_gekko(sat.opt_state, i,landmark_objects)[j]) # Generate bearing measurements element by element for each satellite
            # sat.m.Equation(sat.y_est_landmark[i][j] == sat.h_landmark_gekko(sat.opt_state, i,landmark_objects)[j]) # Generate bearing measurements element by element for each satellite
            sat.y_est[i][j] = sat.m.Intermediate(sat.h_landmark_gekko(sat.opt_state, i,landmark_objects)[j]) # Generate bearing measurements element by element for each satellite

        k = 0
        for other_sat in sats:
            if other_sat.id != sat.id:
                # sat.y_est_inter_range[i][k] = sat.m.Intermediate(sat.h_inter_range_gekko(i, other_sat.curr_pos)) # Generate range measurements element by element for each satellite
                # sat.m.Equation(sat.y_est_inter_range[i][k] == sat.h_inter_range_gekko(i, other_sat.curr_pos))
                sat.y_est[i][k+bearing_dim] = sat.m.Intermediate(sat.h_inter_range_gekko(i, other_sat.curr_pos)) 
                k += 1

# We can now use GEKKO to solve the nonlinear least squares problem

for sat in sats:

    print(f"SOLVING FOR SATELLITE {sat.id}")
    # m = GEKKO(remote=False)

    for i in range(N):
        sat.m.Equation(sat.opt_state[i,0].dt() == sat.opt_state[i,3])
        sat.m.Equation(sat.opt_state[i,1].dt() == sat.opt_state[i,4])
        sat.m.Equation(sat.opt_state[i,2].dt() == sat.opt_state[i,5])
        sat.m.Equation(sat.opt_state[i,3].dt() == -MU*sat.opt_state[i,0]/(sat.opt_state[i,0]**2 + sat.opt_state[i,1]**2 + sat.opt_state[i,2]**2)**1.5)
        sat.m.Equation(sat.opt_state[i,4].dt() == -MU*sat.opt_state[i,1]/(sat.opt_state[i,0]**2 + sat.opt_state[i,1]**2 + sat.opt_state[i,2]**2)**1.5)
        sat.m.Equation(sat.opt_state[i,5].dt() == -MU*sat.opt_state[i,2]/(sat.opt_state[i,0]**2 + sat.opt_state[i,1]**2 + sat.opt_state[i,2]**2)**1.5)
    
        # Set initial conditions
        sat.m.Equation(sat.opt_state[i,0] == x_traj[i,0,sat.id])
        sat.m.Equation(sat.opt_state[i,1] == x_traj[i,1,sat.id])
        sat.m.Equation(sat.opt_state[i,2] == x_traj[i,2,sat.id])
        sat.m.Equation(sat.opt_state[i,3] == x_traj[i,3,sat.id])
        sat.m.Equation(sat.opt_state[i,4] == x_traj[i,4,sat.id])
        sat.m.Equation(sat.opt_state[i,5] == x_traj[i,5,sat.id])
    
    objective = 0
    for i in range(N):
        # for j in range(bearing_dim):
        #     diff = y[i,j,sat.id] - sat.y_est_landmark[i][j]
        #     objective += (diff*diff)*(1/R[j,j])

        # for j in range(n_sats-1):
        #     diff = y[i,j+bearing_dim,sat.id] - sat.y_est_inter_range[i][j]
        #     objective += (diff * diff)*(1/R[j+bearing_dim,j+bearing_dim])


        diff = (y[i,:,sat.id] - sat.y_est[i][:])
        objective += (diff.T @ np.linalg.inv(R) @ diff)

    sat.m.Minimize(objective)

    # sat.m.Minimize(sum((y[i,:,sat.id] - sat.y_est[i,:]).T@np.linalg.inv(R)@(y[i,:,sat.id] - sat.y_est[i,:]) for i in range(N)))
    sat.m.solve(disp=True)

    # print("r_x:", r_x.value)
    # print("r_y:", r_y.value)
    # print("r_z:", r_z.value)
    # print("v_x:", v_x.value)
    # print("v_y:", v_y.value)
    # print("v_z:", v_z.value)

    print("done")
    

# The earth landmarks are in ECEF coordinates. We will make the assumption for now that only the closest landmark is used for the bearing measurement.
# So calculate distance to all landmarks and then choose the closest one and calculate bearing to that.

