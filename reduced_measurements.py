from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

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

        self.info_matrix = np.linalg.inv(self.cov_m)
        self.info_vector = self.info_matrix@self.x_0
        self.x_p = self.x_0 # Initialize the prior vector exactly the same as the measurement vector
        self.cov_p = self.cov_m # Initialize the prior covariance exactly the same as the measurement covariance

        self.curr_pos = self.x_0[0:3] #Determines the current position of the satellite (Necessary for landmark bearing and satellite ranging)
        self.other_sats_pos = np.ndarray(shape=(N,3,n_sats-1)) # Provides the position of the other satellites for all N timesteps

        # Gekko optimization variables
        # self.m = GEKKO(remote=False)
        # self.m.time = np.linspace(0, N, N) # Time vector
        # self.m.options.IMODE = 5
        # self.opt_state = self.m.Array(self.m.Var, (N, dim)) # State vector to be optimized
        # self.y_est = [[None]*meas_dim]*N
    
    def h_inter_range_gekko(self, i, sat_pos): # This function calculates the range measurement between the satellite and another satellite
        norm = self.m.sqrt((self.opt_state[i, 0] - sat_pos[0])**2 + (self.opt_state[i, 1] - sat_pos[1])**2 + (self.opt_state[i, 2] - sat_pos[2])**2)
        return norm

    def measure_z_range(self, sats: list) -> np.ndarray:
        z = np.empty((0))
        for sat in sats:
            if sat.id != self.id:
                # TODO: Implement a check to see whether the earth is in the way. Right now we are just measuring the straight line distances, potentially through the earth.
                d = np.array([np.linalg.norm(self.curr_pos - sat.curr_pos)]) + np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(1))
                z = np.append(z,d,axis=0)
        return z
    

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
N = 200
f = 1 #Hz
dt = 1/f
n_sats = 3
R_weight = 0.25
meas_dim = n_sats-1 # + bearing_dim
R = np.eye(meas_dim)*R_weight
state_dim = 6


x_traj = np.zeros((N, 6, n_sats)) # Discretized trajectory of satellite states over time period 

### Satellite Initialization ###
x00 = np.array([R_SAT, 0, 0, 0, 0, np.sqrt(MU/R_SAT)]) # Initial state of satellite. For now assume a circular orbit
x01 = np.array([0, R_SAT, 0, 0, 0, np.sqrt(MU/R_SAT)]) # Initial state of satellite. For now assume a circular orbit
x02 = np.array([-R_SAT, 0, 0, 0, 0, np.sqrt(MU/R_SAT)]) # Initial state of satellite. For now assume a circular orbit

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

sat2 = satellite(
    state=x02,
    rob_cov_init=0.25,
    robot_id=2,
    dim=6,
    meas_dim=meas_dim,
    R_weight=R_weight,
    Q_weight=0.25,
    N=N,
    n_sats=n_sats,
    landmarks=landmark_objects
)

sats = [sat0, sat1, sat2]

# Generate synthetic data for exponential decay in three dimensions with noise
np.random.seed(0)        # For reproducibility
time_data = np.linspace(0, N, N*f)  # Time points
#

for sat in sats: 
    x = sat.x_0
    for i in range(N):
        x = rk4_discretization(x, dt)
        x_traj[i,:,sat.id] = x



# Generate synthetic data for satellite ranging
y_m = np.zeros((N,meas_dim,n_sats))
for i in range(N):

    for sat in sats:
        sat.curr_pos = x_traj[i,0:3,sat.id]

    for sat in sats:
        y_m[i,:,sat.id] = sat.measure_z_range(sats) # This sets the measurement

for sat in sats:
    sat_i = 0 #iterator variable
    for other_sat in sats:
        if sat.id != other_sat.id:
            sat.other_sats_pos[:,:,sat_i] = x_traj[:,0:3,other_sat.id] # Transfer all N 3D positions of the other satellites from x_traj
            sat_i += 1
        


for sat in sats:
    # Initialize GEKKO model
    m = GEKKO(remote=False)
    m.time = time_data  # Define time vector in the model


    x1 = m.SV(value=x_traj[0,0,sat.id])  # Initial guess for x1
    x2 = m.SV(value=x_traj[0,1,sat.id])  # Initial guess for x2
    x3 = m.SV(value=x_traj[0,2,sat.id])  # Initial guess for x3
    v1 = m.SV(value=x_traj[0,3,sat.id])  # Initial guess for v1
    v2 = m.SV(value=x_traj[0,4,sat.id])  # Initial guess for v2
    v3 = m.SV(value=x_traj[0,5,sat.id])  # Initial guess for v3

    m.Equation(x1.dt() == v1)
    m.Equation(x2.dt() == v2)
    m.Equation(x3.dt() == v3)
    m.Equation(v1.dt() == -MU*x1/(x1**2 + x2**2 + x3**2)**(3/2))
    m.Equation(v2.dt() == -MU*x2/(x1**2 + x2**2 + x3**2)**(3/2))
    m.Equation(v3.dt() == -MU*x3/(x1**2 + x2**2 + x3**2)**(3/2))

    y = [None]*(meas_dim)
    for i in range(n_sats-1):
        xref = m.Param(value=sat.other_sats_pos[:, 0, i])  # x_ref for x1
        yref = m.Param(value=sat.other_sats_pos[:, 1, i])  # y_ref for x2
        zref = m.Param(value=sat.other_sats_pos[:, 2, i])  # z_ref for x3
        y[i] = m.Intermediate(m.sqrt((x1 - xref)**2 + (x2 - yref)**2 + (x3 - zref)**2))

    # # Create reference parameters in GEKKO (use slices if x_traj is time-dependent)
    # xref = m.Param(value=sat.other_sats_pos[:, 0, 0])  # x_ref for x1
    # yref = m.Param(value=sat.other_sats_pos[:, 1, 0])  # y_ref for x2
    # zref = m.Param(value=sat.other_sats_pos[:, 2, 0])  # z_ref for x3
    # y1 = m.Intermediate(m.sqrt((x1 - xref)**2 + (x2 - yref)**2 + (x3 - zref)**2))
    
    obj = 0
    for i in range(N):
        for j in range(meas_dim):
            tmp = y[j] - y_m[i,j,sat.id]
            obj += (y[j] - y_m[i,j,sat.id])**2
        # obj += (y1 - y_m[i,0,sat.id])**2

    m.Minimize(obj)

    m.options.IMODE = 5  # Dynamic estimation mode
    m.solve(disp=True)
    


    
    # Plot the results
    plt.figure()
    for m_dim in range(meas_dim):
        plt.subplot(2,1,m_dim+1)
        plt.plot(time_data, y_m[:,m_dim,sat.id], 'o', label=f'Noisy Data sat {m_dim} for satellite {sat.id}')
        plt.plot(time_data, y[m_dim].value, '-', label=f'Fitted Model sat {m_dim} for satellite {sat.id}')
        plt.xlabel('Time')
        plt.ylabel('y1')
        plt.legend()


plt.show()
