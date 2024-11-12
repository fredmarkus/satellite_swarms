# This code considers the non-linear least square implementation of the problem described before
# We now consider the dynamics  of an actual satellite flying in a polar orbit.

#Simulate a polar orbit with a satellite that has states r and v where both are 3 dimensional vectors
# Dynamics are given by: 
# r_dot = v
# v_dot = a = MU * r / ||r||^3
# where r is the position vector, v is the velocity vector, a is the acceleration, and MU is the gravitational parameter

# Satellite flies at an altitude of ca. 550km above the earth's surface
# The earth's radius is 6378km and the gravitational parameter is 3.986004418 x 10^5 km^3/s^2

from gekko import GEKKO
import numpy as np
import math
import csv

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

    def __init__(self, state: np.ndarray, rob_cov_init: float, robot_id: int, dim: int, meas_dim: int, Q_weight: float, R_weight: float) -> None:
        """
        Initializes the state and parameters for the non-linear least squares estimator.

        Args:
            state (np.ndarray): Initial state vector of the satellite.
            rob_cov_init (float): Initial covariance value for the robot.
            robot_id (int): Unique identifier for the satellite.
            dim (int): State dimension of the satellite (e.g., 6 for 3 position + 3 velocity).
            meas_dim (int): Dimension of the measurement vector.
            Q_weight (float): Variance weight for the process noise.
            R_weight (float): Variance weight for the measurement noise.

        Attributes:
            x_0 (np.ndarray): Initial state vector.
            cov_m (np.ndarray): Initial covariance matrix.
            id (int): Unique identifier for the satellite.
            dim (int): State dimension of the satellite.
            meas_dim (int): Dimension of the measurement vector.
            R_weight (float): Variance weight for the measurement noise.
            Q_weight (float): Variance weight for the process noise.
            info_matrix (np.ndarray): Information matrix, inverse of the covariance matrix.
            info_vector (np.ndarray): Information vector, product of the information matrix and initial state vector.
            x_p (np.ndarray): Prior state vector, initialized to the initial state vector.
            cov_p (np.ndarray): Prior covariance matrix, initialized to the initial covariance matrix.
            curr_pos (np.ndarray): Current position of the satellite.
            min_landmark (None): Placeholder for the closest current landmark.

        Returns: 
            None
        """
        self.x_0 = state
        self.cov_m = rob_cov_init*np.eye(dim)
        self.id = robot_id # Unique identifier for the satellite
        self.dim = dim # State dimension of the satellite (currently 3 position + 3 velocity)
        self.meas_dim = meas_dim
        self.R_weight = R_weight # This is the variance weight for the measurement noise
        self.Q_weight = Q_weight # This is the variance weight for the process noise

        self.info_matrix = np.linalg.inv(self.cov_m)
        self.info_vector = self.info_matrix@self.x_0
        self.x_p = self.x_0 # Initialize the prior vector exactly the same as the measurement vector
        self.cov_p = self.cov_m # Initialize the prior covariance exactly the same as the measurement covariance

        self.curr_pos = self.x_0[0:3] #Determines the current position of the satellite (Necessary for landmark bearing and satellite ranging)
        self.min_landmark = None # The closest current landmark

    # Helper function to determine closest landmark to the satellite at the current state position based on Euclidean distance
    def closest_landmark(self, landmarks: list) -> object:
        """
        Find the closest landmark to the current position.

        Args:
            landmarks (list): A list of landmark objects, each having a 'pos' attribute.

        Returns:
            Landmark: The landmark object that is closest to the current position. 
            If no landmarks are provided, returns None.
        """
        min_dist = np.inf
        closest_landmark = None

        for landmark in landmarks:
            dist = np.linalg.norm(self.curr_pos - landmark.pos)
            if dist < min_dist:
                min_dist = dist
                closest_landmark = landmark
                
        if closest_landmark is not None:
            self.min_landmark = closest_landmark
        
        return closest_landmark

    def h_landmark(self, landmarks: list) -> np.ndarray:
        """
        Calculate the normalized vector (bearing) from the current position to the closest landmark.

        Args:
            landmarks (list): A list of landmark objects, each having a 'pos' attribute representing its position.

        Returns:
            numpy.ndarray: A normalized vector pointing from the current position to the closest landmark.
        """
        min_landmark = self.closest_landmark(landmarks)
        return (self.curr_pos - min_landmark.pos)/np.linalg.norm(self.curr_pos - min_landmark.pos) # This provides a normalized vector (the bearing)

    def H_landmark(self, landmark: object) -> np.ndarray:
        """
        Calculate the Hessian matrix of the landmark.

        Args:
            landmark (object): The landmark object which contains the position attribute 'pos'.

        Returns:
            numpy.ndarray: The Hessian matrix of the landmark with respect to the current position.
        """
        return -(self.curr_pos - landmark.pos)@(self.curr_pos - landmark.pos).T/np.linalg.norm(self.curr_pos - landmark.pos)**3 + np.eye(self.dim/2)/np.linalg.norm(self.curr_pos - landmark.pos)
    
    
    def h_inter_range(self, sat_pos: np.ndarray) -> np.ndarray:
        """
        Calculate the Euclidean distance between the current position and a satellite position.

        Args:
            sat_pos (numpy.ndarray): The position of the satellite as a numpy array.

        Returns:
            numpy.ndarray: A numpy array containing the Euclidean distance.
        """
        return np.array([np.linalg.norm(self.curr_pos - sat_pos)])
    
    def H_inter_range(self, sat_pos: np.ndarray) -> np.ndarray:
        """
        Calculate the partial derivatives of the range with respect to the satellite position.

        This function computes the partial derivatives of the range (distance) between the current position
        and the satellite position with respect to the satellite's x, y, and z coordinates.

        Args:
            sat_pos (numpy.ndarray): A 3-element array representing the satellite's position [x, y, z].

        Returns:
            numpy.ndarray: A 1x3 array containing the partial derivatives [dx, dy, dz].
        """
        dx = (sat_pos[0] - self.curr_pos[0])/np.linalg.norm(self.curr_pos[0:3] - sat_pos)
        dy = (sat_pos[1] - self.curr_pos[1])/np.linalg.norm(self.curr_pos[0:3] - sat_pos)
        dz = (sat_pos[2] - self.curr_pos[2])/np.linalg.norm(self.curr_pos[0:3] - sat_pos)
        return np.array([[dx, dy, dz]])
    
    def h_combined(self, landmarks: list, sats: list) -> np.ndarray:
        """
        Compute the combined measurement vector for landmarks and satellites.

        This function calculates the measurement vector `h` by first computing the 
        landmark measurements using `self.h_landmark(landmarks)`. It then iterates 
        through the provided satellites (`sats`) and appends the inter-satellite 
        range measurements to `h` for each satellite that does not have the same 
        ID as `self.id`.

        Args:
            landmarks (list): A list of landmark positions.
            sats (list): A list of satellite objects, each with attributes `id` and `curr_pos`.

        Returns:
            numpy.ndarray: The combined measurement vector `h` for the given landmarks 
            and satellites.
        """
        h = self.h_landmark(landmarks)
        for sat in sats:
            if sat.id != self.id:
                h = np.append(h, self.h_inter_range(sat.curr_pos),axis=0)

        return h

    # This function combines the H matrices for the bearing and inter-satellite range measurements
    def combined_H(self, landmark: object, sats: list) -> np.ndarray:
        """
        Computes the combined Jacobian matrix H for a given landmark and a list of satellites.

        Args:
            landmark (object): The landmark object for which the Jacobian is computed.
            sats (list): A list of satellite objects.

        Returns:
            numpy.ndarray: The combined Jacobian matrix H.
        """
        H = self.H_landmark(landmark)
        for sat in sats:
            if sat.id != self.id:
                H = np.vstack((H, self.H_inter_range(sat.curr_pos)))

        return H
    

    def measure_z_landmark(self, landmarks: list) -> np.ndarray:
        """
        Determine the closest landmark to the satellite at the current state and calculate the bearing to that landmark.

        Args:
            landmarks (list): A list of landmark objects, each having a 'pos' attribute representing its position.
        Returns:
            numpy.ndarray: A vector representing the bearing to the closest landmark, with added noise.
        Raises:
            Exception: If no landmarks are found.
        """
        # Determine the closest landmark to the satellite at the current state and calculate the bearing to that landmark
        min_landmark = self.closest_landmark(landmarks)

        if min_landmark is None:
            raise Exception("No landmarks found")
        
        vec = (self.curr_pos - min_landmark.pos) + np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(self.dim/2))
        return vec/np.linalg.norm(self.curr_pos - min_landmark.pos)

    def measure_z_range(self, sats: list) -> np.ndarray:
        """
        Measures the range (distance) to a list of satellites.

        For each satellite in the provided list, this method calculates the 
        Euclidean distance from the current position of this object to the 
        satellite's current position. It adds some Gaussian noise to the 
        distance measurement to simulate real-world inaccuracies. The method 
        excludes the satellite with the same ID as this object.

        Args:
            sats (list): A list of satellite objects, each having 'id' and 
                         'curr_pos' attributes.

        Returns:
            numpy.ndarray: An array of measured distances to the satellites, 
                           with added Gaussian noise.
        """
        z = np.empty((0))
        for sat in sats:
            if sat.id != self.id:
                # TODO: Implement a check to see whether the earth is in the way. Right now we are just measuring the straight line distances, potentially through the earth.
                d = np.array([np.linalg.norm(self.curr_pos[0:3] - sat.curr_pos[0:3])]) + np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(1))
                z = np.append(z,d,axis=0)
        return z
    
    def measure_z_combined(self, landmarks: list, sats: list) -> np.ndarray:
        """
        Measure the combined observation vector from landmarks and satellites.

        This method first takes a bearing measurement to the closest landmark.
        If there are multiple satellites, it appends the range measurements from the satellites
        to the observation vector.

        Args:
            landmarks (list): A list of landmarks to measure.
            sats (list): A list of satellites to measure.

        Returns:
            np.ndarray: The combined observation vector containing bearing measurements to the closest landmark
                        and range measurements from the satellites.
        """
        z = self.measure_z_landmark(landmarks) # Take bearing measurement to the closest landmark
        if len(sats) > 1:
            z = np.append(z, self.measure_z_range(sats),axis=0) # Append the range measurement

        return z


# Numerical solution of the dynamics using Euler's method TODO: Implement RK4
def euler_discretization(x: np.ndarray, dt: float) -> np.ndarray:
    r = x[0:3]
    v = x[3:6]
    r_new = r + v*dt
    v_new = v + (-MU/(np.linalg.norm(r)**3))*r*dt
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
N = 1000
n_sats = 2
f = 1 # Hz
dt = 1/f
meas_dim = n_sats + 3 # 1 bearing measurement that has dimension 3 and n_sats range measurements each with dimension 1


### Satellite Initialization ###
x00 = np.array([R_SAT, 0, 0, 0, 0, np.sqrt(MU/R_SAT)]) # Initial state of satellite. For now assume a circular orbit
x01 = np.array([0, R_SAT, 0, 0, 0, np.sqrt(MU/R_SAT)]) # Initial state of satellite. For now assume a circular orbit

sat0 = satellite(
    state=x00,
    rob_cov_init=0.25,
    robot_id=0,
    dim=6,
    meas_dim=meas_dim,
    R_weight=0.25,
    Q_weight=0.25
)

sat1 = satellite(
    state=x01,
    rob_cov_init=0.25,
    robot_id=1,
    dim=6,
    meas_dim=meas_dim,
    R_weight=0.25,
    Q_weight=0.25
)

sats = [sat0, sat1]


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


'''
TODO: Place everything in the GEKKO solver to get nonlinear least squares solution for states. Consider the best version of the problem to use. (Consider the numerically advantageous option that uses the Cholesky decomposition.)
TODO: Scale this to multiple satellites and landmarks
'''

t = np.linspace(0, N*dt, N) # Time vector
x_traj = np.zeros((N, 6, n_sats)) # Discretized trajectory of satellite states over time period 

for sat in sats:
    x = sat.x_0
    for i in range(N):
        x = euler_discretization(x, dt) # Euler discretization of satellite state dynamics
        x_traj[i,:,sat.id] = x  # Store the trajectory of the satellite states

y = np.zeros((N, meas_dim, n_sats)) # Measurements for each satellite over the time period
y_est = np.zeros((N, meas_dim, n_sats)) # Estimated measurements for each satellite over the time period

for i in range(N):

    for sat in sats:
        sat.curr_pos = x_traj[i,:,sat.id][0:3] # Update the current position of all satellites for the current measurement

    for sat in sats:
        y[i,:,sat.id] = sat.measure_z_combined(landmark_objects, sats) # Generate the faked measurements for each satellite
        y_est[i,:,sat.id] = sat.h_combined(landmark_objects, sats) # Generate the estimated measurements for each satellite

# Now we have the measurements and the estimated measurements for each satellite over the time period
# We can now use GEKKO to solve the nonlinear least squares problem




# The earth landmarks are in ECEF coordinates. We will make the assumption for now that only the closest landmark is used for the bearing measurement.
# So calculate distance to all landmarks and then choose the closest one and calculate bearing to that.

