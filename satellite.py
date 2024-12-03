import jax.numpy as jnp
import jax
import math
import numpy as np


from utils.math_utils import R_X, R_Z

# Constants
MU = 3.986004418 * 10**5 # km^3/s^2 # Gravitational parameter of the Earth
MASS = 2 # kg # Mass of the satellite
AREA = 1e-9 # km^2 # Cross-sectional area of the satellite
EQ_RADIUS = 6378.1370 # km # Equatorial radius of the Earth
POLAR_RADIUS = 6356.7523 # km # Polar radius of the Earth

class satellite:

    def __init__(self, 
                 pos_cov_init: float, 
                 vel_cov_init: float, 
                 robot_id: int, 
                 dim: int, 
                 meas_dim: int, 
                 R_weight: float, 
                 N: int, 
                 n_sats: int, 
                 landmarks: object,
                 orbital_elements: dict,
                 camera_exists: bool,
                 camera_fov: float) -> None:
        
        # Calculate initial state based on orbital elements placing 
        a = float(orbital_elements["a"])
        e = float(orbital_elements["e"])
        i = np.radians(float(orbital_elements["i"]))
        omega = np.radians(float(orbital_elements["omega"]))
        omega_dot = np.radians(float(orbital_elements["omega_dot"]))
        M = np.radians(float(orbital_elements["M"]))
        
        # Position calculation
        r = a*(1-e**2)/(1+e*np.cos(M))
        perifocal = np.array([r*np.cos(M), r*np.sin(M), 0])
        Q = R_X(-omega_dot)@R_Z(-omega)@R_X(-i)
        pos_0 = Q@perifocal

        # Velocity calculation
        v_x = -math.sqrt(MU/a)*np.sin(M)/(1+e*np.cos(M))
        v_y = math.sqrt(MU/a)*(e + np.cos(M))/(1+e*np.cos(M))
        v_z = 0
        vel_0 = Q@np.array([v_x, v_y, v_z])
    
        self.x_0 = np.append(pos_0,vel_0) # Initial state vector of the satellite
        self.cov_m = np.diag(np.array([float(pos_cov_init), float(pos_cov_init), float(pos_cov_init), float(vel_cov_init), float(vel_cov_init), float(vel_cov_init)]))
        self.id = robot_id # Unique identifier for the satellite
        self.dim = dim # State dimension of the satellite (currently 3 position + 3 velocity)
        self.meas_dim = meas_dim
        self.R_weight = R_weight # This is the variance weight for the measurement noise
        self.N = N # Number of time steps
        self.n_sats = n_sats # Number of satellites
        self.landmarks = landmarks
        self.camera_exists = camera_exists
        self.camera_fov = camera_fov
        
        # Initialize the measurement vector with noise
        self.x_m = self.x_0# + np.array([1,0,0,0,0,0]) # Initialize the measurement vector exactly the same as the initial state vector
        x_m_init_noise = np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=int(self.dim/2))
        x_m_init_noise = x_m_init_noise/np.linalg.norm(x_m_init_noise) # Normalize the noise vector
        self.x_m = self.x_m + np.append(x_m_init_noise,np.zeros((3,)),axis=0) # Add the noise to the initial state vector

        self.x_p = self.x_m
        self.cov_p = self.cov_m # Initialize the prior covariance the same as the measurement covariance

        self.min_landmark_list = None # Initialize an array of the closest landmark to the satellite t
        self.curr_pos = self.x_0[0:3] #Determines the current position of the satellite (Necessary for landmark bearing and satellite ranging)
        self.other_sats_pos = np.zeros((N+1, 3, int(n_sats-1))) # Provides the position of the other satellites for all N timesteps
        # self.sats_visible = np.zeros((N,n_sats-1)) # Determines whether the other satellites are visible to this satellite


    def h_landmark(self, x):
        min_landmark = self.closest_landmark(x, self.landmarks)
        norm = jnp.sqrt((x[0] - min_landmark[0])**2 + (x[1] - min_landmark[1])**2 + (x[2] - min_landmark[2])**2)
        
        return (x[0:3] - min_landmark)/norm

    def H_landmark(self, x):
        # Use jax to autodifferentiate
        jac = jax.jacobian(self.h_landmark)(x)
        return jac

    def h_inter_range(self, i, j, x): # This function calculates the range measurement between the satellite and another satellite
        sat_id = j-3 # j is the range measurement index starting from 3
        sat_pos = self.other_sats_pos[i,:,sat_id]
        if self.is_visible_ellipse(sat_pos):
            return jnp.linalg.norm(x[0:3] - sat_pos) 
        else:
            return jnp.linalg.norm(0)

    def H_inter_range(self, i, j, x):
        jac = jax.jacobian(self.h_inter_range, argnums=2)(i, j, x)
        return jac

    def measure_z_range(self, sats: list) -> np.ndarray:
        z = np.empty((0))
        for sat in sats:
            if sat.id != self.id:

                if self.is_visible_ellipse(sat.curr_pos): # If the earth is not in the way, we can measure the range
                    noise = np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(1))
                    d = np.array([np.linalg.norm(self.curr_pos - sat.curr_pos)]) + noise
                    z = np.append(z,d,axis=0)
                    # self.sats_visible[index,sat.id] = 1

                else: # If the earth is in the way , we set the value to nan so it does not feature in the objective function
                   
                    z = np.append(z,np.array([0]),axis=0)
        return z
    
    def measure_z_landmark(self, landmarks: list) -> np.ndarray:
        # Determine the closest landmark to the satellite at the current state and calculate the bearing to that landmark
        if self.min_landmark_list is None:
            self.min_landmark_list = np.array([self.closest_landmark(self.curr_pos, landmarks)]).reshape(1,3)
        else:
            self.min_landmark_list = np.append(self.min_landmark_list,[self.closest_landmark(self.curr_pos, landmarks)],axis=0)
        
        noise = np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(int(self.dim/2)))
        vec = (self.curr_pos  - self.min_landmark_list[-1]) + noise
        return vec/np.linalg.norm(vec)

    def is_visible_ellipse(self, sat_pos) -> bool:
        # Check if the earth is in the way of the two satellites
        d = sat_pos - self.curr_pos
        A = (d[0]**2 + d[1]**2)/(EQ_RADIUS**2) + (d[2]**2)/(POLAR_RADIUS**2)
        B = 2*(self.curr_pos[0]*d[0] + self.curr_pos[1]*d[1])/(EQ_RADIUS**2) + 2*self.curr_pos[2]*d[2]/(POLAR_RADIUS**2)
        C = (self.curr_pos[0]**2 + self.curr_pos[1]**2)/(EQ_RADIUS**2) + (self.curr_pos[2]**2)/(POLAR_RADIUS**2)
        
        # Calculate the discriminant
        discriminant = B**2 - 4*A*(C-1)
        if discriminant < 0:
            # Solution does not intersect the earth as no real solutions exist
            return True
        
        # Discriminant is positive, calculate the solutions
        solution1 = (-B + math.sqrt(discriminant))/(2*A)
        solution2 = (-B - math.sqrt(discriminant))/(2*A)
        if ((solution1 > 0 or solution2 > 0) and (solution1 < 1 or solution2 < 1)):
            #One of the solutions is positive and less than 1, the earth is in the way
            return False
        
        return True
    
    def closest_landmark(self, pos, landmarks: list) -> object:
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
            dist = jnp.linalg.norm(pos[0:3] - landmark.pos)
            if dist < min_dist:
                min_dist = dist
                closest_landmark = landmark

        # print(f"Closest landmark to satellite at position {pos} is {closest_landmark.name} at position {closest_landmark.pos}")
        if closest_landmark is None:
            return np.array([0,0,0])
        return closest_landmark.pos
