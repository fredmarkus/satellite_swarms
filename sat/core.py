from typing import List

import jax.numpy as jnp
import jax
import math
import numpy as np

import brahe
from brahe.epoch import Epoch

from gnc_payload.orbit_determination.landmark_bearing_sensors import GroundTruthLandmarkBearingSensor
from gnc_payload.orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from landmarks.landmark import landmark
from utils.math_utils import R_X
from utils.math_utils import R_Z
from utils.math_utils import az_el_to_vector
from utils.math_utils import vector_to_az_el
from utils.ubi_config_utils import load_config

from gnc_payload.utils.earth_utils import get_nadir_rotation

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
                 R_weight_range: float,
                 R_weight_land_bearing: float, 
                 R_weight_sat_bearing: float,
                 N: int, 
                 n_sats: int, 
                 landmarks: object,
                 orbital_elements: dict,
                 camera_exists: bool,
                 camera_fov: float,
                 verbose: bool,
                 ignore_earth: bool,
                 meas_type: list) -> None:
        
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
        self.R_weight_range = float(R_weight_range)
        self.R_weight_land_bearing = float(R_weight_land_bearing)
        self.R_weight_sat_bearing = float(R_weight_sat_bearing)
        self.n_sats = n_sats # Number of satellites
        self.landmarks = landmarks
        self.camera_exists = camera_exists
        self.camera_fov = camera_fov
        self.verbose = verbose
        self.ignore_earth = ignore_earth
        
        self.land_bearing_dim = 0
        self.sat_bearing_dim = 0
        self.range_dim = 0

        # Initialize the measurement vector with noise
        np.random.seed(123)
        self.x_m = self.x_0 # Initialize the measurement vector exactly the same as the initial state vector
        pos_init_noise = np.random.normal(loc=0,scale=math.sqrt(10),size=int(self.dim/2))
        vel_init_noise = np.random.normal(loc=0,scale=math.sqrt(1e-4),size=int(self.dim/2))
        # # Add the noise to the initial state vector
        self.x_m = self.x_m + np.append(pos_init_noise,vel_init_noise,axis=0) 

        self.x_p = self.x_m
        # Initialize the prior covariance the same as the measurement covariance
        self.cov_p = self.cov_m

        #Determines the current position of the satellite (Necessary for landmark bearing and satellite ranging)
        self.curr_pos = self.x_0[0:3]

        # Provide the position of the other satellites for all N timesteps
        self.other_sats_pos = np.zeros((N+1, 3, int(n_sats-1)))
        self.curr_visible_landmarks = []
        self.meas_type = meas_type
        self.HEIGHT = 550

        config = load_config()
        config["solver"]["world_update_rate"] = 1 / 60  # Hz
        config["mission"]["duration"] = 3 * 90 * 60  # s, roughly 1 orbit

        dt = 1 / config["solver"]["world_update_rate"]
        starting_epoch = Epoch(*brahe.time.mjd_to_caldate(config["mission"]["start_date"]))
        self.data_manager = ODSimulationDataManager(starting_epoch, dt)
        self.landmark_bearing_sensor = GroundTruthLandmarkBearingSensor(config=config)

        self.data_manager.push_next_state(
            np.expand_dims(self.x_0, axis=0), np.expand_dims(get_nadir_rotation(self.x_0), axis=0)
        )

    ### Visibility functions for landmarks and satellites ###

    def is_visible_ellipse(self, own_pos, other_pos) -> bool:
        # Check if the earth is in the way of the own position and the other position
        d = other_pos - own_pos
        A = (d[0]**2 + d[1]**2)/(EQ_RADIUS**2) + (d[2]**2)/(POLAR_RADIUS**2)
        B = 2*(own_pos[0]*d[0] + own_pos[1]*d[1])/(EQ_RADIUS**2) + 2*own_pos[2]*d[2]/(POLAR_RADIUS**2)
        C = (own_pos[0]**2 + own_pos[1]**2)/(EQ_RADIUS**2) + (own_pos[2]**2)/(POLAR_RADIUS**2)
        
        # Calculate the discriminant
        discriminant = B**2 - 4*A*(C-1)
        if discriminant < 0:
            # Solution does not intersect the earth as no real solutions exist
            return True
        
        # Discriminant is positive, calculate the solutions
        solution1 = (-B + jnp.sqrt(discriminant))/(2*A)
        solution2 = (-B - jnp.sqrt(discriminant))/(2*A)
        if ((solution1 > 0 or solution2 > 0) and (solution1 < 1 or solution2 < 1)):
            #One of the solutions is positive and less than 1, the earth is in the way
            return False
        
        return True

    def visible_landmarks_list(self) -> List[landmark]:
        # TODO: can be made faster by inferring that landmarks are more likely to be visible if they were visible in the previous timestep
        self.curr_visible_landmarks = []
        for landmark in self.landmarks:
            if self.is_visible_ellipse(self.curr_pos, landmark.pos): # TODO: Consider ignoring the earth or not the speed up of ignoring is crazy.
                self.curr_visible_landmarks.append(landmark)

        return self.curr_visible_landmarks
    
    def visible_sats_list(self, sats: List["satellite"]) -> List["satellite"]:
        self.curr_visible_sats = []
        for sat in sats:
            if sat.id != self.id:
                if self.ignore_earth or self.is_visible_ellipse(self.curr_pos, sat.curr_pos):
                    self.curr_visible_sats.append(sat)
        return self.curr_visible_sats

    def h_landmark(self, x):
        h = jnp.zeros((len(self.data_manager.curr_landmarks)*3))

        if self.camera_exists:
            for i, landmark in enumerate(self.curr_visible_landmarks):
                norm = jnp.linalg.norm(landmark.pos - x[0:3])
                h = h.at[i*3:i*3+3].set((landmark.pos - x[0:3])/norm)
        return h

    def H_landmark(self, x):
        jac = jax.jacobian(self.h_landmark)(x)
        return jac

    def h_inter_range(self, x):
        h = jnp.zeros((len(self.curr_visible_sats)))
        for i, sat in enumerate(self.curr_visible_sats):
            norm = jnp.linalg.norm(x[0:3] - sat.x_p[0:3])
            h= h.at[i].set(norm)
        
        return h

    def H_inter_range(self, x):
        jac = jax.jacobian(self.h_inter_range)(x)
        return jac
    
    def h_sat_bearing(self, x):
        h = jnp.zeros((len(self.curr_visible_sats)*3))
        for i, sat in enumerate(self.curr_visible_sats):
            h = h.at[i*3:i*3+3].set((x[0:3] - sat.x_p[0:3])/jnp.linalg.norm(x[0:3] - sat.x_p[0:3]))

        return h
        
    def H_sat_bearing(self, x):
        jac = jax.jacobian(self.h_sat_bearing)(x)
        return jac

    ### Measurement functions for landmarks and satellites ###

    def measure_z_range(self) -> np.ndarray:
        z = np.zeros((len(self.curr_visible_sats)))
        for i, sat in enumerate(self.curr_visible_sats):
            if self.verbose and ("range" in self.meas_type):
                print(f"Satellite {self.id} can take range measurement to satellite {sat.id}")
            
            noise = np.random.normal(loc=0,scale=math.sqrt(self.R_weight_range),size=(1))
            z[i] = np.linalg.norm(self.curr_pos - sat.curr_pos) + noise
            
        return z
    

    def measure_z_landmark(self) -> np.ndarray:
        z_l = np.zeros((len(self.curr_visible_landmarks)*3))
        if self.camera_exists:
            for i, landmark in enumerate(self.curr_visible_landmarks):
                if self.verbose and ("land" in self.meas_type):
                    print(f"Satellite {self.id} can see landmark {landmark.name}")
                vec = landmark.pos - self.curr_pos
                vec = vec/np.linalg.norm(vec)
                az, el = vector_to_az_el(vec)
                az = az + np.random.normal(loc=0,scale=math.sqrt(0.001),size=1)
                el = el + np.random.normal(loc=0,scale=math.sqrt(0.001),size=1)
                vec = az_el_to_vector(az, el)
                z_l[i*3:i*3+3] = vec

        return z_l
    
    def measure_z_sat_bearing(self) -> np.ndarray:
        z = np.zeros((len(self.curr_visible_sats)*3))
        if self.camera_exists:
            for i, sat in enumerate(self.curr_visible_sats):
                if self.verbose and ("sat_bearing" in self.meas_type):
                    print(f"Satellite {self.id} can take bearing measurement to satellite {sat.id}")
                vec = self.curr_pos - sat.curr_pos
                vec = vec/np.linalg.norm(vec)
                az, el = vector_to_az_el(vec)
                az = az + np.random.normal(loc=0,scale=math.sqrt(0.001),size=1)
                el = el + np.random.normal(loc=0,scale=math.sqrt(0.001),size=1)
                vec = az_el_to_vector(az, el)
                z[i*3:i*3+3] = vec

        return z

    ## SIMPLIFIED INTERCEPTOR FOR ROUND EARTH ASSUMPTION ##
    # def landmark_visible(self, landmark_pos, r_earth, theta_t) -> bool:
    #     # Check if a landmark can be seen from the current position of the satellite when taking a picture
    #     # Assuming that the camera is pointing straight down
    #     # FOV is about 60 degrees 
    #     # ASSUMPTION: The satellite is flying at an altitude of around 550km consistently
    #     # Otherwise we need a separate way to determine the altitude of the satellite 
        
    #     theta_l = jnp.rad2deg(jnp.arccos(jnp.dot(r_earth,landmark_pos)/(jnp.linalg.norm(r_earth)*jnp.linalg.norm(landmark_pos))))
        
    #     if theta_l < theta_t:
    #         return True
    #     else:
    #         return False
