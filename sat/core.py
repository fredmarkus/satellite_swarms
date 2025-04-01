from typing import List, Tuple

import jax.numpy as jnp
import jax
import math
import numpy as np

import brahe
from brahe.epoch import Epoch
import quaternion

from gnc_payload.orbit_determination.landmark_bearing_sensors import GroundTruthLandmarkBearingSensor
from gnc_payload.orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from gnc_payload.sensors.camera_model import CameraModelManager
from gnc_payload.utils.orbit_utils import get_sso_orbit_state
from landmarks.landmark import landmark
from my_utils.math_utils import az_el_to_vector
from my_utils.math_utils import vector_to_az_el
from my_utils.config_utils import load_config


from gnc_payload.dynamics.ekf_dynamics import EKFDynamics
from gnc_payload.utils.math_utils import left_q, Drp2q, G, rot_2_q, R
from gnc_payload.sensors.imu import IMU, IMUNoiseParams
from gnc_payload.sensors.sensor import SensorNoiseParams
from gnc_payload.sensors.bias import BiasParams
# Constants
MU = 3.986004418 * 10**5 # km^3/s^2 # Gravitational parameter of the Earth
MASS = 1 # kg # Mass of the satellite
AREA = 1e-7 # km^2 # Cross-sectional area of the satellite
EQ_RADIUS = 6378.1370 # km # Equatorial radius of the Earth
POLAR_RADIUS = 6356.7523 # km # Polar radius of the Earth


def imu_init(dt: float) -> IMU:
    """
    Initializes the IMU.

    :param dt: The time step for the simulation.

    :return: The initialized IMU.
    """
    # Initialize the IMU
    # bias params are min max range of bias and sigma_w
    # [units] and [(units/s)/sqrt(Hz)]
    bias_params_x = BiasParams.get_random_params([-1e-2, 1e-2], [1e-6, 1e-5])
    bias_params_y = BiasParams.get_random_params([-1e-2, 1e-2], [1e-6, 1e-5])
    bias_params_z = BiasParams.get_random_params([-1e-2, 1e-2], [1e-6, 1e-5])
    # bias_params = BiasParams.get_random_params([0, 0], [0, 0])
    # sigma_v [units/sqrt(Hz)] & scale_factor_error [-]
    sensor_noise_params_accel_x = SensorNoiseParams.get_random_params(
        bias_params_x, [0, 0.0], [0, 0.0]
    )
    sensor_noise_params_accel_y = SensorNoiseParams.get_random_params(
        bias_params_y, [0, 0.0], [0, 0.0]
    )
    sensor_noise_params_accel_z = SensorNoiseParams.get_random_params(
        bias_params_z, [0, 0.0], [0, 0.0]
    )
    sensor_noise_params_accel = [
        sensor_noise_params_accel_x,
        sensor_noise_params_accel_y,
        sensor_noise_params_accel_z,
    ]
    # sigma_v [units/sqrt(Hz)] & scale_factor_error [-]
    sensor_noise_params_gyro_x = SensorNoiseParams.get_random_params(
        bias_params_x, [1e-6, 1e-5], [0, 0.01]
    )
    sensor_noise_params_gyro_y = SensorNoiseParams.get_random_params(
        bias_params_y, [1e-6, 1e-5], [0, 0.01]
    )
    sensor_noise_params_gyro_z = SensorNoiseParams.get_random_params(
        bias_params_z, [1e-6, 1e-5], [0, 0.01]
    )
    sensor_noise_params_gyro = [
        sensor_noise_params_gyro_x,
        sensor_noise_params_gyro_y,
        sensor_noise_params_gyro_z,
    ]

    imu_noise_params = IMUNoiseParams(
        gyro_params=sensor_noise_params_gyro, accel_params=sensor_noise_params_accel
    )
    imu = IMU(
        dt=dt,
        IMU_noise_params=imu_noise_params,
        misalignment_range=[0, 0.01],
    )

    return imu


class satellite:

    def __init__(
            self, 
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
            meas_type: list,
            Q_noise: np.ndarray,
            freq: float,
            time_horizon: float,
            lat: float,
            lon: float,
            starting_epoch: Epoch,
            ua: np.ndarray = None
                 ) -> None:
        

        lon = lon - robot_id
        initial_state = get_sso_orbit_state(starting_epoch, lat, lon, 600e3, northwards=True)
        r_0 = initial_state[0:3] / 1000
        v_0 = initial_state[3:6] / 1000

        # Initial position, velocity vector of the satellite [m, m/s]
        self.id = robot_id # Unique identifier for the satellite
        # self.dim = dim # State dimension of the satellite (currently 3 position + 3 velocity)
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
        # # Add the noise to the initial state vector
        self.r_m = r_0 + np.random.normal(0, 1, 3)
        self.r_p = self.r_m

        self.v_m = v_0 + np.random.normal(0, 0.01, 3)
        self.v_p = self.v_m

        #Determines the current position of the satellite (Necessary for landmark bearing and satellite ranging)
        self.curr_pos = r_0

        # Provide the position of the other satellites for all N timesteps
        self.other_sats_pos = np.zeros((N+1, 3, int(n_sats-1)))
        self.curr_visible_landmarks = []
        self.meas_type = meas_type
        # self.HEIGHT = 550

        self.config = load_config()
        self.config["solver"]["world_update_rate"] = freq  # Hz
        self.config["mission"]["duration"] = time_horizon  # s, roughly 1 orbit

        self.dt = 1 / self.config["solver"]["world_update_rate"]
        starting_epoch = Epoch(*brahe.time.mjd_to_caldate(self.config["mission"]["start_date"]))
        self.data_manager = ODSimulationDataManager(starting_epoch, self.dt)
        self.landmark_bearing_sensor = GroundTruthLandmarkBearingSensor()
        self.camera_model_manager = CameraModelManager()

        init_rot = np.eye(3)
        noisy_rot = init_rot + np.random.normal(0, 1e-2, (3, 3))
        noisy_rot = noisy_rot @ np.linalg.inv(np.linalg.cholesky(noisy_rot.T @ noisy_rot))

        # Assert orthonormality
        assert np.allclose(noisy_rot @ noisy_rot.T, np.eye(3), atol=1e-3) and np.isclose(
            np.linalg.det(noisy_rot), 1
        ), "Rotation matrix is not a proper rotation matrix"

        # Ground truth states
        x_0 = np.concatenate([r_0, v_0])
        self.data_manager.push_next_state(x_0,init_rot)

        self.ua_scale = 10
        self.gyro_bias_scale = 2

        self.imu = imu_init(self.dt)

        self.ekf_dynamics = EKFDynamics(
            config=self.config,
            use_drag=False,
            use_j2=False,
            use_unmodelled_a=True,
            use_drag_scalar=True,
            use_j34=False,
            use_moon_grav=False,
            use_sun_grav=False,
            ua_scale=self.ua_scale
        )

        # Extra states
        self.w_b = (self.imu.get_bias()[0] + np.random.normal(0, 5e-5, 3)) * self.gyro_bias_scale
        self.q_m = quaternion.as_float_array(quaternion.from_rotation_matrix(noisy_rot))
        self.q_p = self.q_m
        self.ua = np.random.normal(0, 1e-8, 3) * self.ua_scale
        self.drag_est = np.array([1])

        self.Q_noise = Q_noise

        self.z1 = None
        self.measurement_camera_names = None
        self.day_time = False


    def predict(self, u: np.ndarray, epoch: Epoch = None) -> None:
        """
        Predict the next prior state. This corresponds to the prior update step in the EKF algorithm.
        Using Zac Manchester's formulation as defined in his inertial filter examples notebook
        https://github.com/RoboticExplorationLab/inertial-filter-examples

        :param u: IMU measurements consisting of angular velocity and linear acceleration with shape (6,)
        :param epoch: The epoch at which the prediction is made. If None is passed, no epoch is used.

        :return: None
        """

        # TODO: Use IMU measurements and update quaternion estimate

        w = u[0:3]  # angular velocity measurement from IMU

        x = np.concatenate([self.r_m, self.v_m, self.ua, self.drag_est])
        x_new = self.ekf_dynamics.perturbed_f(x=x, dt=self.dt, epoch=epoch)
        A_pos = self.ekf_dynamics.perturbed_f_jac(x=x, dt=self.dt, epoch=epoch)

        self.q_p = left_q(self.q_m) @ quaternion.as_float_array(
            quaternion.from_rotation_vector(self.dt * (w - self.w_b / self.gyro_bias_scale))
        )

        self.r_p = x_new[0:3]
        self.v_p = x_new[3:6]
        self.x_p = np.concatenate([self.r_p, self.v_p, self.ua, self.drag_est, quaternion.as_rotation_vector(quaternion.as_quat_array(self.q_p)), self.w_b])

        dqdq = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(-1 * self.dt * (w - self.w_b / self.gyro_bias_scale))
        )
        dqdw = (
            -1
            * self.dt
            * G(self.q_p).T
            @ left_q(self.q_m)
            @ Drp2q(self.dt * (w - self.w_b / self.gyro_bias_scale))
        )

        self.A = np.block(
            [
                [A_pos, np.zeros((10, 6))],
                [np.zeros((3, 10)), dqdq, dqdw],
                [np.zeros((3, 13)), np.eye(3)],
            ]
        )

        # self.cov_p = self.A @ self.cov_m @ self.A.T + self.Q_noise
    
    ### Visibility functions for landmarks and satellites ###
    def is_visible_ellipse(self, own_pos, other_pos) -> bool:
        # Check if the earth is in the way of the own position and the other position
        own_pos = own_pos
        other_pos = other_pos
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
            if self.is_visible_ellipse(self.data_manager.latest_state[0:3], landmark.pos): # TODO: Consider ignoring the earth or not the speed up of ignoring is crazy.
                self.curr_visible_landmarks.append(landmark)

        return self.curr_visible_landmarks
    
    def visible_sats_list(self, sats: List["satellite"]) -> List["satellite"]:
        self.curr_visible_sats = []
        for sat in sats:
            if sat.id != self.id:
                if self.ignore_earth or self.is_visible_ellipse(self.data_manager.latest_state[0:3], sat.data_manager.latest_state[0:3]):
                    self.curr_visible_sats.append(sat)
        return self.curr_visible_sats

    def h_landmark_actual(
        self,
        z: np.ndarray,
        camera_model_manager: CameraModelManager,
        measurement_camera_names: np.ndarray,
        x_p: jnp.ndarray,
        epoch: Epoch,
    ) -> jnp.ndarray:
        """
        Generate an estimate from measurements made. Using the known locations of the landmarks, we can provide
        a bearing estimate.

        :param z: Measurements of the landmarks in frame, consisting of just the ECI coordinates of the landmarks
        with shape (N, 3)
        :param camera_model_manager: The camera model manager used to manage the cameras.
        :param measurement_camera_names: Array of names of the cameras that took each measurement.
        :param x_p: Prior state estimate consisting of [position, velocity, rotation_vector] with shape (9,)
        :param epoch: The epoch at which the measurement is made. Epoch must be provided for ecef-eci transformation!

        :return: Estimate of the bearing vectors to all landmarks in the body frame with shape (N * 3, )
        """
        estimate = jnp.zeros((len(z) * 3))

        # Define rotation matrices
        # transform rotation_vector to rotation matrix via quaternion
        eci_R_body = R(rot_2_q(x_p[10:13]))
        ecef_R_eci = brahe.frames.rECItoECEF(epc=epoch)
        ecef_R_body = ecef_R_eci @ eci_R_body

        # Transform landmarks and position from ECI to ECEF
        landmarks_ecef = (ecef_R_eci @ z.T).T
        position_ecef = ecef_R_eci @ x_p[0:3]

        # Assert landmarks and measurement camera names are the same length
        assert landmarks_ecef.shape[0] == len(
            measurement_camera_names
        ), "Landmarks and measurement camera names must be the same length"

        # Calculate estimated bearing unit vectors in ECEF and transform to body frame
        for i, land_pos_ecef in enumerate(landmarks_ecef):
            # account for camera position in ECEF
            camera_position_ecef = camera_model_manager[
                measurement_camera_names[i]
            ].get_camera_position(position_ecef, ecef_R_body)

            vec_ecef = land_pos_ecef - camera_position_ecef
            vec_ecef /= jnp.linalg.norm(vec_ecef)
            body_vec = ecef_R_body.T @ vec_ecef
            estimate = estimate.at[i * 3 : i * 3 + 3].set(body_vec)

        return estimate
    
    def H_landmark_actual(
        self,
        z: np.ndarray,
        camera_model_manager: CameraModelManager,
        measurement_camera_names: np.ndarray,
        x_p: jnp.ndarray,
        epoch: Epoch,
    ) -> jnp.ndarray:
        """
        Calculate the Jacobian of the measurement model with respect to the state.

        :param z: Measurement consisting of the landmark locations in ECI coordinates with shape (N, 3)
        :param camera_model_manager: The camera model manager used to manage the cameras.
        :param measurement_camera_names: Array of names of the cameras that took each measurement.
        :param x_p: Prior state estimate consisting of position, quaternion and velocity with shape (9,)
        :param epoch: The epoch at which the measurement is made. Epoch must be provided for ecef-eci transformation!

        :return: The Jacobian of the measurement model with respect to the state.
        """
        jac = jax.jacobian(self.h_landmark_actual, argnums=3)(
            z, camera_model_manager, measurement_camera_names, x_p, epoch=epoch
        )

        return jac

    # def h_landmark(self, x):
    #     h = jnp.zeros((len(self.data_manager.curr_landmarks)*3))

    #     if self.camera_exists:
    #         for i, landmark in enumerate(self.curr_visible_landmarks):
    #             norm = jnp.linalg.norm(landmark.pos - x[0:3])
    #             h = h.at[i*3:i*3+3].set((landmark.pos - x[0:3])/norm)
    #     return h

    # def H_landmark(self, x):
    #     jac = jax.jacobian(self.h_landmark)(x)
    #     return jac

    def h_inter_range(self, x):
        h = jnp.zeros((len(self.curr_visible_sats)))
        for i, sat in enumerate(self.curr_visible_sats):
            norm = jnp.linalg.norm((x[0:3] - sat.x_p[0:3])/1e3)
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
            
            noise = np.random.normal(loc=0,scale=0.01,size=(1))
            z[i] = np.linalg.norm((self.data_manager.latest_state[0:3] - sat.data_manager.latest_state[0:3])/1e3) + noise
            
        return z
    

    def measure_z_landmark(self) -> np.ndarray:
        z_l = np.zeros((len(self.curr_visible_landmarks)*3))
        if self.camera_exists:
            for i, landmark in enumerate(self.curr_visible_landmarks):
                if self.verbose and ("land" in self.meas_type):
                    print(f"Satellite {self.id} can see landmark {landmark.name}")
                vec = landmark.pos - self.data_manager.latest_state[0:3]
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
                vec = self.data_manager.latest_state[0:3] - sat.data_manager.latest_state[0:3]
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
