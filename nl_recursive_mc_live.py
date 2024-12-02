#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import math
import yaml
import jax
import jax.numpy as jnp
import copy
import threading
import time
from collections import deque

from utils.math_utils import R_X, R_Z


# from functools import partial


# Constants
MU = 3.986004418 * 10**5 # km^3/s^2 # Gravitational parameter of the Earth
R_EARTH = 6378 # km # Radius of the earth
HEIGHT = 550 # km # Height of the satellite
R_SAT = HEIGHT + R_EARTH
ROH_0 = 1.225e9 # kg/km^3 # Density of air at sea level
H_0 = 0 # km # Height of sea level
MASS = 2 # kg # Mass of the satellite
AREA = 1e-9 # km^2 # Cross-sectional area of the satellite
SCALE_HEIGHT = 8.4 # km # Scale height of the atmosphere
C_D = 2.2 # Drag coefficient of the satellite
EQ_RADIUS = 6378.137 # km # Equatorial radius of the Earth
POLAR_RADIUS = 6356.7523 # km # Polar radius of the Earth
J2 = 1.08263e-3 # J2 perturbation coefficient

DENSITY = ROH_0*np.exp(-(HEIGHT-H_0)/SCALE_HEIGHT) # kg/km^3 # Density of the atmosphere at the height of the satellite

class landmark: # Class for the landmark object. Coordinates in ECEF (TODO: Check with Paulo or Zac if this is correct)
    def __init__(self, x: float, y: float, z: float, name: str) -> None:
        self.pos = np.array([x,y,z])
        self.name = name


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
                 camera_exists: bool) -> None:
        
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

        # Initialize the measurement vector with noise
        self.x_m = self.x_0# + np.array([1,0,0,0,0,0]) # Initialize the measurement vector exactly the same as the initial state vector
        x_m_init_noise = np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=int(self.dim/2))
        x_m_init_noise = x_m_init_noise/np.linalg.norm(x_m_init_noise) # Normalize the noise vector
        self.x_m = self.x_m + np.append(x_m_init_noise,np.zeros((3,)),axis=0) # Add the noise to the initial state vector

        self.x_p = self.x_m
        self.cov_p = self.cov_m # Initialize the prior covariance the same as the measurement covariance

        self.min_landmark_list = None # Initialize an array of the closest landmark to the satellite t
        self.curr_pos = self.x_0[0:3] #Determines the current position of the satellite (Necessary for landmark bearing and satellite ranging)
        self.other_sats_pos = np.zeros((N, 3, int(n_sats-1))) # Provides the position of the other satellites for all N timesteps
        # self.sats_visible = np.zeros((N,n_sats-1)) # Determines whether the other satellites are visible to this satellite


    def h_landmark(self, x):
        min_landmark = closest_landmark(x, self.landmarks)
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
            self.min_landmark_list = np.array([closest_landmark(self.curr_pos, landmarks)]).reshape(1,3)
        else:
            self.min_landmark_list = np.append(self.min_landmark_list,[closest_landmark(self.curr_pos, landmarks)],axis=0)
        
        noise = np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(int(self.dim/2)))
        vec = (self.curr_pos  - self.min_landmark_list[-1]) + noise
        return vec/np.linalg.norm(vec)


    def is_visible(self, sat_pos) -> bool:
        # Check if the earth is in the way of the two satellites
        threshold_angle = math.atan(R_EARTH/R_SAT) # Note this is an approximation based on the assumption of a CIRCULAR orbit and earth
        # Calculate the angle between the two satellites
        vec = sat_pos - self.curr_pos
        vec_earth = np.array([0,0,0]) - self.curr_pos
        # Calculate the angle between the two vectors
        angle = math.acos(round(np.dot(vec,vec_earth)/(np.linalg.norm(vec)*np.linalg.norm(vec_earth)),6)) 
        if abs(angle) < threshold_angle:
            return False
        
        return True
    
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
        dist = jnp.linalg.norm(pos[0:3] - landmark.pos)
        if dist < min_dist:
            min_dist = dist
            closest_landmark = landmark

    # print(f"Closest landmark to satellite at position {pos} is {closest_landmark.name} at position {closest_landmark.pos}")
    if closest_landmark is None:
        return np.array([0,0,0])
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


def gravitational_acceleration(r):
    return (-MU / (jnp.linalg.norm(r)**3)) * r

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


def rk4_discretization(x, dt: float):
    r = x[0:3]
    v = x[3:6]

    def dr_dt(v):
        """Derivative of r with respect to time is velocity v."""
        return v

    def dv_dt(r,v):
        """Derivative of v with respect to time is gravitational acceleration."""
        return gravitational_acceleration(r) + atmospheric_drag(v) + j2_dynamics(r)

    # Calculate k1 for r and v
    k1_r = dr_dt(v)
    k1_v = dv_dt(r,v)

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


#General Parameters
N = 250
f = 1 #Hz
dt = 1/f
n_sats = 3
R_weight = 10e-4
bearing_dim = 3
state_dim = 6
meas_dim = n_sats-1 + bearing_dim
R = np.eye(meas_dim)*R_weight
# Process noise covariance matrix based on paper "Autonomous orbit determination and observability analysis for formation satellites" by OU Yangwei, ZHANG Hongbo, XING Jianjun
# page 6
Q = np.diag(np.array([10e-6,10e-6,10e-6,10e-12,10e-12,10e-12]))


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



### Satellite Initialization ###
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

sats = []

for sat_config in config["satellites"]:

    sat_config["N"] = N
    sat_config["landmarks"] = landmark_objects
    sat_config["meas_dim"] = meas_dim
    sat_config["n_sats"] = n_sats
    sat_config["R_weight"] = R_weight

    satellite_inst = satellite(**sat_config)
    sats.append(satellite_inst)


# Function for calculating recursive state estimation in separate thread
def data_processing_loop():
    try:

        first_run = True
        total_steps = 0

        while True:
            # Generate ground-truth synthetic data for the next N timesteps
            x_traj = np.zeros((N, state_dim, n_sats)) # Discretized trajectory of satellite states over time period
            # This loop is deterministic as we are always doing the same discretization so we do not need to regenerate the trajectories.
            for sat in sats:

                if first_run: 
                    x = sat.x_0
                else:
                    x = x_0_next[:,sat.id]

                for i in range(N):
                    x_traj[i,:,sat.id] = x
                    x = rk4_discretization(x, dt)

            # Save the final state as initial state for the next block of N timesteps
            x_0_next = x_traj[-1,:,:]

            # Get the positions of the other satellites for each satellite at all N timesteps
            for sat in sats:
                    sat_i = 0 #iterator variable
                    for other_sat in sats:
                        if sat.id != other_sat.id:
                            sat.other_sats_pos[:,:,sat_i] = x_traj[:,0:3,other_sat.id] # Transfer all N 3D positions of the other satellites from x_traj
                            sat_i += 1


            ## Calculate FIM directly in recursive fashion.
            fim = np.zeros((N*state_dim, N*state_dim))
            
            # If it is the first run, initialize the prior and posterior FIM matrices. Otherwise continue using the previous values.
            if first_run:
                f_prior = np.zeros((state_dim, state_dim))
                f_post = np.zeros((state_dim, state_dim))
                first_run = False


            J = np.zeros((meas_dim, state_dim))
            R_inv = np.linalg.inv(R)

            cov_hist = np.zeros((N,n_sats,state_dim,state_dim))


            y_m = jnp.zeros((N,meas_dim,n_sats))
            H = np.zeros((meas_dim,state_dim))
            h = np.zeros((meas_dim))

            for i in range(N-1):

                for sat in sats:

                    sat.curr_pos = x_traj[i+1, 0:3, sat.id]  # Update position

                    # State prediction
                    A = state_transition(sat.x_m)
                    sat.x_p = rk4_discretization(sat.x_m, dt)
                    sat.cov_p = A @ sat.cov_m @ A.T + Q

                for sat in sats:
                    # Measurement update
                    H[0:bearing_dim, :] = sat.H_landmark(sat.x_p)
                    for j in range(bearing_dim, meas_dim):
                        H[j, :] = sat.H_inter_range(i+1, j, sat.x_p)

                    K = sat.cov_p @ H.T @ np.linalg.pinv(H @ sat.cov_p @ H.T + R)

                    if sat.camera_exists:
                        y_m = y_m.at[i,0:bearing_dim,sat.id].set(sat.measure_z_landmark(tuple(landmark_objects))) # This sets the bearing measurement
                        h[0:bearing_dim] = sat.h_landmark(sat.x_p[0:3])
                    else:
                        y_m = y_m.at[i,0:bearing_dim,sat.id].set(np.zeros((bearing_dim,))) # Set to zero if camera is off
                        h[0:bearing_dim] = np.zeros((bearing_dim,))

                    # Simulate measurements
                    y_m = y_m.at[i,bearing_dim:meas_dim,sat.id].set(sat.measure_z_range(sats)) # This sets the range measurement

                    # Expected measurements

                    for j in range(bearing_dim, meas_dim):
                        h[j] = sat.h_inter_range(i+1, j, sat.x_p[0:3])

                    # State and covariance update
                    sat.x_m = sat.x_p + K @ (y_m[i,:,sat.id] - h)
                    sat.cov_m = (np.eye(state_dim) - K @ H) @ sat.cov_p @ (np.eye(state_dim) - K @ H).T + K @ R @ K.T


                # Record timestep
                with data_lock:
                    error = np.abs(sats[0].x_m[0:3] - sats[0].curr_pos)
                    total_steps += 1

                    error_state_x.append(error[0])
                    error_state_y.append(error[1])
                    error_state_z.append(error[2])
                    timestep_vec.append(total_steps)
                
                    # print("Appended timestep")

                # time.sleep(dt)  # Simulate real-time processing delay
            


                #Assume no knowledge at initial state so we don't place any information in the first [state_dim x state_dim] block
                if i > 0:
                    start_i = i * state_dim
                    A = state_transition(sats[0].x_m)
                    f_prior = A@f_post@A.T + np.linalg.inv(Q) # with process noise
                    J[0:bearing_dim,0:3] = sats[0].H_landmark(sats[0].x_m[0:3])
                    for j in range(3,meas_dim): ## Consider checks for nan values
                        J[j,0:3] = sats[0].H_inter_range(i+1, j, sats[0].x_m[0:3])
                    f_post = f_prior + J.T@R_inv@J
                    fim[start_i:start_i+state_dim,start_i:start_i+state_dim] = f_post

                    # print(f"Satellite {sat.id} at time {i} has covariance {sat.cov_m}")

            if stop_event.is_set():
                print('Stopping data processing loop')
                break

            # Using pseudo-inverse to invert the matrix
            crb = np.linalg.pinv(fim)

            # Plot the covariance matrix and the FIM diagonal entries.
            crb_diag = np.diag(crb)


    except Exception as e:
        print(f"Error in data_processing_loop: {e}")

MAX_POINTS = 1000  # Adjust based on your requirements

# Initialize timestep_vec with a fixed maximum length
timestep_vec = deque(maxlen=MAX_POINTS)

# Initialize error_state for the first satellite's x-error with a fixed maximum length
error_state_x = deque(maxlen=MAX_POINTS)
error_state_y = deque(maxlen=MAX_POINTS)
error_state_z = deque(maxlen=MAX_POINTS)

# Start the data processing thread

# timestep_vec = []
# error_state = [[] for _ in range(n_sats)]
data_lock = threading.Lock()
stop_event = threading.Event()


processing_thread = threading.Thread(target=data_processing_loop, daemon=True,
                                     name='DataProcessingThread')
processing_thread.start()

# data_processing_loop()

# Set up the live plot
fig, ax = plt.subplots()

line_x, = ax.plot([], [], label='Sat0 X-Error', color='r')
line_y, = ax.plot([], [], label='Sat0 Y-Error', color='g')
line_z, = ax.plot([], [], label='Sat0 Z-Error', color='b')


ax.set_xlabel('Timestep')
ax.set_ylabel('Error')
ax.set_title('Live Error State')
ax.legend(loc='upper right')
ax.grid(True)


def init():
    line_x.set_data([], [])
    line_y.set_data([], [])
    line_z.set_data([], [])
    ax.set_autoscalex_on(True)
    ax.set_autoscaley_on(True)
    return line_x, line_y, line_z,

def update(frame):
    with data_lock:
        if not timestep_vec:
            return line_x, line_y, line_z,
    
        current_timestep = list(timestep_vec)
        current_error_x = list(error_state_x)
        current_error_y = list(error_state_y)
        current_error_z = list(error_state_z)       
            
    
    # Update the line data
    line_x.set_data(current_timestep, current_error_x)
    line_y.set_data(current_timestep, current_error_y)
    line_z.set_data(current_timestep, current_error_z)
    
    # Optionally adjust y-axis based on data
    ax.relim()
    ax.autoscale_view(scalex=True,scaley=True) # Only autoscale y-axis

    return line_x, line_y, line_z,

def infinite_frames():
    frame = 0
    while True:
        yield frame
        frame += 1

ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=infinite_frames,    # Use the infinite generator
    save_count=400,           # Specify a reasonable number of frames to cache
    blit=False,
    interval=100,
    repeat=False
)


plt.show()
