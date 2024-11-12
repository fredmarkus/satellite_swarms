# This code considers the non-linear least square implementation of the problem described before
# We now consider the dynamics  of an actual satellite flying in a polar orbit.

#Simulate a polar orbit with a satellite that has states r and v where both are 3 dimensional vectors
# Dynamics are given by: 
# r_dot = v
# v_dot = a = mu * r / ||r||^3
# where r is the position vector, v is the velocity vector, a is the acceleration, and mu is the gravitational parameter

# Satellite flies at an altitude of ca. 550km above the earth's surface
# The earth's radius is 6378km and the gravitational parameter is 3.986004418 x 10^5 km^3/s^2

from gekko import GEKKO
import numpy as np
import math
import csv

# Constants
mu = 3.986004418 * 10**5 # km^3/s^2
r_earth = 6378
r_sat = 550 + r_earth

class satellite:

    def __init__(self, state, rob_cov_init, robot_id, dim, meas_dim, Q_weight, R_weight):
        self.x_m = state
        self.cov_m = rob_cov_init*np.eye(dim)
        self.id = robot_id # Unique identifier for the satellite
        self.dim = dim
        self.meas_dim = meas_dim
        self.R_weight = R_weight # This is the variance weight for the measurement noise
        self.Q_weight = Q_weight # This is the variance weight for the process noise

        # self.A = np.array([[0,1],[1,0]]) # TODO: Use actual dynamics
        self.info_matrix = np.linalg.inv(self.cov_m)
        self.info_vector = self.info_matrix@self.x_m
        self.x_p = self.x_m # Initialize the prior vector exactly the same as the measurement vector
        self.cov_p = self.cov_m # Initialize the prior covariance exactly the same as the measurement covariance


    def h_landmark(self, landmark):
        return (self.x_p[0:3] - landmark.pos)/np.linalg.norm(self.x_p[0:3] - landmark.pos) # This provides a normalized vector (the bearing)

    def H_landmark(self, landmark):
        return -(self.x_p[0:3] - landmark.pos)@(self.x_p[0:3] - landmark.pos).T/np.linalg.norm(self.x_p[0:3] - landmark.pos)**3 + np.eye(self.dim/2)/np.linalg.norm(self.x_p[0:3] - landmark.pos)
    
    
    def h_inter_range(self, sat_pos):
        return np.array([np.linalg.norm(self.x_p[0:3] - sat_pos)])
    
    def H_inter_range(self, sat_pos):
        dx = (sat_pos[0] - self.x_p[0])/np.linalg.norm(self.x_p[0:3] - sat_pos)
        dy = (sat_pos[1] - self.x_p[1])/np.linalg.norm(self.x_p[0:3] - sat_pos)
        dz = (sat_pos[2] - self.x_p[2])/np.linalg.norm(self.x_p[0:3] - sat_pos)
        return np.array([[dx, dy, dz]])
    
    def h_combined(self, landmark, sats):
        h = self.h_landmark(landmark)
        for sat in sats:
            if sat.id != self.id:
                h = np.append(h, self.h_inter_range(sat.x_p),axis=0)

        return h

    # This function combines the H matrices for the bearing and inter-satellite range measurements
    def combined_H(self, landmark, sats):
        H = self.H_landmark(landmark)
        for sat in sats:
            if sat.id != self.id:
                H = np.vstack((H, self.H_inter_range(sat.x_p)))

        return H
    

    def measure_z_landmark(self, landmark):
        vec = (self.x_p[0:3] - landmark.pos) + np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(self.dim/2))
        return vec/np.linalg.norm(self.x_p - landmark.pos)

    def measure_z_range(self, sats):
        z = np.empty((0))
        for sat in sats:
            if sat.id != self.id:
                d = np.array([np.linalg.norm(self.x_p[0:3] - sat.x_p[0:3])]) + np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(1))
                z = np.append(z,d,axis=0)
        # print(f'ranges of sat{self.id} is: {z}')
        return z
    
    def measure_z_combined(self, landmark, sats):
        z = self.measure_z_landmark(landmark) # Take bearing measurement
        if len(sats) > 1:
            z = np.append(z, self.measure_z_range(sats),axis=0) # Append the range measurement

        return z


# Parameters
N = 1000
f = 1 # Hz
dt = 1/f

# Initialization of parameters
x0 = np.array([r_sat, 0, 0, 0, 0, np.sqrt(mu/r_sat)]) # Initial state of the satellite. For now assume a circular orbit

# Numerical solution of the dynamics using Euler's method TODO: Implement RK4
def euler_discretization(x, dt):
    r = x[0:3]
    v = x[3:6]
    r_new = r + v*dt
    v_new = v + (-mu/(np.linalg.norm(r)**3))*r*dt
    return np.concatenate((r_new, v_new))



'''
TODO: Discretize the dynamics of the satellite and implement timesteps that allow us to use nonlinear least squares. Potentially add some noise to the process.
TODO: Implement measurement model between the satellite and some landmarks and inter-range satellites. Then use this to fake data by adding some noise
TODO: Place everything in the GEKKO solver to get nonlinear least squares solution for states. Consider the best version of the problem to use. (Consider the numerically advantageous option that uses the Cholesky decomposition.)
TODO: Scale this to multiple satellites and landmarks
'''

# Test the euler discretization
x = x0
for i in range(N):
    x = euler_discretization(x, dt) # These are the updated states of the satellites. 
    # print(x)
    # print(np.linalg.norm(x[0:3])) # Check if the orbit remains circular (it should) Compare to r_sat (6928))


# Import csv data for the landmarks
landmarks = []
with open('landmark_coordinates.csv', newline='',) as csvfile:
    reader = csv.reader(csvfile, delimiter=',',)
    for row in reader:
        landmarks.append(np.array([float(row[1]), float(row[2]), float(row[3])]))

def latlon2ecef(landmarks):
    ecef = np.array([])
    a = 6378.137
    b = 6356.7523
    e = 1 - (b**2/a**2)
    
    # helper function
    def N(a,b,lat):
        return a**2/np.sqrt(a**2*np.cos(lat)**2 + b**2*np.sin(lat)**2)
    
    for landmark in landmarks:
        X = (N(a,b,landmark[0]) + landmark[2])*np.cos(landmark[0])*np.cos(landmark[1])
        Y = (N(a,b,landmark[0]) + landmark[2])*np.cos(landmark[0])*np.sin(landmark[1])
        Z = (N(a,b,landmark[0])*(1-e) + landmark[2])*np.sin(landmark[0])
        ecef = np.append(ecef, np.array([X,Y,Z]), axis=0)
    
    print(ecef.shape)
    return ecef.reshape(-1,3)


landmarks_ecef = latlon2ecef(landmarks)
print(landmarks_ecef) 

# The earth landmarks are in ECEF coordinates. We will make the assumption for now that only the closest landmark is used for the bearing measurement.
# So calculate distance to all landmarks and then choose the closest one and calculate bearing to that.

