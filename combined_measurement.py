# This is supposed to combine the previous example measurements into a single example.
# We have the inter-satellite range measurements between each individual pairing of satellites, as well as the bearing measurements between the satellite and the landmark.
# Optionally we could add the range measurements between the satellite and some kind of beacon.
# Right now we are considering purely the range measurements between the satellites but not really taking the dynamics into account in any meaningful way. 

import numpy as np
import math

class landmark:
    def __init__(self, x, y):
        self.pos = np.array([x,y])


class robot:

    def __init__(self, position, rob_cov_init, robot_id, dim, meas_dim, Q_weight, R_weight):
        self.pos_m = np.array([position[0],position[1]])
        self.cov_m = rob_cov_init*np.eye(2)
        self.id = robot_id # Unique identifier for the satellite
        self.dim = dim
        self.meas_dim = meas_dim
        self.R_weight = R_weight # This is the variance weight for the measurement noise
        self.Q_weight = Q_weight # This is the variance weight for the process noise

        self.actual_pos = np.array([position[0],position[1]])
        self.A = np.array([[0,1],[1,0]])
        self.info_matrix = np.linalg.inv(self.cov_m)
        self.info_vector = self.info_matrix@self.pos_m
        self.pos_p = self.pos_m # Initialize the prior vector exactly the same as the measurement vector
        self.cov_p = self.cov_m # Initialize the prior covariance exactly the same as the measurement covariance


    def h_landmark(self, landmark):
        return (self.pos_p - landmark.pos)/np.linalg.norm(self.pos_p - landmark.pos) # This provides a normalized vector (the bearing)

    def H_landmark(self, landmark):
        # coeff = 1/np.linalg.norm(self.pos_p - landmark.pos)
        H11 = (((self.pos_p[0] - landmark.pos[0])**2)/(np.linalg.norm(self.pos_p - landmark.pos))**3) + 1/np.linalg.norm(self.pos_p - landmark.pos)
        H12 = (self.pos_p[0] - landmark.pos[0])*(self.pos_p[1] - landmark.pos[1])/(np.linalg.norm(self.pos_p - landmark.pos)**3)
        H22 = (((self.pos_p[1] - landmark.pos[1])**2)/(np.linalg.norm(self.pos_p - landmark.pos))**3) + 1/np.linalg.norm(self.pos_p - landmark.pos)
        return np.array([[H11, H12],[H12, H22]])
    
    def h_inter_range(self, sat_pos):
        return np.array([np.linalg.norm(self.pos_p - sat_pos)])
    
    def H_inter_range(self, sat_pos):
        dx = (sat_pos[0] - self.pos_p[0])/np.linalg.norm(self.pos_p - sat_pos)
        dy = (sat_pos[1] - self.pos_p[1])/np.linalg.norm(self.pos_p - sat_pos)
        return np.array([[dx, dy]])
    
    def h_combined(self, landmark, sats):
        h = self.h_landmark(landmark)
        for sat in sats:
            if sat.id != self.id:
                h = np.append(h, self.h_inter_range(sat.pos_p),axis=0)

        return h

    # This function combines the H matrices for the bearing and inter-satellite range measurements
    def combined_H(self, landmark, sats):
        H = self.H_landmark(landmark)
        for sat in sats:
            if sat.id != self.id:
                H = np.vstack((H, self.H_inter_range(sat.pos_p)))

        return H
    
    def true_new_pos(self):
        # The satellite dynamics are well known. We don't have process noise that could affect the calculated position
        self.actual_pos = self.A@self.actual_pos #+ np.random.normal(loc=0,scale=math.sqrt(self.Q_weight),size=(self.dim))


    def measure_z_landmark(self, landmark):
        vec = (self.pos_p - landmark.pos) + np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(self.dim))
        return vec/np.linalg.norm(self.pos_p - landmark.pos)
        

    def measure_z_range(self, sats):
        z = np.empty((0))
        for sat in sats:
            if sat.id != self.id:
                d = np.array([np.linalg.norm(self.pos_p - sat.pos_p)]) + np.random.normal(loc=0,scale=math.sqrt(self.R_weight),size=(1))
                z = np.append(z,d,axis=0)
# TODO: CHECK IF THIS IS ACTUALLY CORRECT TO USE SAT.POS_P RATHER THAN SAT.POS_M OR USING A BELIEF STATE OF THE OTHER SATELLITES THAT WE COMMUNICATE
        # print(f'ranges of sat{self.id} is: {z}')
        return z
    
    def measure_z_combined(self, landmark, sats):
        z = self.measure_z_landmark(landmark) # Take bearing measurement
        if len(sats) > 1:
            z = np.append(z, self.measure_z_range(sats),axis=0) # Append the range measurement

        return z


# Parameters
N = 100  # Number of timesteps
n_landmarks = 1 # Number of landmarks
n_satellites = 1 # Number of satellites
Q_weight = 0.01
R_weight = 0.01
rob_cov_init = 0.01
dim = 2 #dimension of the state space
meas_dim = dim + n_satellites # dimension is 1 bearing measurement that has dimension dim and n_satellites range measurements each with dimension 1 


# Init 
landmark = landmark(0,0)

sat1 = robot(position=[1,-1],
             rob_cov_init=rob_cov_init,
             robot_id=0, 
             dim=dim, 
             meas_dim=meas_dim,
             Q_weight=Q_weight,
             R_weight=R_weight)

sat2 = robot(position=[3,0],
             rob_cov_init=rob_cov_init,
             robot_id=1,
             dim=dim,
             meas_dim=meas_dim,
             Q_weight=Q_weight,
             R_weight=R_weight)

sats = [sat1, sat2]

Q = Q_weight*np.eye(dim) # Process noise covariance
R = R_weight*np.eye(meas_dim) # Measurement noise covariance


for i in range(N):
    # All satellites first need to make their prior position update before they all make their measurements
    for sat in sats:
        #Update the ground truth position of the satellite
        sat.true_new_pos()
        sat.pos_p = sat.A@sat.pos_m
        sat.cov_p = sat.A@sat.cov_m@sat.A.T #+ Q # Update the covariance matrix as a P_p = A*P*A^T + LQL^T
        
        # Update the satellites information matrix and vector
        sat.info_matrix = np.linalg.inv(sat.cov_p)
        sat.info_vector = sat.info_matrix@sat.pos_p

    # With updated priors, the satellites can now make measurement updates. 
    for sat in sats:
        H = sat.combined_H(landmark, sats)
        K = sat.cov_p@H.T@np.linalg.pinv(H@sat.cov_p@H.T + R) # Regularization term in the inverse

        # Make the measurement
        z = sat.measure_z_combined(landmark, sats)
        sat.pos_m = sat.pos_p + K@(z - sat.h_combined(landmark, sats))
        sat.cov_m = (np.eye(dim) - K@H)@sat.cov_p@((np.eye(dim) - K@H).T)+ K@R@K.T #Joseph form; regular updated proved to be numerically unstable
        
        # Update the information matrix and vector with measurements
        I_matrix = H.T@np.linalg.inv(R)@H
        I_vector = H.T@np.linalg.inv(R)@z

        sat.info_matrix = sat.info_matrix + I_matrix
        sat.info_vector = sat.info_vector + I_vector

    print(sat2.pos_m)



        #TODO: Implement plotting


#TODO: Plotting and Monte Carlo Sim for Info matrix 
