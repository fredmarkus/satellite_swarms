# This is supposed to combine the previous example measurements into a single example.
# We have the inter-satellite range measurements between each individual pairing of satellites, as well as the bearing measurements between the satellite and the landmark.
# Optionally we could add the range measurements between the satellite and some kind of beacon.
# Right now we are considering purely the range measurements between the satellites but not really taking the dynamics into account in any meaningful way. 

import numpy as np


class landmark:
    def __init__(self, x, y):
        self.pos = np.array([x,y])


class robot:

    def __init__(self, position, rob_cov_init, robot_id):
        self.pos_m = np.array([position[0],position[1]])
        self.cov_m = rob_cov_init*np.eye(2)
        self.actual_pos = np.array([position[0],position[1]])
        self.A = np.array([[0,1],[1,0]])
        self.info_matrix = np.linalg.inv(self.cov_m)
        self.info_vector = self.info_matrix@self.pos_m
        self.pos_p = self.pos_m # Initialize the prior vector exactly the same as the measurement vector
        self.cov_p = self.cov_m # Initialize the prior covariance exactly the same as the measurement covariance
        self.id = robot_id # Unique identifier for the satellite


    def h_bearing(self, landmark):
        return (self.pos_p - landmark.pos)/np.linalg.norm(self.pos_p - landmark.pos) # This provides a normalized vector (the bearing)

    def H_bearing(self, landmark):
        coeff = 1/np.linalg.norm(self.pos_p - landmark.pos)
        H11 = ((self.pos_p[0] - landmark.pos[0])**2)/(np.linalg.norm(self.pos_p - landmark.pos))**6 + 1
        H12 = (self.pos_p[0] - landmark.pos[0])*(self.pos_p[1] - landmark.pos[1])/(np.linalg.norm(self.pos_p - landmark.pos)**6)
        H22 = ((self.pos_p[1] - landmark.pos[1])**2)/(np.linalg.norm(self.pos_p - landmark.pos))**6 + 1
        return coeff*np.array([[H11, H12],[H12, H22]])
    
    def h_inter_range(self, sat_pos):
        return np.array([np.linalg.norm(self.pos_p - sat_pos)])
    
    def H_inter_range(self, sat_pos):
        dx = (sat_pos[0] - self.pos_p[0])/np.linalg.norm(self.pos_p - sat_pos)
        dy = (sat_pos[1] - self.pos_p[1])/np.linalg.norm(self.pos_p - sat_pos)
        return np.array([[dx, dy]])
    
    def h_combined(self, landmark, sats):
        h = self.h_bearing(landmark)
        for sat in sats:
            if sat.id != self.id:
                h = np.append(h, self.h_inter_range(sat.pos_p),axis=0)

        return h

    # This function combines the H matrices for the bearing and inter-satellite range measurements
    def combined_H(self, landmark, sats):
        H = self.H_bearing(landmark)
        for sat in sats:
            if sat.id != self.id:
                H = np.vstack((H, self.H_inter_range(sat.pos_p)))

        return H
    
    def true_new_pos(self, Q_weight):
        self.actual_pos = self.A@self.actual_pos + np.random.normal(loc=0,scale=Q_weight,size=(2))


    def measure_z_landmark(self, landmark):
        return (self.actual_pos -landmark.pos)/np.linalg.norm(self.actual_pos - landmark.pos)
        

    def measure_z_range(self, sats):
        z = np.empty((0))
        for sat in sats:
            if sat.id != self.id:
                z = np.append(z,np.array([np.linalg.norm(self.actual_pos - sat.actual_pos)]),axis=0)

        return z
    
    def measure_z_combined(self, landmark, sats):
        z = self.measure_z_landmark(landmark) # Take bearing measurement
        z = np.append(z, self.measure_z_range(sats),axis=0) # Append the range measurement

        return z


# Parameters
N = 1000  # Number of timesteps
n_landmarks = 1 # Number of landmarks
n_satellites = 1 # Number of satellites
Q_weight = 0.1
R_weight = 0.1
rob_cov_init = 0.1
dim = 2 #dimension of the space
meas_dim = dim + n_satellites # dimension is 1 bearing measurement that has dimension dim and n_satellites range measurements each with dimension 1 


# Init 
landmark = landmark(1,1)

sat1 = robot([1,-1], rob_cov_init, 0)
sat2 = robot([3,0], rob_cov_init, 1)
sats = [sat1, sat2]

Q = Q_weight*np.eye(dim) # Process noise covariance
R = R_weight*np.eye(meas_dim) # Measurement noise covariance


for i in range(N):
    # All satellites first need to make their prior position update before they all make their measurements
    for sat in sats:
        #Update the ground truth position of the satellite
        sat.true_new_pos(Q_weight)
        sat.pos_p = sat.A@sat.pos_m
        sat.cov_p = sat.A@sat.cov_m@sat.A.T + Q # Update the covariance matrix as a P_p = A*P*A^T + LQL^T
        
        # Update the satellites information matrix and vector
        sat.info_matrix = np.linalg.inv(sat.cov_p)
        sat.info_vector = sat.info_matrix@sat.pos_p

    # With updated priors, the robots can now make measurement updates. 
    for sat in sats:
        H = sat.combined_H(landmark, sats)
        K = sat.cov_p@H.T@np.linalg.inv(H@sat.cov_p@H.T + R)

        # Make the measurement
        z = sat.measure_z_combined(landmark, sats)
        sat.pos_m = sat.pos_p + K@(z - sat.h_combined(landmark, sats))
        sat.cov_m = (np.eye(dim) - K@H)@sat.cov_p

        # Update the information matrix and vector with measurements
        I_matrix = H.T@np.linalg.inv(R)@H
        I_vector = H.T@np.linalg.inv(R)@z

        sat.info_matrix = sat.info_matrix + I_matrix
        sat.info_vector = sat.info_vector + I_vector

    # print(sat1.info_matrix)
    print(sat1.actual_pos)


        #TODO: Implement plotting


#TODO: Plotting and Monte Carlo Sim for Info matrix 
