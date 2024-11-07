#This is a dummy example to consider the evolution of information using a robot driving back and forth between two points and two beacons.

import numpy as np
import matplotlib.pyplot as plt
import scipy

class beacon:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def ground_truth_d(self, index):
        if index % 2 != 0:
            current_position = np.array([[0],[1]])
        else:
            current_position = np.array([[1],[0]])
        return np.linalg.norm(np.array([current_position[0] - self.x, current_position[1] - self.y]))
    
    def h(self,pos):
        return np.linalg.norm(np.array([pos[0][0] - self.x, pos[1][0] - self.y])) # Measurement model with E[v] = 0
    
    def H(self, pos):
        dx = (pos[0][0] - self.x)/(np.linalg.norm(np.array([pos[0][0] - self.x, pos[1][0] - self.y]))) 
        dy = (pos[1][0] - self.y)/(np.linalg.norm(np.array([pos[0][0] - self.x, pos[1][0] - self.y])))
        return np.array([[dx, dy]])
    

class robot:

    def __init__(self):
        self.pos = np.array([[0],[1]])
        self.cov = 1*np.eye(2)

        
N = 1000 # Number of timesteps

# Init 
beacons = []
n_beacons = 3 # Number of beacons
beacon1 = beacon(0,0)
beacon2 = beacon(1,1)
beacons = [beacon1, beacon2]
# beacon3 = beacon(0,5)
# beacons = [beacon1, beacon2, beacon3]
robot = robot()

Q = 0.25*np.eye(2) # Process noise covariance
R = 0.25*np.eye(n_beacons) # Measurement noise covariance
K = np.zeros((2,n_beacons)) # Kalman gain
H = np.zeros((n_beacons,2)) # Measurement Jacobian
z = np.zeros((n_beacons,1)) # Measurement vector
actual_pos = np.zeros((n_beacons,1))
info_matrix = np.linalg.inv(robot.cov)
info_vector = info_matrix@robot.pos

# Collect data for plotting

info_vectors = np.zeros((N,2))
positions = np.zeros((N,2))


# Dynamics of robot
A = np.array([[0,1],[1,0]])

i = 0
for i in range(N):

    # Prior update (Assume no uncertainty in the robot movement)
    robot.pos = np.dot(A,robot.pos) # Move the robot: x_p = A*x_m + E(v)
    robot.cov = A@robot.cov@A.T + Q # Update the covariance: P_p = A*P*A^T + LQL^T

    #Calculate information matrix and vector
    info_matrix_p = np.linalg.inv(robot.cov)
    info_vector_p = info_matrix_p@robot.pos

    print(info_vector_p)

    # Measurement update & calculate the information matrix and vector for the measurement
    for j,beacon in enumerate(beacons):
        H[j] = beacon.H(robot.pos)
        z[j] = beacon.ground_truth_d(i) + np.random.normal(0,R[j,j],1) # Based on the ground truth position of the robot (no uncertainty in positional movement)
    
        actual_pos[j] = beacon.h(robot.pos) # Based on where the measurement model believes the robot is

    K = robot.cov@H.T@np.linalg.inv(H@robot.cov@H.T + R)
    robot.pos = robot.pos + K@(z - actual_pos)
    robot.cov = (np.eye(2) - K@H)@robot.cov

    # Update the information matrix and vector
    I_matrix = H.T@np.linalg.inv(R)@H
    I_vector = H.T@np.linalg.inv(R)@z
    
    info_matrix = info_matrix_p + I_matrix
    info_vector = info_vector_p + I_vector

    # Store the information vector and position vector

    info_vectors[i] = info_vector.T
    positions[i] = robot.pos.T
    

#Plotting
for i in range (N):
    plt.plot(positions[i,0],positions[i,1],'ro')
plt.axis([-2,2,-2,2])
plt.title('Robot position')


plt.figure()
plt.plot(info_vectors[:, 0], info_vectors[:, 1], marker='x', color='blue', linestyle='-', markersize=5)

plt.title('Information vector')


sum_info = np.sum(np.absolute(info_vectors), axis=1)

plt.figure()
plt.plot(sum_info, marker='x', color='green', linestyle='-', markersize=1)
plt.title('sum of dimensions of information vector')
plt.show()
