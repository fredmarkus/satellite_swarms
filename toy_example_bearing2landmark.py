# Bearing to landmark toy example using another measurement model

#This is a dummy example to consider the evolution of information using a robot driving back and forth between two points and two beacons.

import numpy as np
import matplotlib.pyplot as plt

class landmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class robot:

    def __init__(self, rob_cov_init):
        self.pos = np.array([[2],[0]])
        self.cov = rob_cov_init*np.eye(2)

    def h(self, landmark): #Provide a bearing
        return (self.pos - np.array([[landmark.x],[landmark.y]]))/np.linalg.norm(self.pos - np.array([[landmark.x],[landmark.y]])) # This provides a normalized vector (the bearing)
    
    def H(self, landmark):
        coeff = 1/np.linalg.norm(self.pos - np.array([[landmark.x],[landmark.y]]))
        H11 = ((self.pos[0][0] - landmark.x)**2)/(np.linalg.norm(self.pos - np.array([[landmark.x],[landmark.y]])))**6 + 1
        H12 = (self.pos[0][0] - landmark.x)*(self.pos[1][0] - landmark.y)/(np.linalg.norm(self.pos - np.array([[landmark.x],[landmark.y]]))**6)
        H22 = ((self.pos[1][0] - landmark.y)**2)/(np.linalg.norm(self.pos - np.array([[landmark.x],[landmark.y]])))**6 + 1
        return coeff*np.array([[H11, H12],[H12, H22]])
    
    def z_measure(self, index, R_weight):
        if index % 2 == 0:
            current_position = np.array([[0],[2]])
        else:
            current_position = np.array([[2],[0]])

        current_position = current_position + np.random.normal(loc=0,scale=R_weight,size=(2,1))

        return current_position/np.linalg.norm(current_position)

    



# Parameters
N = 1000  # Number of timesteps
n_landmarks = 1 # Number of landmarks
Q_weight = 0.5
R_weight = 0.25
rob_cov_init = 0.25

# Init 

landmark = landmark(0,0)

robot = robot(rob_cov_init)

Q = Q_weight*np.eye(2) # Process noise covariance
R = R_weight*np.eye(2) # Measurement noise covariance
K = np.zeros((2,2)) # Kalman gain
H = np.zeros((2,2)) # Measurement Jacobian
z = np.zeros((2,1)) # Measurement vector
actual_pos = np.zeros((2,1))
info_matrix = np.linalg.inv(robot.cov)
info_vector = info_matrix@robot.pos

# Collect data for plotting

info_vectors = np.zeros((N,2))
positions = np.zeros((N,2))


# Dynamics of robot
A = np.array([[0,1],[1,0]]) # Simple back and forth motion between (0,1) and (1,0)

i = 0
for i in range(N):

    # Prior update (Assume no uncertainty in the robot movement)
    robot.pos = A@robot.pos  
    robot.cov = A@robot.cov@A.T + Q # Update the covariance: P_p = A*P*A^T + LQL^T

    #Calculate information matrix and vector
    info_matrix_p = np.linalg.inv(robot.cov)
    info_vector_p = info_matrix_p@robot.pos

    # Measurement update & calculate the information matrix and vector for the measurement
    # for j,landmark in enumerate(n_landmarks):
    H = robot.H(landmark)
    z = robot.z_measure(i, R_weight) # Based on the ground truth position of the robot with some noise
    actual_pos = robot.h(landmark) # Based on where the measurement model believes the robot is

    K = robot.cov@H.T@np.linalg.inv(H@robot.cov@H.T + R)
    robot.pos = robot.pos + K@(z - actual_pos)
    robot.cov = (np.eye(2) - K@H)@robot.cov


    # Update the information matrix and vector
    I_matrix = H.T@np.linalg.inv(R)@H   
    I_vector = H.T@np.linalg.inv(R)@z
    
    info_matrix = info_matrix_p + I_matrix
    info_vector = info_vector_p + I_vector

    # print("-------------------------------------")
    # print("Measurement Jacobian: \n", H)
    # print("Robot covariance", robot.cov)
    # print("Information matrix: ", info_matrix)

    # Store the information vector and position vector
    # info_vectors[i] = info_vector.T
    positions[i] = robot.pos.T


#Plotting
for i in range (N):
    if i == 0:
        plt.plot(positions[i,0],positions[i,1],'ro', label='Robot positions')
    else:
        plt.plot(positions[i,0],positions[i,1],'ro')

# for i in range(n_beacons):
#     if i == 0:
#         plt.plot(beacons[i].x,beacons[i].y,'yx',label='Beacon')
#     else:
#         plt.plot(beacons[i].x,beacons[i].y,'yx')
# plt.axis([-4,4,-4,4])
# plt.title('Robot position')
# plt.legend()


# plt.figure()
# plt.plot(info_vectors[:, 0], info_vectors[:, 1], marker='x', color='blue', linestyle='-', markersize=5)
# plt.title('Information vector')


# sum_info = np.sum(np.absolute(info_vectors), axis=1)
# plt.figure()
# plt.plot(sum_info, marker='x', color='green', linestyle='-', markersize=1)
# plt.title('sum of dimensions of information vector')
plt.show()
