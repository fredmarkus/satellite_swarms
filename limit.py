#This code explores the Cramer Rao Bound for satellite constellation and measurements

import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt

num = 5 #number of satellites
mesh = "full" #other options are star (possibly also dense)
dim = 2 #dimension of the space

# Initialize the correct number of distance parameters that need to be estimated. 
params = np.zeros((int(num * (num-1) / 2),1))


# Randomly num satellites in dim space
sat_pos = np.zeros((num,dim))
for i in range(num):
    sat_pos[i,0] = np.random.randint(-100,100)
    sat_pos[i,1] = np.random.randint(-100,100)

# Calculate satellite distances
sat_dist = np.zeros((num,num))
ranges = np.zeros((0,0))
for i in range(num):
    for j in range(i+1,num):
        sat_dist[i,j] = np.linalg.norm(sat_pos[i,:] - sat_pos[j,:])
        ranges = np.append(ranges, [np.linalg.norm(sat_pos[i,:] - sat_pos[j,:])])


print(sat_dist)
print(ranges)

# Plot the satellite positions
plt.plot(sat_pos[:,0],sat_pos[:,1],'ro')
for i in range(num-1):
    for j in range(num-1):
        plt.plot([sat_pos[i,0],sat_pos[j+1,0]],[sat_pos[i,1],sat_pos[j+1,1]],linestyle='dashed',color='red')
plt.axis([-100,100,-100,100])
plt.title('Satellite positions')
plt.show()


# Initialize Gaussian random variables for every distance measurement


# Generate samples from the normal distribution


# Generate covariance matrices
# cov = np.zeros(())

