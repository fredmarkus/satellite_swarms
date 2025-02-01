import numpy as np
from scipy.linalg import block_diag


def R_X(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def R_Y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def R_Z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

# Angle must be in radians
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Define Jacobian with respect to noise
def M_Jac(y):
    M = []
    for i in range(int(y.shape[0]/3)):
        y_i = y[i*3:(i+1)*3]
        lM = 1/np.linalg.norm(y_i) * np.eye(3) - np.outer(y_i, y_i) / np.linalg.norm(y_i)**3
        M.append(lM)
    
    return block_diag(*M)
