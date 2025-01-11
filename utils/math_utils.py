import numpy as np


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

# Converts v to a matrix such that v x u = hat(v)u
def hat(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

# Create a stack of hat matrices for a vector of stacked observations
def hat_stack(v):
    for i in range(0, len(v), 3):
        if i == 0:
            result = hat(v[i:i+3])
        else:
            result = np.concatenate((result, hat(v[i:i+3])))

    return result


# Convert a scalar-first unit quaternion to the left-side matrix for quaternion multiplication
def quaternion_to_left_matrix(q):
    qs, qv = q[0], q[1:4]
    dr = qs*np.eye(3) + hat(qv)

    m1 = np.concatenate((np.array([qs]),   -qv.T))
    m2 = np.concatenate(([qv],   dr)).transpose()
    M = np.concatenate(([m1], m2), axis=0)
    return M

# Convert a scalar-first unit quaternion to a rotation matrix
def quaternion_to_rotation_matrix(q):
    qs, qv = q[0], q[1:4]
    
    return np.eye(3) + 2*hat(qv)@(qs*np.eye(3) + hat(qv))

# Multiply two quaternions
def quaternion_mul(q1,q2):
    return quaternion_to_left_matrix(q1)@q2

