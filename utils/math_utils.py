"""
Module for mathematical utilities.
"""
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

def vector_to_az_el(vec: np.ndarray) -> tuple:
    """
    Convert a 3D vector to azimuth and elevation.
    Args:
        vec: 3D vector
    
    Returns:
        Azimuth and elevation in radians
    """
    # Assumes vec is [x, y, z]
    if np.linalg.norm(vec) == 0:
        return 0, 0
    
    x, y, z = vec
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    return azimuth, elevation

def az_el_to_vector(az: float, el: float) -> np.ndarray:
    """
    Convert azimuth and elevation to a 3D vector.
    Args:
        az: Azimuth in radians.
        el: Elevation in radians.

    Returns:
        3D vector
    """
    # Assumes azimuth and elevation are in radians
    y = np.cos(el[0]) * np.sin(az[0])
    x = np.cos(el[0]) * np.cos(az[0])
    z = np.sin(el[0])
    return np.array([x, y, z])

def transform_eci_to_lvlh(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Transform ECI coordinates to LVLH frame.
    Args:
        x: ECI position vector
        v: ECI velocity vector
    
    Returns:
        R: Rotation matrix from ECI to LVLH frame
    """
    h_new = np.cross(x, v)
    h_new = h_new / np.linalg.norm(h_new)
    z_new = -x / np.linalg.norm(x)
    x_new = np.cross(z_new, h_new)
    # Define y_new to complete right handed system
    y_new = np.cross(z_new, x_new)

    R = np.column_stack((x_new, y_new, z_new))

    return R


