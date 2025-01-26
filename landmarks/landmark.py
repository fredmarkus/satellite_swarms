import numpy as np

# Constants
EQ_RADIUS = 6378.1370 # km # Equatorial radius of the Earth
POLAR_RADIUS = 6356.7523 # km # Polar radius of the Earth

class landmark: # Class for the landmark object. Coordinates in ECEF (TODO: Check with Paulo or Zac if this is correct)
    def __init__(self, x: float, y: float, z: float, name: str) -> None:
        self.pos = np.array([x,y,z])
        self.name = name

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
    ecef = []
    e_sq = 1 - (POLAR_RADIUS**2/EQ_RADIUS**2)
    
    # helper function
    def N(a,b,lat):
        return a**2 / np.sqrt(a**2 * np.cos(lat)**2 + b**2 * np.sin(lat)**2)
    
    for mark in landmarks:
        lat_deg = float(mark[1])
        lon_deg = float(mark[2])
        h = float(mark[3])

        # Convert degrees to radians
        lat_rad = np.deg2rad(lat_deg)
        lon_rad = np.deg2rad(lon_deg)

        N_val = N(EQ_RADIUS, POLAR_RADIUS, lat_rad)
        X = (N_val + h) * np.cos(lat_rad) * np.cos(lon_rad)
        Y = (N_val + h) * np.cos(lat_rad) * np.sin(lon_rad)
        Z = (N_val * (1 - e_sq) + h) * np.sin(lat_rad)

        ecef.append([mark[0], X, Y, Z])

    
    return np.array(ecef)
