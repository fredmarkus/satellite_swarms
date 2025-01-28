"""
Satellite dynamics module.
"""

from typing import List

import jax
import jax.numpy as jnp
import numpy as np

# pylint: disable=import-error
from sat.core import satellite

# Constants
MU = 3.986004418 * 10**5 # km^3/s^2 # Gravitational parameter of the Earth
HEIGHT = 550 # km # Height of the satellite
ROH_0 = 1.225e9 # kg/km^3 # Density of air at sea level
H_0 = 0 # km # Height of sea level
MASS = 2 # kg # Mass of the satellite
AREA = 1e-9 # km^2 # Cross-sectional area of the satellite
SCALE_HEIGHT = 8.4 # km # Scale height of the atmosphere
C_D = 2.2 # Drag coefficient of the satellite
EQ_RADIUS = 6378.1370 # km # Equatorial radius of the Earth
POLAR_RADIUS = 6356.7523 # km # Polar radius of the Earth
J2 = 1.08263e-3 # J2 perturbation coefficient

DENSITY = ROH_0*np.exp(-(HEIGHT-H_0)/SCALE_HEIGHT) # kg/km^3 # Density of the atmosphere at the height of the satellite


def gravitational_acceleration(r):
    return (-MU / (jnp.linalg.norm(r)**3)) * r


# TODO: Refactor to make this dependent on the satellite object specifically and account for eccentricity of orbit leading to varying density
# Reason: We assume that all satellites have the same mass and area values which is not necessarily true
def atmospheric_drag(v):
    drag = (-0.5*C_D*DENSITY*AREA*v*jnp.linalg.norm(v)**2)/MASS
    return drag


def j2_dynamics(r):
    r_norm = jnp.linalg.norm(r)

    F = 3*MU*J2*EQ_RADIUS**2/(2*r_norm**5)
    a_x = F*(r[0])*(5*(r[2]/r_norm)**2 - 1)
    a_y = F*(r[1])*(5*(r[2]/r_norm)**2 - 1)
    a_z = F*(r[2])*(5*(r[2]/r_norm)**2 - 3)

    return jnp.array([a_x, a_y, a_z])


def j2_jacobian(r):
    jac = jax.jacobian(j2_dynamics)(r)
    return jac


def rk4_discretization(x, dt: float):
    r = x[0:3]
    v = x[3:6]

    def dr_dt(v):
        """Derivative of r with respect to time is velocity v."""
        return v

    def dv_dt(r, v):
        """Derivative of v with respect to time is gravitational acceleration."""
        return gravitational_acceleration(r) + j2_dynamics(r) + atmospheric_drag(v)

    # Calculate k1 for r and v
    k1_r = dr_dt(v)
    k1_v = dv_dt(r, v)

    # Calculate k2 for r and v
    k2_r = dr_dt(v + 0.5 * dt * k1_v)
    k2_v = dv_dt(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v)

    # Calculate k3 for r and v
    k3_r = dr_dt(v + 0.5 * dt * k2_v)
    k3_v = dv_dt(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v)

    # Calculate k4 for r and v
    k4_r = dr_dt(v + dt * k3_v)
    k4_v = dv_dt(r + dt * k3_r, v + dt * k3_v)

    # Combine the k terms to get the next position and velocity
    r_new = r + (dt / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    v_new = v + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    # Return the updated state vector
    return jnp.concatenate((r_new, v_new))


# Numerical solution of the dynamics using Euler's method
def euler_discretization(x: np.ndarray, dt: float) -> np.ndarray:
    r = x[0:3]
    v = x[3:6]
    r_new = r + v*dt
    v_new = v + (-MU/(np.linalg.norm(r)**3))*r*dt
    return np.concatenate((r_new, v_new))


def state_transition(x):
    I = np.eye(3)
    # Account for j2 dynamics
    j2_jac = j2_jacobian(x[0:3])

    # Account for gravitational acceleration
    grav_jac = -MU/(np.linalg.norm(x[0:3])**3)*I + 3*MU*np.outer(x[0:3],x[0:3])/(np.linalg.norm(x[0:3])**5)

    # Account for atmospheric drag
    drag_jac = -DENSITY*C_D*AREA/(2*MASS)*(np.outer(x[3:6],x[3:6])/np.linalg.norm(x[3:6]) + I*np.linalg.norm(x[3:6]))

    A = np.block([[np.zeros((3,3)), I], [grav_jac+j2_jac, drag_jac]])
    return A


def simulate_nominal_trajectories(timesteps: int, dt: float, sats: List[satellite], state_dim: int) -> np.ndarray:
    """
    Calculate the nominal trajectories of the satellites over a given number of timesteps.

    Args:
    timesteps (int): Number of timesteps to simulate the trajectories for.
    dt (float): Time step size.
    sats (List[satellite]): List of satellite objects for which to simulate the trajectories.
    state_dim (int): Dimension of the state vector.

    Returns:
        np.ndarray: Discretized trajectory of satellite states over the time period for all satellites.
    """

    x_traj = np.zeros(
        (timesteps + 1, state_dim, len(sats))
    )
    for sat in sats:
        x = sat.x_0
        for i in range(timesteps + 1):
            x_traj[i, :, sat.id] = x
            x = rk4_discretization(x, dt)
    return x_traj


def exchange_trajectories(sats: List[satellite], x_traj: np.ndarray,) -> List[satellite]:
    """
    Transfer the positions of the other satellites for each satellite at all timesteps.

    Args:
    sats (List[satellite]): List of satellite objects for which to transfer the positions.
    x_traj (np.ndarray): Discretized trajectory of satellite states over the time period for all satellites.

    Returns:
        List[satellite]: List of satellite objects with the positions of the other satellites transferred.
    """
    for sat in sats:
        sat_i = 0
        for other_sat in sats:
            if sat.id != other_sat.id:
                sat.other_sats_pos[:, :, sat_i] = x_traj[
                    :, 0:3, other_sat.id
                ]  # Transfer all N+1 3D positions of the other satellites from x_traj
                sat_i += 1

    return sats
