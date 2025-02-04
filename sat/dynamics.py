"""
Functions for implementing orbital position dynamics and its jacobian under just the force of gravity.
J2 perturbations are not included.
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
            x = f(x, dt)
    return x_traj


def exchange_trajectories(sats: List[satellite], x_traj: np.ndarray,) -> List[satellite]:
    """
    Transfer the positions of the other satellites for each satellite.

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

def state_derivative(x: np.ndarray) -> np.ndarray:
    """
    The continuous-time state derivative function, \dot{x} = f_c(x), for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :return: A numpy array of shape (6,) containing the state derivative.
    """
    r = x[:3]
    v = x[3:]
    a = -r * MU / (np.linalg.norm(r) ** 3)

    return np.concatenate([v, a])


def state_derivative_jac(x: np.ndarray) -> np.ndarray:
    """
    The continuous-time state derivative Jacobian function, d(f_c)/dx, for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :return: A numpy array of shape (6, 6) containing the state derivative Jacobian.
    """
    I = np.eye(3)
    # Account for j2 dynamics
    # j2_jac = j2_jacobian(x[0:3])

    r = x[:3]
    r_norm = np.linalg.norm(r)
    dv_dr = np.zeros((3, 3))
    da_dr = (-MU / r_norm ** 3) * np.eye(3) + (3 * MU / r_norm ** 5) * np.outer(r, r)
    dv_dv = np.eye(3)
    da_dv = np.zeros((3, 3))
    return np.block([[dv_dr, dv_dv],
                     [da_dr, da_dv]])


def RK4(x, func, dt):
    """
    Computes the state at the next timestep from the current state and the continuous-time state transition function
    using Runge-Kutta 4th order integration.

    :param x: The current state vector.
    :param func: The continuous-time state transition function, \dot{x} = f_c(x).
    :param dt: The amount of time between each time step.
    :return: The state vector at the next timestep.
    """
    k1 = func(x)
    k2 = func(x + 0.5 * dt * k1)
    k3 = func(x + 0.5 * dt * k2)
    k4 = func(x + dt * k3)

    x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


def RK4_jac(x, func, func_jac, dt):
    """
    Computes the Jacobian of the RK4-discretized state transition function.

    :param x: The current state vector.
    :param func: The continuous-time state transition function, \dot{x} = f_c(x).
    :param func_jac: The continuous-time state transition Jacobian function, d(f_c)/dx.
    :param dt: The amount of time between each time step.
    :return: The Jacobian of the RK4-discretized state transition function at the current state vector.
    """
    k1 = func(x)
    k2 = func(x + 0.5 * dt * k1)
    k3 = func(x + 0.5 * dt * k2)

    k1_jac = func_jac(x)
    k2_jac = func_jac(x + 0.5 * dt * k1) @ (np.eye(6) + 0.5 * dt * k1_jac)
    k3_jac = func_jac(x + 0.5 * dt * k2) @ (np.eye(6) + 0.5 * dt * k2_jac)
    k4_jac = func_jac(x + dt * k3) @ (np.eye(6) + dt * k3_jac)

    return np.eye(6) + (dt / 6) * (k1_jac + 2 * k2_jac + 2 * k3_jac + k4_jac)


def f(x: np.ndarray, dt: float) -> np.ndarray:
    """
    The discrete-time state transition function, x_{t+1} = f_d(x_t), for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :param dt: The amount of time between each time step.
    :return: A numpy array of shape (6,) containing the next state (position and velocity).
    """
    return RK4(x, state_derivative, dt)


def f_jac(x: np.ndarray, dt: float) -> np.ndarray:
    """
    The discrete-time state transition Jacobian function, d(f_d)/dx, for orbital position dynamics under gravity.
    J2 perturbations are not included.

    :param x: A numpy array of shape (6,) containing the current state (position and velocity).
    :param dt: The amount of time between each time step.
    :return: A numpy array of shape (6, 6) containing the state transition Jacobian.
    """
    return RK4_jac(x, state_derivative, state_derivative_jac, dt)
