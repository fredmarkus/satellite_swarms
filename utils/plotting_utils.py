# plotting_utils.py

import matplotlib.pyplot as plt
import numpy as np


def plot_covariance_crb(crb_diag, state_dim, sat_cov_hist):
    # Plot the covariance matrix and the FIM diagonal entries.
    plt.figure()
    plt.plot(crb_diag[0::state_dim], label='x position CRB', color='red')
    plt.plot(crb_diag[1::state_dim], label='y position CRB', color='blue')
    plt.plot(crb_diag[2::state_dim], label='z position CRB', color='green')
    # plt.plot(crb_diag[3::state_dim], label='x velocity CRB', color='red')
    # plt.plot(crb_diag[4::state_dim], label='y velocity CRB', color='blue')
    # plt.plot(crb_diag[5::state_dim], label='z velocity CRB', color='green')
    plt.plot(sat_cov_hist[:,0,0], label='x position Covariance', color='red', linestyle='--')
    plt.plot(sat_cov_hist[:,1,1], label='y position Covariance', color='blue', linestyle='--')
    plt.plot(sat_cov_hist[:,2,2], label='z position Covariance', color='green', linestyle='--')
    # plt.plot(sat1_cov_hist[:,3,3], label='x velocity Covariance', color='red', linestyle='--')
    # plt.plot(sat1_cov_hist[:,4,4], label='y velocity Covariance', color='blue', linestyle='--')
    # plt.plot(sat1_cov_hist[:,5,5], label='z velocity Covariance', color='green', linestyle='--')
    plt.title('Covariance Matrix and CRB for Satellite 1 ')
    plt.xlabel('Timestep')
    # plt.ylabel('Covariance')
    plt.legend()


def plot_trajectory(x_traj, filter_position, N):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    time = np.linspace(0, 1, N)  # Normalized time from 0 to 1
    scatter1 = ax.scatter(
        x_traj[:-1,0,0], 
        x_traj[:-1,1,0], 
        x_traj[:-1,2,0],
        c=time,
        cmap='viridis',
        label='Trajectory',
        marker='o'
    )
    scatter2 = ax.scatter(
        np.mean(filter_position, axis=0)[:,0,0], 
        np.mean(filter_position, axis=0)[:,1,0],
        np.mean(filter_position, axis=0)[:,2,0], 
        c=time,
        cmap='plasma', 
        label='Filtered Position',
        marker='^'
    )
    cbar = plt.colorbar(scatter2, ax=ax, shrink=0.5, aspect=10)
    # cbar.set_label('Normalized Time')
    cbar2 = plt.colorbar(scatter1, ax=ax, shrink=0.5, aspect=10)
    # cbar2.set_label('Normalized Time')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Position Groundtruth for Satellite 1')
    plt.legend()


def plot_position_error(pos_error):
    plt.figure()
    error_x = np.abs(np.mean(pos_error,axis=0)[:,0,0])
    error_y = np.abs(np.mean(pos_error,axis=0)[:,1,0])
    error_z = np.abs(np.mean(pos_error,axis=0)[:,2,0])

    # plt.plot(filter_position[0,:,0,0], label='x position', color='red')
    # plt.plot(filter_position[0,:,1,0], label='y position', color='blue')
    # plt.plot(filter_position[0,:,2,0], label='z position', color='green')
    # plt.plot(x_traj[:,0,0], label='x position truth', color='red', linestyle='--')
    # plt.plot(x_traj[:,1,0], label='y position truth', color='blue', linestyle='--')
    # plt.plot(x_traj[:,2,0], label='z position truth', color='green', linestyle='--')

    plt.plot(error_x, label='x position error', color='red')
    plt.plot(error_y, label='y position error', color='blue')
    plt.plot(error_z, label='z position error', color='green')
    plt.title('Absolute Position Error for Satellite 1 ')
    plt.xlabel('Timestep')
    plt.legend()
plot_covariance_crb
