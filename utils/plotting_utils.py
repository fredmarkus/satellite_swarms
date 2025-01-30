# plotting_utils.py

import matplotlib.pyplot as plt
import numpy as np
import os
import random

def plot_covariance_crb(crb_diag, state_dim, cov_hist):
    # Plot the covariance matrix and the FIM diagonal entries.
    plt.figure()
    # Ignore the first timestep to improve visualization
    plt.plot(crb_diag[6::state_dim], label='x position CRB', color='red')
    plt.plot(crb_diag[7::state_dim], label='y position CRB', color='blue')
    plt.plot(crb_diag[8::state_dim], label='z position CRB', color='green')
    # plt.plot(crb_diag[3::state_dim], label='x velocity CRB', color='red')
    # plt.plot(crb_diag[4::state_dim], label='y velocity CRB', color='blue')
    # plt.plot(crb_diag[5::state_dim], label='z velocity CRB', color='green')
    plt.plot(cov_hist[1:,0,0], label='x position Covariance', color='red', linestyle='--')
    plt.plot(cov_hist[1:,1,1], label='y position Covariance', color='blue', linestyle='--')
    plt.plot(cov_hist[1:,2,2], label='z position Covariance', color='green', linestyle='--')
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


def plot_covariance_crb_trace(crb_trace, cov_trace):
    plt.figure()
    plt.plot(crb_trace[1:], label='CRB Trace', color='red')
    plt.plot(cov_trace[1:], label='COV Trace', color='blue')
    plt.title('Covariance Matrix and CRB ')
    plt.xlabel('Timestep')
    plt.legend()

def plot_all_sat_crb_trace():

    linestyles = ["-", "--", "-.", ":"] 
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'yellow', 'pink', 'brown', 'cyan']

    directory_path = "data/trace/"

    plt.figure()
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".npy"):
            i = int(file_name[0:file_name.index("_")])
            file_path = os.path.join(directory_path, file_name)
            arr = np.load(file_path)
            if "crb" in file_name:
                plt.plot(arr[1:], label=f'CRB Trace {i} sat', color=colors[i % len(colors)], linestyle='--')
            elif "cov" in file_name:
                plt.plot(arr[1:], label=f'COV Trace {i} sat', color=colors[i % len(colors)], linestyle='-')
            
    
    plt.title('Covariance Matrix and CRB trace for different numbers of satellites')
    plt.xlabel('Timestep')
    plt.legend()


def all_sat_position_error(pos_error,n_sats,meas_type):
    linestyles = ["-", "--", "-.", ":"] 
    plt.figure()
    for i in range(n_sats):
        error = np.abs(np.sum(pos_error[:,i*3:(i+1)*3],axis=1))

        plt.plot(error, label=f'position error sat {i}', color='red',linestyle=linestyles[i % len(linestyles)])
        plt.title(f'Absolute Position Error for {n_sats} satellites for measurement type {meas_type}')
        plt.xlabel('Timestep')
        plt.legend()

def random_color():
    # Return a random hex color like '#3A2F1B'
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))
