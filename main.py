"""
Main file for the codebase.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from combined_state_sim import run_combined_state_sim
from utils.config_utils import load_sat_config
from utils.data_io_utils import import_landmarks
from utils.data_io_utils import setup_data_dir
from utils.plotting_utils import plot_all_sat_crb_trace
from utils.yaml_autogen_utils import generate_satellites_yaml

def sim_setup(args: argparse.Namespace) -> None:

    bearing_dim = len(args.landmark_objects) * 3

    # Process noise covariance matrix based on paper "Autonomous orbit determination and observability analysis for formation satellites"
    # by OU Yangwei, ZHANG Hongbo, XING Jianjun page 6
    Q = np.diag(np.array([10e-6, 10e-6, 10e-6, 10e-12, 10e-12, 10e-12]))

    ### Satellite Initialization ###
    sats = load_sat_config(args=args, bearing_dim=bearing_dim)

    ind_cov = np.diag(
        np.array([1, 1, 1, 0.1, 0.1, 0.1])
    )  # Individual covariance matrix for each satellite

    sim_args = {}
    # TODO: add switch cases for different sim options
    if args.sim_type == "combined_state":
        sim_args["dt"] = 1/args.f
        sim_args["ind_cov"] = ind_cov
        sim_args["N"] = args.N
        sim_args["num_trials"] = args.num_trials
        sim_args["n_sats"] = args.n_sats
        sim_args["Q"] = Q
        sim_args["sats"] = sats
        sim_args["state_dim"] = args.state_dim
        sim_args["verbose"] = args.verbose
        
        run_combined_state_sim(sim_args)


def main(args: argparse.Namespace) -> None:

    setup_data_dir()

    if args.random_yaml:
        if not os.path.exists("config"):
            os.makedirs("config")
        generate_satellites_yaml(filename="config/sat_autogen.yaml", n_sats=args.n_sats)

    ### Landmark Initialization ###
    args.landmark_objects = import_landmarks()

    if args.run_all:
        for i in range(1, args.n_sats + 1):
            args.n_sats = i
            sim_setup(args)
    else:
        sim_setup(args)

    
    if args.sim_type == "combined_state":
        plot_all_sat_crb_trace()
    
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Nonlinear Recursive Monte Carlo Simulation"
    )
    parser.add_argument("--N", type=int, default=100, help="Number of timesteps")
    parser.add_argument(
        "--f", type=float, default=1, help="Frequency of the simulation"
    )
    parser.add_argument(
        "--ignore_earth",
        action="store_true",
        default=False,
        help="Ignore the Earth from blocking measurements. Only applies to range measurements. \
                        Bearing measurements always consider the earth.",
    )
    parser.add_argument(
        "--num_trials", type=int, default=1, help="Number of Monte Carlo trials"
    )
    parser.add_argument("--n_sats", type=int, default=1, help="Number of satellites")
    parser.add_argument(
        "--random_yaml",
        action="store_true",
        default=False,
        help="Use random satellite configuration",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        default=False,
        help="Run simulations for all number of satellites from 1 to n_sats",
    )
    parser.add_argument(
        "--state_dim", type=int, default=6, help="Dimension of the state vector"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Print information"
    )
    parser.add_argument(
        "--sim_type", type=str, default="combined-state", help="Simulation type"
    )
    args = parser.parse_args()

    main(args)
