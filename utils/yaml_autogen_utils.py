# The goal of this file is to automatically generate a yaml file that provides all necessary parameters for a satellite, using random parameters within predefined, sensible ranges.

import argparse
import os
import random
import yaml

def generate_satellites_yaml(filename: str, n_sats: int):
    
    if os.path.exists(filename):
        os.remove(filename)

    data = {}
    satellites = []


    for i in range (n_sats):


        # Generate random parameters for the satellite
        pos_cov_init = random.uniform(0.8, 1.2)
        vel_cov_init = random.uniform(0.08, 0.12)
        e = random.uniform(0.0, 0.2)
        a = random.uniform(6900.0, 6960.0)
        inc = random.uniform(1.0, 180.0)
        omega_dot = random.uniform(1.0, 360.0)
        omega = random.uniform(0.0, 180.0)
        M = random.uniform(0.0, 170.0)
        R_weight_range = random.uniform(9e-6, 9e-5)
        R_weight_bearing = random.uniform(0.1, 1)


        new_satellite = {
            "robot_id": i,
            "pos_cov_init": float(f"{pos_cov_init:.3g}"),
            "vel_cov_init": float(f"{vel_cov_init:.3g}"),
            "dim": 6,
            "landmarks": "landmark_objects",
            "orbital_elements": {
                "e": float(f"{e:.4g}"),
                "a": float(f"{a:.4g}"),
                "i": float(f"{inc:.4g}"),
                "omega_dot": float(f"{omega_dot:.4g}"),
                "omega": float(f"{omega:.4g}"),
                "M": float(f"{M:.4g}")
            },
            "R_weight_range": float(f"{R_weight_range:.4g}"),
            "R_weight_bearing": float(f"{R_weight_bearing:.4g}"),
            "camera_exists": 1,
            "camera_fov": 60.0,
            "verbose": 0,
            "ignore_earth": 1

        }

        # R_weight has been taken out because the code does not support different weights for different satellites
        satellites.append(new_satellite)

    data['satellites'] = satellites

    with open(filename, 'w') as file:
        # Use sort_keys=False if you want to preserve the order in the final YAML
        yaml.dump(data, file, sort_keys=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate a YAML file for satellite parameters')
    parser.add_argument('--n_sats', type=int, default=1, help='Number of satellites')
    args = parser.parse_args()

    generate_satellites_yaml("sat_autogen.yaml", args.n_sats)