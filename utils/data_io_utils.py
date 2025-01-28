"""
Common utility functions for reading and writing data.
"""

import csv
import os
from typing import List

import numpy as np
import yaml

#pylint: disable=import-error
from landmarks.landmark import landmark
from landmarks.landmark import latlon2ecef
from sat.core import satellite


def import_landmarks() -> List[landmark]:
    """
    Imports landmark coordinates from a CSV file, converts them to ECEF coordinates,
    and initializes landmark objects.

    Returns:
        List[landmark]: A list of landmark objects with their names and ECEF coordinates.

    Raises:
        FileNotFoundError: If the landmark coordinates file is not found.
        ValueError: If there is an error reading the landmark coordinates.
    """

    landmarks = []
    try:
        with open(
            "landmarks/landmark_coordinates.csv", newline="", encoding="utf-8"
        ) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                landmarks.append(np.array([row[0], row[1], row[2], row[3]]))

        landmarks_ecef = latlon2ecef(landmarks)
        landmark_objects = []

        # Initialize the landmark objects with their correct name and the ECEF coordinates
        for landmark_obj in landmarks_ecef:
            landmark_objects.append(
                landmark(
                    x=float(landmark_obj[1]),
                    y=float(landmark_obj[2]),
                    z=float(landmark_obj[3]),
                    name=(landmark_obj[0]),
                )
            )

        return landmark_objects
    except FileNotFoundError as exc:
        raise FileNotFoundError("Landmark coordinates file not found.") from exc
    except ValueError as exc:
        raise ValueError(f"Error reading landmark coordinates: {exc}") from exc


def store_all_data(n_sats: int,
                   cov_trace: np.ndarray,
                   crb_trace: np.ndarray,
                   pos_error: np.ndarray,
                   sats: List[satellite]) -> None:
    """
    Stores the simulation data in the data directory.

    Args:
        n_sats (int): The number of satellites in the simulation.
        cov_trace (np.ndarray): The trace of the covariance matrix.
        crb_trace (np.ndarray): The trace of the CRB matrix.
        pos_error (np.ndarray): The position error.
        sats (List[satellite]): A list of satellite objects.

    Raises:
        FileNotFoundError: If the data directory cannot be created.
    """

    np.save(f"data/trace/{n_sats}_cov-trace.npy", cov_trace)
    np.save(f"data/trace/{n_sats}_crb-trace.npy", crb_trace)

    for i in range(n_sats):
        np.save(
            f"data/pos_error/pos_error_sat_{i+1}", pos_error[:, i * 3 : (i + 1) * 3]
        )
        store_sat_instance(sats[i], f"data/sat_instance/sat_{i+1}.yaml")


def store_sat_instance(sat_instance: satellite, filename: str) -> None:
    """
    Store the satellite instance in a YAML file.

    Args:
        sat_instance: The satellite instance to store.
        filename: The filename to store the satellite
    """
    with open (filename, 'w', encoding='utf-8') as file:
        sat_dict = sat_instance.__dict__
        sat_dict.pop("landmarks")
        sat_dict.pop("other_sats_pos")
        sat_dict.pop("curr_visible_landmarks")
        sat_dict.pop("HEIGHT")
        sat_dict.pop("curr_pos")
        sat_dict.pop("cov_p")
        sat_dict.pop("x_p")
        sat_dict.pop("x_m")

        for elem in sat_dict:
            if isinstance(sat_dict[elem], np.ndarray):
                sat_dict[elem] = sat_dict[elem].tolist()

        yaml.dump(sat_dict, file, indent=4)


def setup_data_dir() -> None:
    
    if os.path.exists("data"):
        os.system("rm -r ./data")

    os.makedirs("data/pos_error")
    os.makedirs("data/trace")
    os.makedirs("data/sat_instance")
