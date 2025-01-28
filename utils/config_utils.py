"""
This file contains the method to initialize the satellite objects 
based on the configuration provided in the yaml file.
"""

from typing import List

import argparse
import yaml

#pylint: disable=import-error
from sat.core import satellite

def load_sat_config(args: argparse.Namespace, bearing_dim: int) -> List[satellite]:
    """
    Generate a list of satellite configurations based on the provided arguments 
    and YAML configuration files.

    Args:
        args (argparse.Namespace): Command-line arguments containing satellite 
            configuration parameters.
        bearing_dim (int): The dimension of the bearing measurements.

    Returns:
        List[satellite]: A list of satellite instances configured according to the provided arguments 
            and YAML files.

    Raises:
        ValueError: If the number of satellites specified in the arguments is greater than the number of satellites
                    available in the YAML configuration file and the --random_yaml flag is not set.

    Notes:
        - If the --random_yaml flag is set in the arguments, the function will load the configuration from 
          "config/sat_autogen.yaml".
        - If the --random_yaml flag is not set, the function will load the configuration from "config/config.yaml".
        - The function will only create the number of satellites specified in the arguments, 
            ignoring the rest in the YAML file.
        - Certain parameters in the YAML file will be overwritten by the values provided in the arguments.
    """

    if args.random_yaml:
        with open("config/sat_autogen.yaml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    else:
        with open("config/config.yaml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

            if len(config["satellites"]) < args.n_sats:
                raise ValueError(
                    """Number of satellites specified is greater than the number of satellites in the yaml file. 
                    Add --random_yaml flag to generate random satellite configuration for the provided number of 
                    satellites or create custom satellites in config/config.yaml"""
                )

    sats = []

    for i, sat_config in enumerate(config["satellites"]):

        # Only create the number of satellites specified in the argument. The rest of the yaml file is ignored
        if i < args.n_sats:
            # Overwrite the following yaml file parameters with values provided in this script
            sat_config["N"] = args.N
            sat_config["landmarks"] = args.landmark_objects
            sat_config["n_sats"] = args.n_sats
            sat_config["bearing_dim"] = bearing_dim
            sat_config["verbose"] = args.verbose
            sat_config["ignore_earth"] = args.ignore_earth

            satellite_inst = satellite(**sat_config)
            sats.append(satellite_inst)
        else:
            break

    return sats
