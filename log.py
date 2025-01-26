import numpy as np
import yaml

def store_sat_instance(sat_instance, filename):
    with open (filename, 'w') as file:
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
