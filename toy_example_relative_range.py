#We care about the relative measurements between the robots to each other. 

#The robot now not only does the positioning update but also a measurement of the distance between itself and the other robot(s).

import numpy as np
class robot:

    def __init__(self, rob_cov_init):
        self.pos = np.array([[0],[1]])
        self.cov = rob_cov_init*np.eye(2)
