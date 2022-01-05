"""
This script only contains the knowledge node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""

import numpy as np


class Knowledge:

    def __init__(self, coordinates, mu_conditioned, Sigma_conditioned, threshold_salinity, kernel,
                 ind_prev, ind_now, distance_lateral, distance_vertical, distanceTolerance):
        self.coordinates = coordinates
        self.mu_conditioned = mu_conditioned
        self.Sigma_conditioned = Sigma_conditioned
        self.ind_prev = ind_prev
        self.ind_now = ind_now
        self.distance_neighbours = np.sqrt(distance_lateral ** 2 + distance_vertical ** 2) + distanceTolerance
        self.threshold_salinity = threshold_salinity
        self.kernel = kernel

        self.ind_cand = []
        self.ind_next = []



