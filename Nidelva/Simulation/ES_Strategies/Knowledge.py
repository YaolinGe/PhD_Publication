"""
This script only contains the knowledge node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""

# import numpy as np


class Knowledge:

    def __init__(self, coordinates, mu, Sigma, threshold_salinity, kernel, ind_prev,
                 ind_now, distance_neighbour, distance_self):
        self.coordinates = coordinates
        self.mu = mu
        self.Sigma = Sigma
        self.ind_prev = ind_prev
        self.ind_now = ind_now
        self.distance_neighbours = distance_neighbour
        self.distance_self = distance_self
        self.threshold_salinity = threshold_salinity
        self.kernel = kernel

        self.ind_cand = [] # save all potential candidate locations
        self.ind_cand_filtered = [] # save filtered candidate locations, [#1-No-PopUp-Dive, #2-No-Sharp-Turn]
        self.ind_next = []
        self.eibv = []
        self.trajectory = []
        self.step_no = 0





