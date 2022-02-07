"""
This script only contains the knowledge node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05 ~ 2022-01-08
"""

from usr_func import *


class Knowledge:

    def __init__(self, coordinates, polygon, mu, Sigma, threshold_salinity, kernel, ind_prev,
                 ind_now, distance_lateral, distance_vertical, distance_tolerance, distance_self):
        # knowing
        self.coordinates = coordinates
        self.polygon = polygon
        self.mu = mu
        self.Sigma = Sigma
        self.excursion_prob = None
        self.excursion_set = None

        self.ind_prev = ind_prev
        self.ind_now = ind_now
        self.distance_lateral = distance_lateral
        self.distance_vertical = distance_vertical
        self.distance_tolerance = distance_tolerance
        self.distance_neighbours = np.sqrt(distance_lateral ** 2 + distance_vertical ** 2) + distance_tolerance
        self.distance_self = distance_self
        self.threshold_salinity = threshold_salinity
        self.kernel = kernel

        # learned
        self.ind_cand = [] # save all potential candidate locations
        self.ind_cand_filtered = [] # save filtered candidate locations, [#1-No-PopUp-Dive, #2-No-Sharp-Turn]
        self.ind_next = []
        self.ind_visited = []
        self.trajectory = []
        self.step_no = 0

        # criteria
        self.integratedBernoulliVariance = []
        self.rootMeanSquaredError = []
        self.expectedVariance = []
        self.distance_travelled = [0]

