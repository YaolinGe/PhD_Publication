"""
This script generates candidate location within the range
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-22
"""
from usr_func import *


class Radar:

    def __init__(self, knowledge=None):
        self.knowledge = knowledge
        self.find_candidates_loc()
        pass

    def find_candidates_loc(self):
        delta_x, delta_y = latlon2xy(self.knowledge.coordinates[:, 0], self.knowledge.coordinates[:, 1],
                                     self.knowledge.coordinates[self.knowledge.ind_now, 0],
                                     self.knowledge.coordinates[self.knowledge.ind_now, 1])  # using the distance

        delta_z = self.knowledge.coordinates[:, 2] - self.knowledge.coordinates[self.knowledge.ind_now, 2]  # depth distance in z-direction
        self.knowledge.ind_cand = np.where(delta_x ** 2 / self.knowledge.distance_neighbours ** 2 +
                                           delta_y ** 2 / self.knowledge.distance_neighbours ** 2 +
                                           delta_z ** 2 / (self.knowledge.distance_vertical) ** 2 <= 1)[0]



