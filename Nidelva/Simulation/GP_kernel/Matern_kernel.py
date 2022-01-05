"""
This script contains GP kernel
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""

from usr_func import *


class Matern_Kernel:

    def __init__(self, coordinates, sill, range_lateral, range_vertical, nugget):
        self.sill = sill
        self.range_lateral = range_lateral
        self.range_vertical = range_vertical
        self.nugget = nugget

        self.sigma = np.sqrt(self.sill)
        self.eta = 4.5 / self.range_lateral
        self.ksi = self.range_lateral / self.range_vertical
        self.R = np.diagflat(self.nugget)

        self.coordinates = coordinates

    @property
    def Sigma(self):
        x, y = latlon2xy(self.coordinates[:, 0], self.coordinates[:, 1], 0, 0)
        x, y, z = map(vectorise, [x, y, self.coordinates[:, 2]])
        DistanceMatrix_x = x @ np.ones([1, x.shape[0]]) - np.ones([x.shape[0], 1]) @ x.T
        DistanceMatrix_y = y @ np.ones([1, y.shape[0]]) - np.ones([y.shape[0], 1]) @ y.T
        DistanceMatrix_z = z @ np.ones([1, z.shape[0]]) - np.ones([z.shape[0], 1]) @ z.T
        DistanceMatrix = np.sqrt(DistanceMatrix_x ** 2 + DistanceMatrix_y ** 2 + (self.ksi * DistanceMatrix_z) ** 2)
        Cov = self.sigma ** 2 * (1 + self.eta * DistanceMatrix) * np.exp(-self.eta * DistanceMatrix)
        return Cov


