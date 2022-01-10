"""
This script generates matern kernel for case Nidelva
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-07
"""

from usr_func import *


class MaternKernel:

    def __init__(self, coordinatesI, coordinatesII, sill, range_lateral, range_vertical, nugget):
        self.sill = sill
        self.range_lateral = range_lateral
        self.range_vertical = range_vertical
        self.nugget = nugget

        self.sigma = np.sqrt(self.sill)
        self.eta = 4.5 / self.range_lateral # 3/2 matern
        self.ksi = self.range_lateral / self.range_vertical
        self.R = np.diagflat(self.nugget)

        self.coordinatesI = coordinatesI
        self.coordinatesII = coordinatesII

    def compute_DistanceMatrix(self):
        x1, y1 = latlon2xy(self.coordinatesI[:, 0], self.coordinatesI[:, 1], 0, 0)
        x2, y2 = latlon2xy(self.coordinatesII[:, 0], self.coordinatesII[:, 1], 0, 0)
        x1, y1, z1, x2, y2, z2 = map(vectorise, [x1, y1, self.coordinatesI[:, 2],
                                                 x2, y2, self.coordinatesII[:, 2]])
        DistanceMatrix_x = x1 @ np.ones([1, x2.shape[0]]) - np.ones([x1.shape[0], 1]) @ x2.T
        DistanceMatrix_y = y1 @ np.ones([1, y2.shape[0]]) - np.ones([y1.shape[0], 1]) @ y2.T
        DistanceMatrix_z = z1 @ np.ones([1, z2.shape[0]]) - np.ones([z1.shape[0], 1]) @ z2.T
        self.DistanceMatrix = np.sqrt(DistanceMatrix_x ** 2 + DistanceMatrix_y ** 2 + (self.ksi * DistanceMatrix_z) ** 2)

    @property
    def Sigma(self):
        self.compute_DistanceMatrix()
        Cov = self.sigma ** 2 * (1 + self.eta * self.DistanceMatrix) * np.exp(-self.eta * self.DistanceMatrix)
        return Cov



