"""
This script generates the grid used during the experiment in Nidelva May, 2021
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-10
"""

from usr_func import *
import warnings


class GridGenerator:

    def __init__(self, gridConfig):
        self.coordinates_grid_wgs = []
        self.xyz_grid_usr = []
        self.gridConfig = gridConfig
        self.getGridCoordinates()

    def getGridXYZ(self):
        grid_x = np.linspace(self.gridConfig.xlim[0], self.gridConfig.xlim[1], self.gridConfig.number_of_points_x)
        grid_y = np.linspace(self.gridConfig.ylim[0], self.gridConfig.ylim[1], self.gridConfig.number_of_points_y)
        grid_z = np.linspace(self.gridConfig.zlim[0], self.gridConfig.zlim[1], self.gridConfig.number_of_points_z)
        for i in range(len(grid_x)):
            for j in range(len(grid_y)):
                for k in range(len(grid_z)):
                    self.xyz_grid_usr.append([grid_x[i], grid_y[j], grid_z[k]])
        self.xyz_grid_usr = np.array(self.xyz_grid_usr)

    def getGridCoordinates(self):
        self.getGridXYZ()
        RotationalMatrix_USR2WGS = getRotationalMatrix_USR2WGS(self.gridConfig.angle_rotation)
        self.xyz_grid_wgs = (RotationalMatrix_USR2WGS @ self.xyz_grid_usr.T).T
        grid_lat, grid_lon = xy2latlon(self.xyz_grid_wgs[:, 0], self.xyz_grid_wgs[:, 1], self.gridConfig.lat_pivot, self.gridConfig.lon_pivot)
        self.coordinates_grid_wgs = np.hstack((vectorise(grid_lat), vectorise(grid_lon), vectorise(self.xyz_grid_wgs[:, 2])))
        self.grid_comparison = np.vstack((self.xyz_grid_usr, self.xyz_grid_wgs))

    @property
    def xyz(self):
        return self.xyz_grid_usr

    @property
    def coordinates(self):
        return self.coordinates_grid_wgs




