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
        self.grid_coordinates = []
        self.grid_xyz = []
        self.gridConfig = gridConfig
        self.getGridCoordinates()

    def getGridXYZ(self):
        grid_x = np.linspace(self.gridConfig.xlim[0], self.gridConfig.xlim[1], self.gridConfig.number_of_points_x)
        grid_y = np.linspace(self.gridConfig.ylim[0], self.gridConfig.ylim[1], self.gridConfig.number_of_points_y)
        grid_z = np.linspace(self.gridConfig.zlim[0], self.gridConfig.zlim[1], self.gridConfig.number_of_points_z)
        for i in range(len(grid_x)):
            for j in range(len(grid_y)):
                for k in range(len(grid_z)):
                    self.grid_xyz.append([grid_x[i], grid_y[j], grid_z[k]])
        self.grid_xyz = np.array(self.grid_xyz)

    def getGridCoordinates(self):
        self.getGridXYZ()
        RotationalMatrix = np.array([[np.cos(self.gridConfig.angle_rotation), np.sin(self.gridConfig.angle_rotation), 0],
                                    [-np.sin(self.gridConfig.angle_rotation), np.cos(self.gridConfig.angle_rotation), 0],
                                    [0, 0, 1]])
        print(RotationalMatrix)
        self.grid_xyz_rotated = (RotationalMatrix @ self.grid_xyz.T).T
        grid_lat, grid_lon = xy2latlon(self.grid_xyz_rotated[:, 0], self.grid_xyz_rotated[:, 1], self.gridConfig.lat_pivot, self.gridConfig.lon_pivot)
        self.grid_coordinates = np.hstack((vectorise(grid_lat), vectorise(grid_lon), vectorise(self.grid_xyz_rotated[:, 2])))
        self.grid_comparison = np.vstack((self.grid_xyz, self.grid_xyz_rotated))

    @property
    def xyz(self):
        return self.grid_xyz

    @property
    def coordinates(self):
        return self.grid_coordinates




