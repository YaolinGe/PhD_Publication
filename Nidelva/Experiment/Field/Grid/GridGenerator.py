"""
This script generates the grid used during the experiment in Nidelva May, 2021
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-07
"""

from usr_func import *
import warnings


class GridGenerator:

    def __init__(self):
        self.getGrid()

    def BBox(self, lat, lon, distance, alpha):
        lat4 = deg2rad(lat)
        lon4 = deg2rad(lon)

        lat2 = lat4 + distance * np.sin(deg2rad(alpha)) / circumference * 2 * np.pi
        lat1 = lat2 + distance * np.cos(deg2rad(alpha)) / circumference * 2 * np.pi
        lat3 = lat4 + distance * np.sin(np.pi / 2 - deg2rad(alpha)) / circumference * 2 * np.pi

        lon2 = lon4 + distance * np.cos(deg2rad(alpha)) / (circumference * np.cos(lat2)) * 2 * np.pi
        lon3 = lon4 - distance * np.cos(np.pi / 2 - deg2rad(alpha)) / (circumference * np.cos(lat3)) * 2 * np.pi
        lon1 = lon3 + distance * np.cos(deg2rad(alpha)) / (circumference * np.cos(lat1)) * 2 * np.pi

        box = np.vstack((np.array([lat1, lat2, lat3, lat4]), np.array([lon1, lon2, lon3, lon4]))).T

        return rad2deg(box)

    def getCoordinates(self, box, nx, ny, distance, alpha):
        R = np.array([[np.cos(deg2rad(alpha)), np.sin(deg2rad(alpha))],
                      [-np.sin(deg2rad(alpha)), np.cos(deg2rad(alpha))]])

        lat_origin, lon_origin = box[-1, :]
        x = np.arange(nx) * distance
        y = np.arange(ny) * distance
        gridx, gridy = np.meshgrid(x, y)

        lat = np.zeros([nx, ny])
        lon = np.zeros([nx, ny])
        for i in range(nx):
            for j in range(ny):
                xnew, ynew = R @ np.vstack((gridx[i, j], gridy[i, j]))
                lat[i, j] = lat_origin + rad2deg(xnew * np.pi * 2.0 / circumference)
                lon[i, j] = lon_origin + rad2deg(ynew * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat[i, j]))))
        coordinates = np.hstack((lat.reshape(-1, 1), lon.reshape(-1, 1)))
        return coordinates

    def getGrid(self):
        warnings.warn("Only applies to case Nidelva")
        lat4, lon4 = 63.446905, 10.419426  # right bottom corner
        alpha = deg2rad(60)
        origin = [lat4, lon4]
        distance = 1000
        depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5]  # planned depth to be observed
        box = self.BBox(lat4, lon4, distance, 60)
        N1 = 25  # number of grid points along north direction
        N2 = 25  # number of grid points along east direction
        N3 = 5  # number of layers in the depth dimension
        N = N1 * N2 * N3  # total number of grid points

        XLIM = [0, distance]
        YLIM = [0, distance]
        ZLIM = [0.5, 2.5]
        x = np.linspace(XLIM[0], XLIM[1], N1)
        y = np.linspace(YLIM[0], YLIM[1], N2)
        z = np.array(depth_obs)
        grid = []
        for k in z:
            for i in x:
                for j in y:
                    grid.append([i, j, k])
        grid = np.array(grid)
        xv = grid[:, 0].reshape(-1, 1)
        yv = grid[:, 1].reshape(-1, 1)
        zv = grid[:, 2].reshape(-1, 1)
        dx = x[1] - x[0]
        grid = []
        for k in z:
            for i in x:
                for j in y:
                    grid.append([i, j, k])
        grid = np.array(grid)
        coordinates = self.getCoordinates(box, N1, N2, dx, 60)


