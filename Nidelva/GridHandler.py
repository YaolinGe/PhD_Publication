import math
from usr_func import *
import numpy as np
import matplotlib.pyplot as plt

class GridHandler:
    lat4, lon4 = 63.446905, 10.419426  # right bottom corner
    origin = [lat4, lon4]
    distance = 1000 # distance along each direction
    depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5]  # planned depth to be observed
    nx = 25  # number of grid points along x-direction
    ny = 25  # number of grid points along y-direction
    L = 1000  # distance of the square
    alpha = 60  # angle of the inclined grid
    distance_depth = depth_obs[1] - depth_obs[0]
    R = np.array([[np.cos(math.radians(alpha)), np.sin(math.radians(alpha))],
                  [-np.sin(math.radians(alpha)), np.cos(math.radians(alpha))]])

    def __init__(self):
        print("Here is the rectangular grid generator")
        self.generate_grid()

    def generate_grid(self):
        x = np.linspace(0, self.L, self.nx)
        y = np.linspace(0, self.L, self.ny)
        self.grid = []
        self.grid_plot = []
        for i in range(self.nx):
            for j in range(self.ny):
                xtemp, ytemp = self.R @ np.array([x[i], y[j]])
                lat, lon = xy2latlon(xtemp, ytemp, self.origin[0], self.origin[1])
                for k in self.depth_obs:
                    self.grid.append([lat, lon, k])
                    self.grid_plot.append([x[i], y[j], k])
        self.grid = np.array(self.grid)
        self.grid_plot = np.array(self.grid_plot)
        np.savetxt("Nidelva/Config/grid.txt", self.grid, delimiter=", ")
        np.savetxt("Nidelva/Config/grid_plot.txt", self.grid_plot, delimiter=", ")

if __name__ == "__main__":
    a = GridHandler()


