"""
This script generates regular grid points within a circular boundary
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-04
"""

from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
import warnings
from usr_func import *


class GridGeneratorCircularBoundary:

    def __init__(self, centre=None, radius=0, npoints=10, distance_neighbour = 0, no_children = 6):
        if centre is None:
            self.centre = [0, 0]
            raise ValueError("Circle centre is none, please check")
        if radius == 0:
            raise ValueError("Circle radius cannot be zero, please check")
        if distance_neighbour == 0:
            raise ValueError("Neighbour distance cannot be 0, please check it again")
        if no_children != 6:
            warnings.warn("Grid to be generated may not be regular")
        self.centre = centre # [lat_centre, lon_centre]
        self.radius = radius
        self.npoints = npoints
        self.distance_neighbour = distance_neighbour
        self.no_children = no_children
        self.getCircularBoundary()
        self.generateGrid()

    def getCircularBoundary(self):
        self.theta = np.linspace(0, np.pi * 2, self.npoints)
        self.x = self.radius * np.sin(self.theta)
        self.y = self.radius * np.cos(self.theta)
        self.lat_circle, self.lon_circle = xy2latlon(self.x, self.y, self.centre[0], self.centre[1])

    def generateGrid(self):
        self.circle = np.hstack((self.lat_circle.reshape(-1, 1), self.lon_circle.reshape(-1, 1)))
        self.grid = GridGenerator(polygon=self.circle, distance_neighbour=self.distance_neighbour, no_children=self.no_children).grid




