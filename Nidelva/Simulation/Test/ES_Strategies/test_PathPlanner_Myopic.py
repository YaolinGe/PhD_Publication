"""
This script tests Myopic path planning
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""

from Nidelva.Simulation.ES_Strategies.PathPlanner_Myopic import MyopicPlanning
from Nidelva.Simulation.ES_Strategies.Knowledge import Knowledge
from Nidelva.Simulation.Field.Data.DataInterpolator import DataInterpolator
from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from Nidelva.Simulation.GP_kernel.Matern_kernel import Matern_Kernel

import matplotlib.pyplot as plt
from usr_func import *

polygon = np.array([[6.344800000000000040e+01, 1.040000000000000036e+01],
                   [6.344800000000000040e+01, 1.041999999999999993e+01],
                   [6.346000000000000085e+01, 1.041999999999999993e+01],
                   [6.346000000000000085e+01, 1.040000000000000036e+01]])
grid = GridGenerator(polygon = polygon, distance_neighbour = 120, no_children=6).grid
depth = [.5, 1, 1.5]
coordinates = []
for i in range(grid.shape[0]):
    for j in range(len(depth)):
        coordinates.append([grid[i, 0], grid[i, 1], depth[j]])
coordinates = np.array(coordinates)
data_interpolator = DataInterpolator(coordinates = coordinates)
dataset_interpolated = data_interpolator.dataset_interpolated

matern_kernel = Matern_Kernel(coordinates=coordinates, sill=.5, range_lateral=550, range_vertical=2, nugget=.04)

knowledge = Knowledge(coordinates=coordinates, mu=vectorise(dataset_interpolated["salinity"]),
                      Sigma=matern_kernel.Sigma, threshold_salinity=28, kernel=matern_kernel, ind_prev=0,
                      ind_now=1, distance_lateral=120, distance_vertical=1, distanceTolerance=.001)

print(knowledge.coordinates.shape)
print(knowledge.mu.shape)
print(knowledge.Sigma.shape)
print(knowledge.ind_prev)
print(knowledge.ind_now)
print(knowledge.distance_neighbours)
print(knowledge.threshold_salinity)
lat_next, lon_next, depth_next = MyopicPlanning(knowledge = knowledge).next_waypoint
print(lat_next, lon_next, depth_next)


