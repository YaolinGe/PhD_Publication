"""
This script tests Matern kernel
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""

from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from Nidelva.Simulation.GP_kernel.Matern_kernel import Matern_Kernel

import matplotlib.pyplot as plt
from usr_func import *

polygon = np.array([[6.344800000000000040e+01, 1.040000000000000036e+01],
                   [6.344800000000000040e+01, 1.041999999999999993e+01],
                   [6.346000000000000085e+01, 1.041999999999999993e+01],
                   [6.346000000000000085e+01, 1.040000000000000036e+01]])
grid = GridGenerator(polygon = polygon, distance_neighbour = 120, no_children=6).grid
depth = [1, 2, 3]
coordinates = []
for i in range(grid.shape[0]):
    for j in range(len(depth)):
        coordinates.append([grid[i, 0], grid[i, 1], depth[j]])
coordinates = np.array(coordinates)


matern_cov = Matern_Kernel(coordinates=coordinates, sill=.5, range_lateral=550, range_vertical=2, nugget=.04).Sigma


#%%
plt.imshow(matern_cov)
plt.colorbar()
plt.show()



