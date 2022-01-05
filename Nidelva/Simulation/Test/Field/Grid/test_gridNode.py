"""
This script tests the grid generation for the polygon boundary
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""
from Nidelva.Simulation.Field.Grid.gridNode import gridNode

subGrids_len = 2
subGrids_loc = [1, 2]
grid_loc = [0]

a = gridNode(subGrids_len, subGrids_loc, grid_loc)

print(a.subGrid_len)
print(a.subGrid_loc)
print(a.grid_loc)


