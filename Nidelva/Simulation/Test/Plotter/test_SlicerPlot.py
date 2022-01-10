"""
This script tests 2D slice plot
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-06
"""
import numpy as np

from Nidelva.Simulation.Plotter.SlicerPlot import SlicerPlot, scatter_to_high_resolution, organise_plot
from Nidelva.Simulation.ES_Strategies.PathPlanner_Myopic import MyopicPlanning
from Nidelva.Simulation.ES_Strategies.Knowledge import Knowledge
from Nidelva.Simulation.Field.Data.DataInterpolator import DataInterpolator
from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from Nidelva.Simulation.GP_kernel.Matern_kernel import MaternKernel
from Nidelva.Simulation.Simulator.Sampler import Sampler
from usr_func import *
import time


np.random.seed(0)
DEPTH = [.5, 1, 1.5]
DISTANCE_LATERAL = 120
DISTANCE_VERTICAL = np.abs(DEPTH[1] - DEPTH[0])
DISTANCE_TOLERANCE = .1
THRESHOLD = 28


SILL = .5
RANGE_LATERAL = 1500
RANGE_VERTICAL = 20
NUGGET = .04

''' BACKUP VALUES
SILL = .5
RANGE_LATERAL = 550
RANGE_VERTICAL = 2
NUGGET = .04
'''

t1 = time.time()
polygon = np.array([[6.344800000000000040e+01, 1.040000000000000036e+01],
                         [6.344800000000000040e+01, 1.041999999999999993e+01],
                         [6.346000000000000085e+01, 1.041999999999999993e+01],
                         [6.346000000000000085e+01, 1.040000000000000036e+01]])
gridGenerator = GridGenerator(polygon=polygon, depth=DEPTH, distance_neighbour=DISTANCE_LATERAL, no_children=6)
# grid = gridGenerator.grid
coordinates = gridGenerator.coordinates
data_interpolator = DataInterpolator(coordinates=coordinates)
mu_prior = vectorise(data_interpolator.dataset_interpolated["salinity"])
matern_kernel = MaternKernel(coordinates=coordinates, sill=SILL, range_lateral=RANGE_LATERAL,
                             range_vertical=RANGE_VERTICAL, nugget=NUGGET)
knowledge = Knowledge(coordinates=coordinates, mu=mu_prior,
                           Sigma=matern_kernel.Sigma, threshold_salinity=THRESHOLD, kernel=matern_kernel,
                           ind_prev=[], ind_now=[], distance_lateral=DISTANCE_LATERAL,
                           distance_vertical=DISTANCE_VERTICAL, distanceTolerance=DISTANCE_TOLERANCE)

ground_truth = np.linalg.cholesky(knowledge.Sigma) @ \
                    vectorise(np.random.randn(knowledge.coordinates.shape[0])) + knowledge.mu
t2 = time.time()

SlicerPlot(coordinates=coordinates, value=ground_truth)
print("Finished plotting")
# SlicerPlot(coordinates=coordinates, value=mu_prior)
# print("Finished plotting")






