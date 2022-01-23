"""
This script visualises SINMOD data for the paper
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-06
"""

from Nidelva.Simulation.ES_Strategies.Knowledge import Knowledge
from Nidelva.Simulation.Field.Data.DataInterpolator import DataInterpolator
from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from Nidelva.Simulation.GP_kernel.Matern_kernel import MaternKernel
from Nidelva.Visualisation.Plot3D import Plot3D
from Nidelva.Visualisation.Plot2D import Plot2D
from usr_func import *
import time


# ==== Field Config ====
DEPTH = [.5, 1.5, 2.5, 3.5]
DISTANCE_LATERAL = 120
DISTANCE_VERTICAL = np.abs(DEPTH[1] - DEPTH[0])
DISTANCE_TOLERANCE = 1
DISTANCE_SELF = np.abs(DEPTH[-1] - DEPTH[0])
THRESHOLD = 28
# ==== End Field Config ====

# ==== GP Config ====
SILL = .5
RANGE_LATERAL = 550
RANGE_VERTICAL = 2
NUGGET = .04
# ==== End GP Config ====

# ==== Plot Config ======
VMIN = 16
VMAX = 28
# ==== End Plot Config ==


class SINMODVisualiser:

    knowledge = None

    def __init__(self):
        self.config_visualisation()
        # self.save_benchmark_figure()

    def config_visualisation(self):
        t1 = time.time()
        self.polygon = np.array([[6.344800000000000040e+01, 1.040000000000000036e+01],
                                [6.344800000000000040e+01, 1.041999999999999993e+01],
                                [6.346000000000000085e+01, 1.041999999999999993e+01],
                                [6.346000000000000085e+01, 1.040000000000000036e+01]])
        gridGenerator = GridGenerator(polygon=self.polygon, depth=DEPTH, distance_neighbour=DISTANCE_LATERAL, no_children=6)
        # grid = gridGenerator.grid
        coordinates = gridGenerator.coordinates
        data_interpolator = DataInterpolator(coordinates=coordinates)
        mu_prior = vectorise(data_interpolator.dataset_interpolated["salinity"])
        matern_kernel = MaternKernel(coordinates=coordinates, sill=SILL, range_lateral=RANGE_LATERAL,
                                     range_vertical=RANGE_VERTICAL, nugget=NUGGET)
        self.knowledge = Knowledge(coordinates=coordinates, polygon=self.polygon, mu=mu_prior, Sigma=matern_kernel.Sigma,
                                   threshold_salinity=THRESHOLD, kernel=matern_kernel, ind_prev=[], ind_now=[],
                                   distance_lateral=DISTANCE_LATERAL, distance_vertical=DISTANCE_VERTICAL,
                                   distance_tolerance=DISTANCE_TOLERANCE, distance_self=DISTANCE_SELF)

        # self.ground_truth = np.linalg.cholesky(self.knowledge.Sigma) @ \
        #                     vectorise(np.random.randn(self.knowledge.coordinates.shape[0])) + self.knowledge.mu

        t2 = time.time()
        print("Simulation config is done, time consumed: ", t2 - t1)

    def plot_sinmod_raw(self):
        self.knowledge_prior = self.knowledge
        # self.knowledge_prior.excursion_prob = EP_1D(self.knowledge_prior.mu, self.knowledge_prior.Sigma, self.knowledge_prior.threshold_salinity)
        Plot3D(knowledge=self.knowledge_prior, vmin=VMIN, vmax=VMAX, filename="SINMOD3D", html=True)

    def plot_sinmod_2d(self):
        Plot2D(knowledge=self.knowledge, vmin=VMIN, vmax=VMAX, filename="SINMOD2D")


sinmod_visualiser = SINMODVisualiser()
# sinmod_visualiser.plot_sinmod_raw()
sinmod_visualiser.plot_sinmod_2d()

## === Prepare SINMOD ===




