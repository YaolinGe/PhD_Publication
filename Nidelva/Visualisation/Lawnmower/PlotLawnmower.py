"""
This script generates lawnmower waypoint illustration
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-25
"""


from Nidelva.Visualisation.Lawnmower.ContentPlot import ContentPlot
from Nidelva.Simulation.Plotter.SimulationResultsPlot import SimulationResultsPlot
from Nidelva.Simulation.ES_Strategies.PathPlanner_Myopic3D import MyopicPlanning_3D
from Nidelva.Simulation.ES_Strategies.PathPlanner_Myopic2D import MyopicPlanning_2D
from Nidelva.Simulation.ES_Strategies.PathPlanner_Lawnmower import LawnMowerPlanning
from Nidelva.Simulation.ES_Strategies.Knowledge import Knowledge
from Nidelva.Simulation.Field.Data.DataInterpolator import DataInterpolator
from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from Nidelva.Simulation.GP_kernel.Matern_kernel import MaternKernel
from Nidelva.Simulation.Simulator.Sampler import Sampler
from Nidelva.Simulation.Simulator.SimulationResultContainer import SimulationResultContainer
from usr_func import *
import time


# ==== Field Config ====
DEPTH = [.5, 1, 1.5, 2.0, 2.5]
DISTANCE_LATERAL = 120
DISTANCE_VERTICAL = np.abs(DEPTH[1] - DEPTH[0])
DISTANCE_TOLERANCE = 1
DISTANCE_SELF = 20
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


class Lawnmower:

    knowledge = None
    trajectory = []

    def __init__(self, steps=10, random_seed=0):
        self.seed = random_seed
        np.random.seed(self.seed)
        self.steps = steps
        self.simulation_config()

    def simulation_config(self):
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
        self.ground_truth = np.linalg.cholesky(self.knowledge.Sigma) @ \
                            vectorise(np.random.randn(self.knowledge.coordinates.shape[0])) + self.knowledge.mu
        LawnMowerPlanningSetup = LawnMowerPlanning(knowledge=self.knowledge)
        LawnMowerPlanningSetup.build_3d_lawn_mower()
        self.lawn_mower_path_3d = LawnMowerPlanningSetup.lawn_mower_path_3d
        self.starting_index = 0
        t2 = time.time()
        print("Simulation config is done, time consumed: ", t2 - t1)

    def run_lawn_mower(self):
        lat_start, lon_start, depth_start = self.lawn_mower_path_3d[self.starting_index, :]
        ind_start = get_grid_ind_at_nearest_loc([lat_start, lon_start, depth_start], self.knowledge.coordinates)
        self.knowledge.ind_prev = self.knowledge.ind_now = ind_start
        self.trajectory.append([lat_start, lon_start, depth_start])

        for i in range(self.steps):
            print("Step No. ", i)
            lat_next, lon_next, depth_next = self.lawn_mower_path_3d[self.starting_index + i, :]
            self.trajectory.append([lat_next, lon_next, depth_next])
            ind_sample = get_grid_ind_at_nearest_loc([lat_next, lon_next, depth_next], self.knowledge.coordinates)

            self.knowledge.step_no = i
            self.knowledge = Sampler(self.knowledge, self.ground_truth, ind_sample).Knowledge

            self.get_excursion_set()
            ContentPlot(knowledge=self.knowledge, trajectory=self.trajectory, vmin=VMIN, vmax=VMAX, filename="P_{:02d}".format(i), html=False)

    def get_excursion_set(self):
        self.knowledge.excursion_set = np.zeros_like(self.knowledge.mu)
        self.knowledge.excursion_set[self.knowledge.mu < self.knowledge.threshold_salinity] = True

a = Lawnmower(steps=20)
a.run_lawn_mower()



