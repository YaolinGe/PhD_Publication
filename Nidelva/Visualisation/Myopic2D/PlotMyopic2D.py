"""
This script generates myopic2d waypoint illustration
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-25
"""


from Nidelva.Visualisation.Myopic2D.ContentPlot import ContentPlot
from Nidelva.Simulation.Plotter.SimulationResultsPlot import SimulationResultsPlot
from Nidelva.Simulation.ES_Strategies.PathPlanner_Myopic3D import MyopicPlanning_3D
from Nidelva.Simulation.ES_Strategies.PathPlanner_Myopic2D import MyopicPlanning_2D
from Nidelva.Simulation.ES_Strategies.PathPlanner_Lawnmower import LawnMowerPlanning
from Nidelva.Simulation.ES_Strategies.Knowledge import Knowledge
from Nidelva.Simulation.Field.Data.DataInterpolator import DataInterpolator
from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from Nidelva.Simulation.GP_kernel.Matern_kernel import MaternKernel
from Nidelva.Simulation.Simulator.Sampler import Sampler
from Nidelva.Visualisation.Config.config import *
from Nidelva.Simulation.Simulator.SimulationResultContainer import SimulationResultContainer
from usr_func import *
import time


class Myopic2D:

    knowledge = None
    path_yoyo_ind = []
    path_yoyo_2d = []

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
                            vectorise(np.random.randn(self.knowledge.coordinates.shape[0])) + self.knowledge.mu + GROUND_OFFSET
        t2 = time.time()
        print("Simulation config is done, time consumed: ", t2 - t1)

    def run_2d(self):
        # self.ind_start = np.random.randint(0, self.knowledge.coordinates.shape[0])
        # self.starting_loc = [self.knowledge.coordinates[self.ind_start, 0],
        #                      self.knowledge.coordinates[self.ind_start, 1],
        #                      np.mean(DEPTH)] # middle layer
        self.starting_loc = [63.4489, 10.415, 1.5]
        self.ind_start = get_grid_ind_at_nearest_loc(self.starting_loc, self.knowledge.coordinates) # get nearest neighbour
        self.knowledge.ind_prev = self.knowledge.ind_now = self.ind_sample = self.ind_start
        self.path_yoyo_ind.append(self.ind_sample)

        filename = "myopic2d_offset"
        for i in range(self.steps):
            print("Steps: ", i)
            # filename = "P_{:02d}".format(i)
            self.knowledge = Sampler(self.knowledge, self.ground_truth, self.ind_sample).Knowledge
            lat_next, lon_next, depth_next = MyopicPlanning_2D(knowledge=self.knowledge).next_waypoint
            self.ind_sample = get_grid_ind_at_nearest_loc([lat_next, lon_next, depth_next], self.knowledge.coordinates)
            self.path_yoyo_ind.append(self.ind_sample)
            self.knowledge.step_no = i
            # self.get_excursion_set()
            # ContentPlot(knowledge=self.knowledge, yoyo=[], vmin=VMIN, vmax=VMAX, filename=filename, html=False)

        self.get_yoyo_2d()
        self.get_excursion_set()
        ContentPlot(knowledge=self.knowledge, yoyo=self.path_yoyo_2d, vmin=VMIN, vmax=VMAX, filename=filename, html=False)

    def get_yoyo_2d(self):
        coordinates_middle = self.get_middle_points(self.knowledge.coordinates[self.path_yoyo_ind])
        for i in range(len(coordinates_middle)):
            loc = coordinates_middle[i]
            if isEven(i):
                self.path_yoyo_2d.append([loc[0], loc[1], DEPTH[0]])
            else:
                self.path_yoyo_2d.append([loc[0], loc[1], DEPTH[-1]])

    def get_middle_points(self, coordinates):
        coordinates_shifted = np.roll(coordinates, -1, axis=0)
        coordinates_middle = (coordinates_shifted + coordinates) / 2
        return coordinates_middle[:-1]

    def get_excursion_set(self):
        self.knowledge.excursion_set = np.zeros_like(self.knowledge.mu)
        self.knowledge.excursion_set[self.knowledge.mu < self.knowledge.threshold_salinity] = True

# seed = np.random.randint(100)
seed = 0
a = Myopic2D(steps=20, random_seed=seed)
a.run_2d()
print("seed: ", seed)


