"""
This script generates the next waypoint based on the current knowledge and previous path
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05 ~ 2022-01-06
"""


# from Nidelva.Simulation.Plotter.SlicerPlot import SlicerPlot, scatter_to_high_resolution, organise_plot
from Nidelva.Simulation.Plotter.KnowledgePlot import KnowledgePlot
from Nidelva.Simulation.Plotter.SimulationResultsPlot import SimulationResultsPlot
from Nidelva.Simulation.ES_Strategies.PathPlanner_Myopic3D import MyopicPlanning_3D
from Nidelva.Simulation.ES_Strategies.PathPlanner_Myopic2D import MyopicPlanning_2D
from Nidelva.Simulation.ES_Strategies.Knowledge import Knowledge
from Nidelva.Simulation.Field.Data.DataInterpolator import DataInterpolator
from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from Nidelva.Simulation.GP_kernel.Matern_kernel import Matern_Kernel
from Nidelva.Simulation.Simulator.Sampler import Sampler
from usr_func import *
import time
import copy


np.random.seed(0)

DEPTH = [.5, 1, 1.5]
DISTANCE_LATERAL = 120
DISTANCE_VERTICAL = np.abs(DEPTH[1] - DEPTH[0])
DISTANCE_TOLERANCE = 1
DISTANCE_NEIGHBOUR = np.sqrt(DISTANCE_VERTICAL ** 2 + DISTANCE_LATERAL ** 2) + DISTANCE_TOLERANCE
DISTANCE_SELF = np.abs(DEPTH[-1] - DEPTH[0])
THRESHOLD = 28


SILL = .5
RANGE_LATERAL = 550
RANGE_VERTICAL = 2
NUGGET = .04


class Simulator:

    knowledge = None

    def __init__(self, steps=10):
        self.steps = steps
        self.simulation_config()
        # self.run()

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
        matern_kernel = Matern_Kernel(coordinates=coordinates, sill=SILL, range_lateral=RANGE_LATERAL,
                                      range_vertical=RANGE_VERTICAL, nugget=NUGGET)
        self.knowledge = Knowledge(coordinates=coordinates, mu=mu_prior, Sigma=matern_kernel.Sigma,
                                   threshold_salinity=THRESHOLD, kernel=matern_kernel, ind_prev=[], ind_now=[],
                                   distance_neighbour=DISTANCE_NEIGHBOUR, distance_self=DISTANCE_SELF)

        self.ground_truth = np.linalg.cholesky(self.knowledge.Sigma) @ \
                            vectorise(np.random.randn(self.knowledge.coordinates.shape[0])) + self.knowledge.mu
        t2 = time.time()
        print("Simulation config is done, time consumed: ", t2 - t1)


    def run_2d(self):
        self.starting_loc = [63.46, 10.41, 1.]
        self.ind_start = get_grid_ind_at_nearest_loc(self.starting_loc, self.knowledge.coordinates) # get nearest neighbour
        self.knowledge.ind_prev = self.knowledge.ind_now = self.ind_start

        ''' Only used for plotting true field and prior field
        self.knowledge.excursion_prob = EP_1D(self.knowledge.mu, self.knowledge.Sigma,
                                              self.knowledge.threshold_salinity)
        KnowledgePlot(knowledge=self.knowledge, vmin=16, vmax=30, filename="field_prior")
        self.knowledge.mu = self.ground_truth
        self.knowledge.excursion_prob = EP_1D(self.knowledge.mu, self.knowledge.Sigma,
                                              self.knowledge.threshold_salinity)
        KnowledgePlot(knowledge=self.knowledge, vmin=20, vmax=30, filename="field_ground_truth")
        '''

        self.knowledge = Sampler(self.knowledge, self.ground_truth, self.ind_start).Knowledge # knowledge for 2d simulation

        for i in range(self.steps):
            print("Step No. ", i)
            # print("Sampling layer:", self.knowledge.coordinates[self.knowledge.ind_now, 2])
            # print("eibv:", self.knowledge.integratedBernoulliVariance)
            # print("rmse:", self.knowledge.rootMeanSquaredError)
            # print("ev: ", self.knowledge.expectedVariance)

            ''' 2D simulation
            '''
            # KnowledgePlot(knowledge=self.knowledge, vmin=16, vmax=30, filename="Myopic/2D/field_" + str(i))
            lat_next, lon_next, depth_next = MyopicPlanning_2D(knowledge=self.knowledge).next_waypoint
            depth_next = np.mean(DEPTH)

            ind_sample = get_grid_ind_at_nearest_loc([lat_next, lon_next, depth_next], self.knowledge.coordinates)

            self.knowledge.step_no = i

            self.knowledge = Sampler(self.knowledge, self.ground_truth, ind_sample).Knowledge
            # print("Sampling layer:", self.knowledge.coordinates[self.knowledge.ind_now, 2])
            # SlicerPlot(coordinates=self.knowledge.coordinates, value=self.knowledge.mu)
        print(self.knowledge.integratedBernoulliVariance)

    def run_3d(self):
        self.starting_loc = [63.46, 10.41, 1.]
        self.ind_start = get_grid_ind_at_nearest_loc(self.starting_loc,
                                                     self.knowledge.coordinates)  # get nearest neighbour
        self.knowledge.ind_prev = self.knowledge.ind_now = self.ind_start

        ''' Only used for plotting true field and prior field
        self.knowledge.excursion_prob = EP_1D(self.knowledge.mu, self.knowledge.Sigma,
                                              self.knowledge.threshold_salinity)
        KnowledgePlot(knowledge=self.knowledge, vmin=16, vmax=30, filename="field_prior")
        self.knowledge.mu = self.ground_truth
        self.knowledge.excursion_prob = EP_1D(self.knowledge.mu, self.knowledge.Sigma,
                                              self.knowledge.threshold_salinity)
        KnowledgePlot(knowledge=self.knowledge, vmin=20, vmax=30, filename="field_ground_truth")
        '''

        self.knowledge = Sampler(self.knowledge, self.ground_truth, self.ind_start).Knowledge  # knowledge for 3d simulation

        for i in range(self.steps):
            print("Step No. ", i)
            # print("Sampling layer:", self.knowledge.coordinates[self.knowledge.ind_now, 2])
            # print("eibv:", self.knowledge.integratedBernoulliVariance)
            # print("rmse:", self.knowledge.rootMeanSquaredError)
            # print("ev: ", self.knowledge.expectedVariance)

            # KnowledgePlot(knowledge=self.knowledge, vmin=16, vmax=30, filename="Myopic/3D/field_"+str(i))
            lat_next, lon_next, depth_next = MyopicPlanning_3D(knowledge=self.knowledge).next_waypoint

            ind_sample = get_grid_ind_at_nearest_loc([lat_next, lon_next, depth_next],self.knowledge.coordinates)
            self.knowledge.step_no = i

            self.knowledge = Sampler(self.knowledge, self.ground_truth, ind_sample).Knowledge
            # print("Sampling layer:", self.knowledge.coordinates[self.knowledge.ind_now, 2])

            # SlicerPlot(coordinates=self.knowledge.coordinates, value=self.knowledge.mu)
        print(self.knowledge.integratedBernoulliVariance)


if __name__ == "__main__":
    a = Simulator(steps=30)
    a.run_2d()
    b = Simulator(steps=30)
    b.run_3d()
    SimulationResultsPlot(knowledges=[a.knowledge, b.knowledge], filename="Comparison")








