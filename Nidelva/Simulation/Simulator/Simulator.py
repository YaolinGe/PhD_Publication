"""
This script generates the next waypoint based on the current knowledge and previous path
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05 ~ 2022-01-08
"""


# from Nidelva.Simulation.Plotter.SlicerPlot import SlicerPlot, scatter_to_high_resolution, organise_plot
from Nidelva.Simulation.Plotter.KnowledgePlot import KnowledgePlot
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
import copy


np.random.seed(0)

DEPTH = [.5, 1, 1.5]
DISTANCE_LATERAL = 120
DISTANCE_VERTICAL = np.abs(DEPTH[1] - DEPTH[0])
DISTANCE_TOLERANCE = 1
# DISTANCE_NEIGHBOUR =
DISTANCE_SELF = np.abs(DEPTH[-1] - DEPTH[0])
THRESHOLD = 28


SILL = .5
RANGE_LATERAL = 550
RANGE_VERTICAL = 2
NUGGET = .04


class Simulator:

    knowledge = None

    def __init__(self, steps=10, random_seed=0):
        np.random.seed(random_seed)
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
        matern_kernel = MaternKernel(coordinates=coordinates, sill=SILL, range_lateral=RANGE_LATERAL,
                                     range_vertical=RANGE_VERTICAL, nugget=NUGGET)
        self.knowledge = Knowledge(coordinates=coordinates, polygon=self.polygon, mu=mu_prior, Sigma=matern_kernel.Sigma,
                                   threshold_salinity=THRESHOLD, kernel=matern_kernel, ind_prev=[], ind_now=[],
                                   distance_lateral=DISTANCE_LATERAL, distance_vertical=DISTANCE_VERTICAL,
                                   distance_tolerance=DISTANCE_TOLERANCE, distance_self=DISTANCE_SELF)

        self.ground_truth = np.linalg.cholesky(self.knowledge.Sigma) @ \
                            vectorise(np.random.randn(self.knowledge.coordinates.shape[0])) + self.knowledge.mu
        t2 = time.time()
        print("Simulation config is done, time consumed: ", t2 - t1)

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


    def run_2d(self):

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

            # if isEven(i):
            #     print("add")
            # self.knowledge.distance_travelled.append(self.knowledge.distance_travelled[-1] + np.abs(DEPTH[0] - DEPTH[-1]) * 2)
            self.knowledge.step_no = i

            self.knowledge = Sampler(self.knowledge, self.ground_truth, ind_sample).Knowledge
            # print("Sampling layer:", self.knowledge.coordinates[self.knowledge.ind_now, 2])
            # SlicerPlot(coordinates=self.knowledge.coordinates, value=self.knowledge.mu)
        print(self.knowledge.integratedBernoulliVariance)

    def run_3d(self):

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

    def run_lawn_mower(self):
        LawnMowerPlanningSetup = LawnMowerPlanning(knowledge=self.knowledge)
        LawnMowerPlanningSetup.build_3d_lawn_mower()

        self.lawn_mower_path_3d = LawnMowerPlanningSetup.lawn_mower_path_3d
        # self.starting_loc = self.lawn_mower_path_3d[0, :]
        # self.ind_start = get_grid_ind_at_nearest_loc(self.starting_loc, self.knowledge.coordinates)  # get nearest neighbour
        # self.knowledge.ind_prev = self.knowledge.ind_now = self.ind_start
        # self.knowledge = Sampler(self.knowledge, self.ground_truth, self.ind_start).Knowledge  # knowledge for 3d simulation

        for i in range(self.steps):
            lat_next, lon_next, depth_next = self.lawn_mower_path_3d[i, :]
            ind_sample = get_grid_ind_at_nearest_loc([lat_next, lon_next, depth_next], self.knowledge.coordinates)

            self.knowledge.step_no = i
            self.knowledge = Sampler(self.knowledge, self.ground_truth, ind_sample).Knowledge

        pass


# if __name__ == "__main__":
result_simulation_2d = SimulationResultContainer()
result_simulation_3d = SimulationResultContainer()
result_simulation_lawnmower = SimulationResultContainer()
NUMBER_STEPS = 30
NUMBER_REPLICATES = 20
for i in range(NUMBER_REPLICATES):
    t1 = time.time()
    seed = np.random.randint(NUMBER_REPLICATES)

    simulation_2d = Simulator(steps=NUMBER_STEPS, random_seed=seed)
    simulation_2d.run_2d()
    result_simulation_2d.append(simulation_2d.knowledge)

    simulation_3d = Simulator(steps=NUMBER_STEPS, random_seed=seed)
    simulation_3d.run_3d()
    result_simulation_3d.append(simulation_3d.knowledge)

    simulation_lawnmower = Simulator(steps=NUMBER_STEPS, random_seed=seed)
    simulation_lawnmower.run_lawn_mower()
    result_simulation_lawnmower.append(simulation_lawnmower.knowledge)
    t2 = time.time()
    print('Each replicate takes: ', t2 - t1)
    # SimulationResultsPlot(knowledges=[a.knowledge, b.knowledge, c.knowledge], filename="Comparison")


#%%


from usr_func import *
import matplotlib.pyplot as plt


class SimulationResultsPlot:

    def __init__(self, knowledges, filename):
        self.knowledges = knowledges
        self.filename = filename
        self.plot()

    def plot(self):
        fig = plt.figure(figsize=(20, 20))

        plt.subplot(221)
        for i in range(len(self.knowledges)):
            plt.plot(self.knowledges[i].integratedBernoulliVariance, label = str(i + 2)+"D")
        plt.title("ibv")
        plt.legend()
        plt.xlabel("iteration")

        plt.subplot(222)
        for i in range(len(self.knowledges)):
            plt.plot(self.knowledges[i].rootMeanSquaredError, label = str(i + 2)+"D")
        plt.title("rmse")
        plt.legend()
        plt.xlabel("iteration")

        plt.subplot(223)
        for i in range(len(self.knowledges)):
            plt.plot(self.knowledges[i].expectedVariance, label = str(i + 2)+"D")
        plt.title("ev")
        plt.legend()
        plt.xlabel("iteration")

        plt.subplot(224)
        for i in range(len(self.knowledges)):
            plt.plot(self.knowledges[i].distance_travelled, label=str(i + 2) + "D")
        plt.gca().set_yscale("log")
        plt.title("Distance travelled")
        plt.legend()
        plt.xlabel("iteration")
        plt.suptitle(self.filename)

        plt.show()


SimulationResultsPlot(knowledges=[simulation_2d.knowledge, simulation_3d.knowledge, simulation_lawnmower.knowledge], filename="Comparison")

#%%
t1 = np.array(result_simulation_2d.distanceTravelled)
t2 = np.array(result_simulation_3d.distanceTravelled)
t3 = np.array(result_simulation_lawnmower.distanceTravelled)
plt.boxplot(t1)
plt.boxplot(t2)
plt.boxplot(t3)
plt.gca().set_yscale("log")
plt.show()


