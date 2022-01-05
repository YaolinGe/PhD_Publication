"""
This script generates the next waypoint based on the current knowledge and previous path
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""


from Nidelva.Simulation.ES_Strategies.PathPlanner_Myopic import MyopicPlanning
from Nidelva.Simulation.ES_Strategies.Knowledge import Knowledge
from Nidelva.Simulation.Field.Data.DataInterpolator import DataInterpolator
from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from Nidelva.Simulation.GP_kernel.Matern_kernel import Matern_Kernel
from Nidelva.Simulation.Simulator.Sampler import Sampler
from usr_func import *


DISTANCE_LATERAL = 120
DISTANCE_TOLERANCE = .1
DEPTH = [.5, 1, 1.5]

class Simulator:

    def __init__(self, steps=10):
        print("h")
        self.steps = steps
        self.simulation_config()
        self.run()

    def simulation_config(self):
        self.polygon = np.array([[6.344800000000000040e+01, 1.040000000000000036e+01],
                                [6.344800000000000040e+01, 1.041999999999999993e+01],
                                [6.346000000000000085e+01, 1.041999999999999993e+01],
                                [6.346000000000000085e+01, 1.040000000000000036e+01]])
        grid = GridGenerator(polygon=self.polygon, distance_neighbour=DISTANCE_LATERAL, no_children=6).grid
        coordinates = []
        for i in range(grid.shape[0]):
            for j in range(len(DEPTH)):
                coordinates.append([grid[i, 0], grid[i, 1], DEPTH[j]])
        coordinates = np.array(coordinates)
        data_interpolator = DataInterpolator(coordinates=coordinates)
        dataset_interpolated = data_interpolator.dataset_interpolated
        matern_kernel = Matern_Kernel(coordinates=coordinates, sill=.5, range_lateral=550, range_vertical=2, nugget=.04)
        self.knowledge = Knowledge(coordinates=coordinates, mu_conditioned=vectorise(dataset_interpolated["salinity"]),
                                   Sigma_conditioned=matern_kernel.Sigma, threshold_salinity=28, kernel=matern_kernel,
                                   ind_prev=[], ind_now=[], distance_lateral=DISTANCE_LATERAL,
                                   distance_vertical=np.abs(DEPTH[1] - DEPTH[0]), distanceTolerance=DISTANCE_TOLERANCE)

        self.ground_truth = np.linalg.cholesky(self.knowledge.Sigma_conditioned) @ \
                            vectorise(np.random.randn(self.knowledge.coordinates.shape[0])) + self.knowledge.mu_conditioned


    def run(self):
        print("h")
        self.starting_loc = [63.45, 10.41, 0]
        self.ind_start = get_grid_ind_at_nearest_loc(self.starting_loc, self.knowledge.coordinates)
        self.knowledge.ind_prev = self.knowledge.ind_now = self.ind_start
        self.knowledge = Sampler(self.knowledge, self.ground_truth, self.ind_start).Knowledge

        for i in range(self.steps):
            print("Step No. ", i)
            lat_next, lon_next, depth_next = MyopicPlanning(knowledge=self.knowledge).next_waypoint
            ind_sample = get_grid_ind_at_nearest_loc([lat_next, lon_next, depth_next], self.knowledge.coordinates)
            self.knowledge = Sampler(self.knowledge, self.ground_truth, ind_sample).Knowledge


if __name__ == "__main__":
    a = Simulator()

#%%




