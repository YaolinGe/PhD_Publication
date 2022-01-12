
import time
from usr_func import *
from Nidelva.Simulation.Simulator.Simulator import Simulator
from Nidelva.Simulation.Simulator.SimulationResultContainer import SimulationResultContainer

class SimulationReplicator:

    def __init__(self):
        self.result_simulation_2d = SimulationResultContainer("YoYo-assisted 2D Myopic Strategy")
        self.result_simulation_3d = SimulationResultContainer("3D Myopic Strategy")
        self.result_simulation_lawnmower = SimulationResultContainer("Lawn Mower Strategy")
        self.NUMBER_STEPS = 3
        self.NUMBER_REPLICATES = 1
        self.run_replicate()
        pass

    def run_replicate(self):

        for i in range(self.NUMBER_REPLICATES):
            print("Replicate: ", i)
            t1 = time.time()
            seed = i

            simulation_2d = Simulator(steps=self.NUMBER_STEPS, random_seed=seed)
            simulation_2d.run_2d()
            self.result_simulation_2d.append(simulation_2d.knowledge)

            simulation_3d = Simulator(steps=self.NUMBER_STEPS, random_seed=seed)
            simulation_3d.run_3d()
            self.result_simulation_3d.append(simulation_3d.knowledge)

            simulation_lawnmower = Simulator(steps=self.NUMBER_STEPS, random_seed=seed)
            simulation_lawnmower.run_lawn_mower()
            self.result_simulation_lawnmower.append(simulation_lawnmower.knowledge)

            t2 = time.time()
            print('Each replicate takes: ', t2 - t1)


replicator = SimulationReplicator()

