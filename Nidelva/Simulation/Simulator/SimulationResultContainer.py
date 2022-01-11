

class SimulationResultContainer:

    def __init__(self):

        self.expectedIntegratedBernoulliVariance = []
        self.rootMeanSquaredError = []
        self.expectedVariance = []
        self.distanceTravelled = []

    def append(self, knowledge):
        self.expectedIntegratedBernoulliVariance.append(knowledge.integratedBernoulliVariance)
        self.rootMeanSquaredError.append(knowledge.rootMeanSquaredError)
        self.expectedVariance.append(knowledge.expectedVariance)
        self.distanceTravelled.append(knowledge.distance_travelled)

