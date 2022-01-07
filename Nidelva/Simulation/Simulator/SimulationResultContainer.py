

class SimulationResult:

    def __init__(self, expectedIntegratedBernoulliVariance, rootMeanSquaredError, expectedVariance):

        self.expectedIntegratedBernoulliVariance = expectedIntegratedBernoulliVariance
        self.rootMeanSquaredError = rootMeanSquaredError
        self.expectedVariance = expectedVariance


