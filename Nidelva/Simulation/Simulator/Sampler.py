"""
This script samples the field and returns the updated field
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""


from usr_func import *
from sklearn.metrics import mean_squared_error

class Sampler:

    def __init__(self, knowledge, ground_truth, ind_sample):
        self.knowledge = knowledge
        self.ground_truth = ground_truth
        self.ind_sample = ind_sample
        self.sample()

    def sample(self):
        F = getFVector(self.ind_sample, self.knowledge.coordinates.shape[0])
        self.knowledge.mu, self.knowledge.Sigma = \
            GPupd(mu_cond=self.knowledge.mu, Sigma_cond=self.knowledge.Sigma, F=F,
                  R=self.knowledge.kernel.R, y_sampled=self.ground_truth[self.ind_sample])
        self.knowledge.excursion_prob = EP_1D(self.knowledge.mu, self.knowledge.Sigma, self.knowledge.threshold_salinity)
        self.knowledge.trajectory.append([self.knowledge.coordinates[self.knowledge.ind_now, 0],
                                          self.knowledge.coordinates[self.knowledge.ind_now, 1],
                                          self.knowledge.coordinates[self.knowledge.ind_now, 2]])
        self.knowledge.ind_visited.append(self.knowledge.ind_now)
        self.knowledge.ind_prev = self.knowledge.ind_now
        self.knowledge.ind_now = self.ind_sample

        self.knowledge.rootMeanSquaredError.append(mean_squared_error(self.ground_truth, self.knowledge.mu, squared=False))
        self.knowledge.expectedVariance.append(np.sum(np.diag(self.knowledge.Sigma)))


    @property
    def Knowledge(self):
        return self.knowledge




