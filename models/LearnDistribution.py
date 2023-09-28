import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

class Gaussian:

    def __init__(self):
        self.mean = None
        self.covariance = None
        self.dist = None
        self.EPSILON = 1e-4

    def get_mean(self):
        return self.mean

    def get_covariance(self):
        return self.covariance

    def _learn_mean(self, data):
        return torch.mean(data.type(torch.float64), dim=0)

    def _learn_covariance(self, data):
        cov = torch.cov(data.type(torch.float64))
        cov = cov + self.EPSILON * torch.eye(cov.size(0), dtype=torch.float64)
        return cov

    def learn_distribution(self, data):
        self.mean = self._learn_mean(data)
        self.covariance = self._learn_covariance(data.T)
        self.dist = MultivariateNormal(loc=self.mean.float(), covariance_matrix=self.covariance.float())

    def sample(self, num_sample):
        return self.dist.sample(sample_shape=(num_sample,))


def get_learning_distribution_model():
    return Gaussian()
