import torch
import torch.nn as nn


class Gaussian:

    def __init__(self):
        self.mean = None
        self.covariance = None
        self.dist = None

    def get_mean(self):
        return self.mean

    def get_covariance(self):
        return self.covariance

    def _learn_mean(self, data):
        return torch.mean(data, dim=0)

    def _learn_covariance(self, data):
        return torch.cov(data)

    def learn_distribution(self, data):
        self.mean = self._learn_mean(data)
        self.covariance = self._learn_covariance(data)
        self.dist = torch.distributions.normal.Normal(loc=self.mean, scale=self.covariance)

    def sample(self, num_sample):
        return self.dist.sample(sample_shape=torch.Size([num_sample]))


def get_learning_distribution_model():
    return Gaussian()
