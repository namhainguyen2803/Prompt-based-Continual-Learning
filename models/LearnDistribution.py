import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from math import pi
from models.ClusterAlgorithm import KMeans


class AbstractLearningDistributionMethod(ABC, nn.Module):
    @abstractmethod
    def learn_distribution(self, data):
        pass

    @abstractmethod
    def sample(self, num_sample):
        pass

    @abstractmethod
    def score_samples(self, x):
        pass


class Gaussian(AbstractLearningDistributionMethod):

    def __init__(self):
        super(Gaussian, self).__init__()
        self.mean = None
        self.covariance = None
        self.dist = None
        self.EPSILON = 1e-4

    def _learn_mean(self, data):
        return torch.mean(data, dim=0)

    def _learn_covariance(self, data):
        cov = torch.cov(data)
        cov = cov + self.EPSILON * torch.eye(cov.size(0)).cuda()
        return cov

    def learn_distribution(self, data):
        self.mean = self._learn_mean(data)
        self.covariance = self._learn_covariance(data.T)
        self.dist = MultivariateNormal(loc=self.mean, covariance_matrix=self.covariance)

    def sample(self, num_sample):
        return self.dist.sample(sample_shape=(num_sample,))

    def score_samples(self, x):
        return self.dist.log_prob(x)

class MixtureGaussian(AbstractLearningDistributionMethod):

    def __init__(self, num_clusters=5):
        self.num_clusters = num_clusters
        self.EPS = 1e-6
        self.mu = None
        self.pi = None
        self.sigma = None
        self.gaussian_list = list()
        self._init_mu = None

    def _initialize_set_mean(self, data):
        k_means = KMeans(num_classes=self.num_clusters, max_iter=1000)
        k_means.fit(data)
        return k_means.get_centroids()

    def _initialize_parameters(self, data):
        num_instances = data.shape[0]
        num_features = data.shape[1]

        pi = torch.ones(self.num_clusters).fill_(1 / self.num_clusters).cuda()
        mu = self._initialize_set_mean(data).cuda()  # (num_clusters, num_features)
        self._init_mu = copy.deepcopy(mu)
        sigma = torch.eye(num_features).unsqueeze(0).repeat(self.num_clusters, -1, -1).cuda()

        return mu, sigma, pi

    def _calculate_difference_prior_mu_and_posterior_mu(self):
        mu_diff = self.mu - self._init_mu
        print(f"Relative difference between prior mu and posterior mu: {torch.max(torch.abs(mu_diff))}")

    def log_prob(self, x):
        pass

    def learn_distribution(self, data, epoch=200):
        num_instances = data.shape[0]
        num_features = data.shape[1]
        mu, sigma, pi = self._initialize_parameters(data)
        p_z_given_x = torch.zeros(num_instances, self.num_clusters).cuda()
        it = 0
        while it < epoch:
            it += 1
            # E step
            list_det = torch.linalg.det(sigma)
            for j in range(self.num_clusters):
                sigma_inv = torch.linalg.inv(sigma[j, :])
                sigma_det = list_det[j]
                c = 1. / torch.sqrt(sigma_det)
                x_mu = data - mu[j, :]
                p_z_given_x[:, j] = c * torch.exp((-0.5) * torch.diag(torch.matmul(torch.matmul(x_mu, sigma_inv), x_mu.T))) * pi[j]
            p_z_given_x = p_z_given_x / torch.sum(p_z_given_x, dim=1, keepdim=True)
            # M step
            pi = (1 / self.num_clusters) * torch.sum(p_z_given_x, dim=0)  # shape == (self.num_clusters)

            for j in range(self.num_clusters):
                mu[j, :] = torch.sum(p_z_given_x[:, j].reshape(-1, 1) * data, dim=0, keepdim=True) / \
                           torch.sum(p_z_given_x[:, j], dim=0, keepdim=True)
                x_mu = data - mu[j, :].reshape(1, -1)
                sigma[j, :, :] = torch.matmul((p_z_given_x[:, j].reshape(-1, 1) * x_mu).T, x_mu) / \
                                 torch.sum(p_z_given_x[:, j], dim=0, keepdim=True)

        assert mu.shape == (self.num_clusters, num_features)
        assert pi.shape == (self.num_clusters,)
        assert sigma.shape == (self.num_clusters, num_features, num_features)
        self.mu = mu
        self.pi = pi
        self.sigma = sigma
        for j in range(self.num_clusters):
            gaussian = MultivariateNormal(loc=mu[j, :], covariance_matrix=sigma[j, :, :])
            self.gaussian_list.append(gaussian)

        self._calculate_difference_prior_mu_and_posterior_mu()

    def sample(self, num_sample):
        sample_list = list()
        gaussian_decision = Categorical(probs=self.pi)
        gaussian_decision_index = gaussian_decision.sample(sample_shape=(num_sample,))
        for j in range(self.num_clusters):
            num_sample_per_cluster = torch.sum(gaussian_decision_index == j).item()
            if num_sample_per_cluster > 0:
                sample_set_per_cluster = self.gaussian_list[j].sample(sample_shape=(num_sample_per_cluster,))
                sample_list.append(sample_set_per_cluster)
        sample_list = torch.cat(sample_list, dim=0)
        return sample_list



def get_learning_distribution_model(model_type="gaussian"):
    if model_type == "gaussian":
        return Gaussian()
    elif model_type == "gmm":
        return MixtureGaussian()
