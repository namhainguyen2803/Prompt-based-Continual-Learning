import copy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from models.ClusterAlgorithm import KMeans


class AbstractLearningDistributionMethod(ABC):
    @abstractmethod
    def learn_distribution(self, data):
        pass

    def sample(self, num_sample):
        pass


class Gaussian(AbstractLearningDistributionMethod):

    def __init__(self):
        self.mean = None
        self.covariance = None
        self.dist = None
        self.EPSILON = 1e-4

    def _learn_mean(self, data):
        return torch.mean(data.type(torch.float64), dim=0)

    def _learn_covariance(self, data):
        cov = torch.cov(data.type(torch.float64))
        cov = cov + self.EPSILON * torch.eye(cov.size(0), dtype=torch.float64).cuda()
        return cov

    def learn_distribution(self, data):
        self.mean = self._learn_mean(data)
        self.covariance = self._learn_covariance(data.T)
        self.dist = MultivariateNormal(loc=self.mean.float(), covariance_matrix=self.covariance.float())

    def sample(self, num_sample):
        return self.dist.sample(sample_shape=(num_sample,))


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

        sigma = torch.zeros(self.num_clusters, num_features, num_features, dtype=torch.float64).cuda()
        pi = torch.ones(self.num_clusters, dtype=torch.float64).fill_(1 / self.num_clusters).cuda()
        mu = self._initialize_set_mean(data).cuda()  # (num_clusters, num_features)
        self._init_mu = copy.deepcopy(mu)
        for i in range(self.num_clusters):
            A = torch.randn(num_features, num_features).cuda()
            sigma[i, :, :] = torch.matmul(A, A.T)

        return mu, sigma, pi

    def _calculate_difference_prior_mu_and_posterior_mu(self):
        mu_diff = self.mu - self._init_mu
        print(f"Relative difference between prior mu and posterior mu: {torch.max(torch.abs(mu_diff))}")

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
                p_z_given_x[:, j] = c * torch.exp((-0.5) * torch.diag(x_mu.dot(sigma_inv).dot(x_mu.T))) * pi[j]

            p_z_given_x = p_z_given_x / torch.sum(p_z_given_x, dim=1, keepdim=True)
            # M step
            pi = (1 / self.num_clusters) * torch.sum(p_z_given_x, dim=0)  # shape == (self.num_clusters)

            for j in range(self.num_clusters):
                mu[j, :] = torch.sum(p_z_given_x[:, j].reshape(-1, 1) * data, dim=0, keepdim=True) / \
                           torch.sum(p_z_given_x[:, j], dim=0, keepdim=True)
                x_mu = data - mu[j, :].reshape(1, -1)
                sigma[j, :, :] = (p_z_given_x[:, j].reshape(-1, 1) * x_mu).T.dot(x_mu) / \
                                 torch.sum(p_z_given_x[:, j], dim=0, keepdim=True)

        assert mu.shape == (self.num_clusters, num_features)
        assert pi.shape == (self.num_clusters)
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
