import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from math import pi
from models.ClusterAlgorithm import KMeans


class AbstractLearningDistributionMethod(ABC):
    @abstractmethod
    def learn_distribution(self, data):
        pass

    @abstractmethod
    def sample(self, num_sample):
        pass

    @abstractmethod
    def log_likelihood(self, x):
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
        cov = cov + self.EPSILON * torch.eye(cov.size(0))
        return cov

    def learn_distribution(self, data):
        self.mean = self._learn_mean(data)
        self.covariance = self._learn_covariance(data.T)
        self.dist = MultivariateNormal(loc=self.mean, covariance_matrix=self.covariance)

    def sample(self, num_sample):
        return self.dist.sample(sample_shape=(num_sample,))

    def log_likelihood(self, x):
        return self.dist.log_prob(x)


class MixtureGaussian(AbstractLearningDistributionMethod):

    def __init__(self, num_clusters=5, *args, **kwargs):
        super(MixtureGaussian, self).__init__(*args, **kwargs)
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

        pi = torch.ones(self.num_clusters).fill_(1 / self.num_clusters)
        mu = self._initialize_set_mean(data)  # (num_clusters, num_features)
        self._init_mu = copy.deepcopy(mu)
        sigma = torch.eye(num_features).unsqueeze(0).repeat(self.num_clusters, 1, 1)

        return mu, sigma, pi

    def _calculate_difference_prior_mu_and_posterior_mu(self):
        mu_diff = self.mu - self._init_mu
        print(f"Relative difference between prior mu and posterior mu: {torch.max(torch.abs(mu_diff))}")

    def log_likelihood(self, data):
        """
        Computes log-likelihood of samples under the current model.
        Parameters
        ----------
            data:          torch.Tensor (num_instances, num_features)

        Returns
        -------
            log_prob:      torch.Tensor (num_instances, num_clusters)
        """
        num_instances = data.shape[0]
        num_features = data.shape[1]
        x_minus_mu_times_sigma = torch.zeros(num_instances, self.num_clusters, num_features)
        list_det = torch.linalg.det(self.sigma)  # (num_cluster)
        sigma_inv = torch.linalg.inv(self.sigma)  # (num_cluster, num_features, num_features)
        x_minus_mu = data.unsqueeze(1) - self.mu.unsqueeze(0)  # (num_instance, num_cluster, num_features)
        for c in range(self.num_clusters):
            x_minus_mu_times_sigma[:, c, :] = torch.mm(x_minus_mu[:, c, :], sigma_inv[c, :, :])
            # (num_instance, num_features) @ (num_features, num_features) = (num_instance, num_features)
        assert x_minus_mu_times_sigma.shape == x_minus_mu.shape
        x_minus_mu_times_sigma_times_x_minus_mu = torch.sum(x_minus_mu_times_sigma * x_minus_mu, dim=-1)
        assert x_minus_mu_times_sigma_times_x_minus_mu.shape == (num_instances, self.num_clusters)

        log_det = torch.log(list_det + self.EPS).reshape(1, -1)
        log_manhattan_dist = x_minus_mu_times_sigma_times_x_minus_mu
        log_pi = num_features * np.log(2 * torch.pi)
        log_prob = (-0.5) * (log_manhattan_dist + log_det + log_pi)
        return log_prob

    def _calculate_prob_z_given_x(self, log_prob_x_and_z):
        """
        Handle numerical stability when calculating prob_z_given_x
        Parameters
        ----------
        log_prob_x_and_z         torch.Tensor (num_instances, num_clusters)

        Returns
        -------
        prob_z_given_x           torch.Tensor (num_instances, num_clusters)
        """
        max_element = torch.max(log_prob_x_and_z)
        normalized_log_prob_x_and_z = log_prob_x_and_z - max_element
        exp_normalized_log_prob_x_and_z = torch.exp(normalized_log_prob_x_and_z) + self.EPS
        prob_z_given_x = exp_normalized_log_prob_x_and_z / torch.sum(exp_normalized_log_prob_x_and_z, dim=1).reshape(-1,
                                                                                                                     1)
        return prob_z_given_x

    def log_joint_distribution(self, data):
        """
        Calculate log p(data, class)
        Parameters
        ----------
        data                    torch.Tensor (num_instances, num_features)

        Returns
        -------
        log_prob_x_and_z        torch.Tensor (num_instances, num_clusters)
        """
        log_p_x_given_z = self.log_likelihood(data)
        log_pi = torch.log(self.pi + self.EPS).reshape(1, -1)
        log_prob_x_and_z = log_p_x_given_z + log_pi
        return log_prob_x_and_z

    def log_marginal_distribution(self, data):
        """
        Calculate p(data)
        Parameters
        ----------
        data              torch.Tensor (num_instances, num_features)

        Returns
        -------
        log_prob_x        torch.Tensor (num_instances,)
        """
        return torch.sum(self.log_joint_distribution(data), dim=-1)

    def posterior_distribution(self, data):
        """
        Calculate probability of each class given data p(class|data)
        Parameters
        ----------
        data             torch.Tensor (num_instances, num_features)

        Returns
        -------
        prob_z_given_x    torch.Tensor (num_instances, num_clusters)
        """
        log_p_x_given_z = self.log_likelihood(data)
        log_pi = torch.log(self.pi + self.EPS).reshape(1, -1)
        log_prob_x_and_z = log_p_x_given_z + log_pi
        prob_z_given_x = self._calculate_prob_z_given_x(log_prob_x_and_z)
        return prob_z_given_x

    def _update_parameter(self, prob_z_given_x, data):
        """
        Update pi, mu, sigma based on posterior probability
        Parameters
        ----------
        prob_z_given_x    torch.Tensor (num_instances, num_clusters)
        data              torch.Tensor (num_instances, num_features)

        Returns
        -------
        pi                torch.Tensor (num_clusters,)
        mu                torch.Tensor (num_clusters, num_features)
        var               torch.Tensor (num_features, num_features)
        """
        num_instances = data.shape[0]
        num_features = data.shape[1]

        unormalized_pi = torch.sum(prob_z_given_x, dim=0)
        mu = torch.sum(prob_z_given_x.unsqueeze(-1) * data.unsqueeze(1), dim=0) / (
                    unormalized_pi.reshape(-1, 1) + self.EPS)

        x_minus_mu = data.unsqueeze(1) - mu.unsqueeze(0)

        # OPTION 1
        # resp = prob_z_given_x.unsqueeze(-1)
        # var_2 = torch.sum(x_minus_mu.unsqueeze(-1).matmul(x_minus_mu.unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
        #                 keepdim=True) / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1)
        # var_2 = var_2.squeeze(0)

        # OPTION 2
        var = torch.zeros(self.num_clusters, num_features, num_features)
        for c in range(self.num_clusters):
            cac = x_minus_mu[:, c, :] # (num_instances, num_features)
            p = prob_z_given_x[:, c].reshape(-1, 1)
            var[c,:,:] = torch.mm(cac.transpose(-2, -1), p*cac)

        var = var / (unormalized_pi.reshape(-1, 1, 1))
        pi = unormalized_pi / num_instances

        return pi, mu, var

    def learn_distribution(self, data, epoch=200):
        num_instances = data.shape[0]
        num_features = data.shape[1]
        self.mu, self.sigma, self.pi = self._initialize_parameters(data)

        it = 0
        old_loss = -1e9
        while it < epoch:
            it += 1
            # E step
            prob_z_given_x = self.posterior_distribution(data)

            pi, mu, var = self._update_parameter(prob_z_given_x, data)

            self.pi = pi
            self.mu = mu
            self.var = var

            assert self.mu.shape == (self.num_clusters, num_features)
            assert self.pi.shape == (self.num_clusters,)
            assert self.sigma.shape == (self.num_clusters, num_features, num_features)

            log_p_x = torch.sum(self.log_marginal_distribution(data))
            print(f"Loss function: {log_p_x}. Check if new logP(x) > old logP(x): {log_p_x > old_loss}")
            old_loss = log_p_x


        for j in range(self.num_clusters):
            gaussian = MultivariateNormal(loc=self.mu[j, :], covariance_matrix=self.sigma[j, :, :])
            self.gaussian_list.append(gaussian)

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
