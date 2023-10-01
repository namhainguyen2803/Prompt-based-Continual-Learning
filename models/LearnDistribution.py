from models.ClusterAlgorithm import KMeans, fit_kmeans_many_times

import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from math import pi


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

    def __init__(self, num_clusters=5, covariance_type="full"):
        self.num_clusters = num_clusters
        self.EPS = 1e-4
        self.mu = None
        self.pi = None
        self.sigma = None
        self.gaussian_list = list()
        self._init_mu = None
        self.diff_threshold = 1e-5

        self.covariance_type = covariance_type
        assert self.covariance_type in ["full", "diag"]

    def _initialize_set_mean(self, data):
        k_means, loss = fit_kmeans_many_times(features_matrix=data, fit_times=50)
        # k_means = KMeans(num_classes=self.num_clusters, max_iter=1000)
        # k_means.fit(data)
        check_tensor_nan(k_means.get_centroids(), "init_mu")
        return k_means.get_centroids()

    def _initialize_parameters(self, data):
        num_instances = data.shape[0]
        num_features = data.shape[1]

        pi = torch.ones(self.num_clusters).fill_(1 / self.num_clusters)
        mu = self._initialize_set_mean(data)  # (num_clusters, num_features)
        self._init_mu = copy.deepcopy(mu)

        if self.covariance_type == "full":
            sigma = torch.eye(num_features).unsqueeze(0).repeat(self.num_clusters, 1, 1)
        else:
            sigma = torch.ones(self.num_clusters, num_features) # just store the diagonal matrix

        return mu, sigma, pi

    def _log_det(self, var):
        log_det = torch.empty(size=(self.num_clusters,))
        for k in range(self.num_clusters):
            log_det[k] = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(var[k])) + self.EPS).sum()
        return log_det

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
        if self.covariance_type == "full":
            x_minus_mu_times_sigma = torch.zeros(num_instances, self.num_clusters, num_features)

            sigma_inv = torch.linalg.inv(self.sigma)  # (num_cluster, num_features, num_features)
            check_tensor_nan(sigma_inv, "sigma_inv")

            x_minus_mu = data.unsqueeze(1) - self.mu.unsqueeze(0)  # (num_instance, num_cluster, num_features)
            for c in range(self.num_clusters):
                x_minus_mu_times_sigma[:, c, :] = torch.mm(x_minus_mu[:, c, :], sigma_inv[c, :, :])
                # (num_instance, num_features) @ (num_features, num_features) = (num_instance, num_features)
            assert x_minus_mu_times_sigma.shape == x_minus_mu.shape
            log_manhattan_dist = torch.sum(x_minus_mu_times_sigma * x_minus_mu, dim=-1)
            assert log_manhattan_dist.shape == (num_instances, self.num_clusters)

            try:
                log_det = self._log_det(self.sigma).reshape(1, -1)
            except:
                list_det = torch.linalg.det(self.sigma)  # (num_cluster,)
                log_det = torch.log(list_det + self.EPS).reshape(1, -1)
                check_tensor_nan(log_det, "log_det")
                # self.sigma = (self.sigma + self.sigma.transpose(-2, -1))/2
                # log_det = self._log_det(self.sigma).reshape(1, -1)
                # print(f"Sigma is not symmetric due to numerical stability")
                # # raise TypeError("Cannot compute log_det")

            log_pi = num_features * np.log(2 * torch.pi)
            log_prob = (-0.5) * (log_manhattan_dist + log_det + log_pi)
            check_tensor_nan(log_prob, "log_prob")

        else:
            std = torch.rsqrt(self.sigma).unsqueeze(0) # (5, 768)
            data = data.unsqueeze(1)
            mu = self.mu.unsqueeze(0)

            log_pi = num_features * np.log(2 * torch.pi)
            log_manhattan_dist = torch.sum((mu * mu + data * data - 2 * data * mu) * std, dim=-1) # (num_instances, num_clusters)
            log_det = torch.sum(torch.log(std), dim=-1) # (1, num_clusters)
            log_prob = (-0.5) * (log_pi + log_manhattan_dist - log_det)

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
        check_tensor_nan(exp_normalized_log_prob_x_and_z, "exp_normalized_log_prob_x_and_z")
        check_tensor_nan(torch.sum(exp_normalized_log_prob_x_and_z, dim=1), "exp_normalized_log_prob_x_and_z (2)")
        prob_z_given_x = exp_normalized_log_prob_x_and_z / torch.sum(exp_normalized_log_prob_x_and_z, dim=1).reshape(-1,
                                                                                                                     1)
        check_tensor_nan(prob_z_given_x, "prob_z_given_x")
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
        log_p_x_given_z = self.log_likelihood(data)  # (num_instance, num_clusters)
        log_pi = torch.log(self.pi + self.EPS).reshape(1, -1)
        log_prob_x_and_z = log_p_x_given_z + log_pi
        check_tensor_nan(log_prob_x_and_z, "log_prob_x_and_z")
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
        num_instances = data.shape[0]
        log_joint_prob = self.log_joint_distribution(data)
        log_marginal_prob = torch.logsumexp(log_joint_prob, dim=1)
        assert log_marginal_prob.shape == (num_instances,)
        return log_marginal_prob

    def posterior_distribution(self, data):
        """
        Calculate probability of each class given data
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

        unormalized_pi = torch.sum(prob_z_given_x, dim=0) + self.EPS
        mu = torch.sum(prob_z_given_x.unsqueeze(-1) * data.unsqueeze(1), dim=0) / (
            unormalized_pi.reshape(-1, 1))

        x_minus_mu = data.unsqueeze(1) - mu.unsqueeze(0)

        if self.covariance_type == "full":
            sigma = torch.zeros(self.num_clusters, num_features, num_features)
            for c in range(self.num_clusters):
                cac = x_minus_mu[:, c, :]  # (num_instances, num_features)
                p = prob_z_given_x[:, c].reshape(-1, 1)
                sigma[c, :, :] = torch.mm(cac.transpose(-2, -1), p * cac)
            sigma = sigma / (unormalized_pi.reshape(-1, 1, 1))

            # det(A + delta) > det(A) given that delta is positive definite(?)
            sigma += torch.eye(num_features).unsqueeze(0).repeat(self.num_clusters, 1, 1) * self.EPS
            assert sigma.shape == (self.num_clusters, num_features, num_features)
        else:
            # (num_instances, num_clusters, num_features)
            X = torch.sum(prob_z_given_x.unsqueeze(-1) * data.unsqueeze(1) *
                          (data.unsqueeze(1) - 2 * mu.unsqueeze(0)), dim=0) \
                / unormalized_pi.reshape(-1, 1)
            sigma = X + mu * mu + self.EPS
            assert sigma.shape == (self.num_clusters, num_features)

        pi = unormalized_pi / num_instances

        return pi, mu, sigma

    def learn_distribution(self, data, epoch=200):
        num_instances = data.shape[0]
        num_features = data.shape[1]
        mu, sigma, pi = self._initialize_parameters(data)

        it = 0
        old_loss = -1e9

        self.pi = pi
        self.mu = mu
        self.sigma = sigma

        while it < epoch:
            it += 1
            # old_pi = copy.deepcopy(self.pi)
            # old_sigma = copy.deepcopy(self.sigma)
            # old_mu = copy.deepcopy(self.mu)

            # E step
            prob_z_given_x = self.posterior_distribution(data)
            pi, mu, sigma = self._update_parameter(prob_z_given_x, data)
            self.pi = pi
            self.mu = mu
            self.sigma = sigma

            assert self.mu.shape == (self.num_clusters, num_features)
            assert self.pi.shape == (self.num_clusters,)
            if self.covariance_type == "full":
                assert self.sigma.shape == (self.num_clusters, num_features, num_features)
            else:
                assert self.sigma.shape == (self.num_clusters, num_features)

            check_tensor_nan(self.mu, "mu")
            check_tensor_nan(self.pi, "pi")
            check_tensor_nan(self.sigma, "sigma")

            log_p_x = torch.mean(self.log_marginal_distribution(data))
            diff_gap = log_p_x - old_loss

            print(f"Iteration {it}. Loss function: {log_p_x}. Check if new logP(x) > old logP(x): {diff_gap}")

            # if diff_gap < 0:
            #     self.pi = old_pi
            #     self.mu = old_mu
            #     self.sigma = old_sigma

            if diff_gap < self.diff_threshold:
                break

            old_loss = log_p_x

        for j in range(self.num_clusters):
            if self.covariance_type == "full":
                gaussian = MultivariateNormal(loc=self.mu[j, :], covariance_matrix=self.sigma[j, :, :])
            else:
                gaussian = MultivariateNormal(loc=self.mu[j, :], covariance_matrix=torch.diag(self.sigma[j, :]))
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


def check_symmetric(mat, num):
    # print(mat == mat.transpose(-2, -1))
    res = torch.sum(mat == mat.transpose(-2, -1)) / mat.numel()
    print(f"Check symmetric in {num}: {res}")


def check_tensor_nan(tensor, tensor_name="a"):
    has_nan = torch.isnan(tensor).any().item()
    if has_nan:
        raise TypeError(f"Tensor {tensor_name} is nan.")
