from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from math import pi


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

    def __init__(self, n_components=5, n_features=768, covariance_type="full", init_params="kmeans",
                 mu_init=None, var_init=None):

        super(MixtureGaussian, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.EPS = 1.e-4

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random"]

        self._init_params()

    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components,
                                           self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (
                self.n_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False).cuda()
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False).cuda()

        if self.covariance_type == "diag":
            if self.var_init is not None:
                # (1, k, d)
                assert self.var_init.size() == (1, self.n_components,
                                                self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (
                    self.n_components, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False).cuda()
            else:
                self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False).cuda()
        elif self.covariance_type == "full":
            if self.var_init is not None:
                # (1, k, d, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features,
                                                self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i, %i)" % (
                    self.n_components, self.n_features, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False).cuda()
            else:
                self.var = torch.nn.Parameter(
                    torch.eye(self.n_features).reshape(1, 1, self.n_features, self.n_features).repeat(1,
                                                                                                      self.n_components,
                                                                                                      1, 1),
                    requires_grad=False).cuda()

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(
            1. / self.n_components).cuda()
        self.params_fitted = False

    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x

    def bic(self, x):

        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(x, as_average=False).mean() * n + free_params * np.log(n)

        return bic

    def learn_distribution(self, data, delta=1e-3, n_iter=100, warm_start=False):

        if not warm_start and self.params_fitted:
            self._init_params()

        data = self.check_size(data)

        if self.init_params == "kmeans" and self.mu_init is None:
            mu = self.get_kmeans_mu(data, n_centers=self.n_components)
            self.mu.data = mu

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(data)
            self.log_likelihood = self.__score(data)

            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(self.log_likelihood):
                device = self.mu.device
                # When the log-likelihood assumes unbound values, reinitialize model
                self.__init__(self.n_components,
                              self.n_features,
                              covariance_type=self.covariance_type,
                              mu_init=self.mu_init,
                              var_init=self.var_init)
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    self.mu.data, = self.get_kmeans_mu(data, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True

    def predict(self, x, probs=False):

        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        else:
            return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))

    def predict_proba(self, x):

        return self.predict(x, probs=True)

    def sample(self, num_sample):

        counts = torch.distributions.multinomial.Multinomial(total_count=num_sample, probs=self.pi.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

        # Only iterate over components with non-zero counts
        for k in np.arange(self.n_components)[counts > 0]:
            if self.covariance_type == "diag":
                x_k = self.mu[0, k] + torch.randn(int(counts[k]), self.n_features, device=x.device) * torch.sqrt(
                    self.var[0, k])
            elif self.covariance_type == "full":
                d_k = torch.distributions.multivariate_normal.MultivariateNormal(self.mu[0, k], self.var[0, k])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

            x = torch.cat((x, x_k), dim=0)

        return x, y

    def score_samples(self, x):

        x = self.check_size(x)

        score = self.__score(x, as_average=False)
        return score

    def _estimate_log_prob(self, x):

        x = self.check_size(x)
        if self.covariance_type == "full":
            mu = self.mu
            var = (self.var + self.var.transpose(-2, -1)) / 2
            precision = torch.linalg.inv(var)
            precision = (precision + precision.transpose(-2, -1)) / 2
            d = x.shape[-1]
            log_2pi = d * np.log(2. * pi)
            log_det = self._calculate_log_det(precision)
            x_mu_T = (x - mu).unsqueeze(-2)
            x_mu = (x - mu).unsqueeze(-1)
            x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)
            return (-0.5) * (log_2pi - log_det + x_mu_T_precision_x_mu)
        elif self.covariance_type == "diag":
            mu = self.mu
            prec = torch.rsqrt(self.var)
            log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * prec, dim=2, keepdim=True)
            log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)
            return (-0.5) * (self.n_features * np.log(2. * pi) + log_p - log_det)

    def _calculate_log_det(self, var):

        log_det = torch.empty(size=(self.n_components,)).to(var.device)

        for k in range(self.n_components):
            log_det[k] = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(var[0, k]))).sum()

        return log_det.unsqueeze(-1)

    def _e_step(self, x):

        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):

        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.EPS
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features) * self.EPS).cuda()
            var = torch.sum((x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
                            keepdim=True) / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1) + eps
        elif self.covariance_type == "diag":
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.EPS

        pi = pi / x.shape[0]

        return pi, mu, var

    def __em(self, x):

        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x, as_average=True):

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score)

    def __update_mu(self, mu):

        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components,
                                                                    self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
            self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu

    def __update_var(self, var):

        if self.covariance_type == "full":
            assert var.size() in [(self.n_components, self.n_features, self.n_features), (
                1, self.n_components, self.n_features,
                self.n_features)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (
                self.n_components, self.n_features, self.n_features, self.n_components, self.n_features,
                self.n_features)

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [(self.n_components, self.n_features), (1, self.n_components,
                                                                         self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
                self.n_components, self.n_features, self.n_components, self.n_features)

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var

    def __update_pi(self, pi):

        assert pi.size() in [
            (1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (
            1, self.n_components, 1)

        self.pi.data = pi

    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):

        if len(x.size()) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)

        min_cost = np.inf
        center = None
        for i in range(init_times):
            tmp_center = x[np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = 0
            for c in range(n_centers):
                cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf

        while delta > min_delta:
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()

            for c in range(n_centers):
                center[c] = x[l2_cls == c].mean(dim=0)

            delta = torch.norm((center_old - center), dim=1).max()

        return center.unsqueeze(0) * (x_max - x_min) + x_min


def calculate_matmul_n_times(n_components, mat_a, mat_b):

    res = torch.zeros(mat_a.shape).cuda()

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)  # (n, 1, d)
        mat_b_i = mat_b[0, i, :, :].squeeze()  # (1, d, d)
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)  # (n,1,d,d)

    return res


def calculate_matmul(mat_a, mat_b):

    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


def get_learning_distribution_model(model_type="gaussian"):
    if model_type == "gaussian":
        return Gaussian()
    elif model_type == "gmm":
        return MixtureGaussian()
