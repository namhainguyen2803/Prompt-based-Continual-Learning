import copy

import torch
import numpy as np


class KMeans():

    def __init__(self, num_classes=5, max_iter=1000, init_times=50):

        self._centroids = None
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.init_times = init_times
        self._tol = 1e-8

    def get_centroids(self):
        return self._centroids

    def _get_initial_centroids(self, features_matrix):
        if self.init_times > 1:
            optimal_init = None
            optimal_init_purity = 1e9
            for i in range(self.init_times):
                permute_tensor_range = torch.randperm(features_matrix.shape[0])[:self.num_classes]
                centroids = features_matrix[permute_tensor_range]
                centroids = centroids.reshape(self.num_classes, features_matrix.shape[1])
                purity = self.loss_function(features_matrix, centroids)
                if purity < optimal_init_purity:
                    optimal_init_purity = purity
                    optimal_init = centroids
            return optimal_init
        else:
            permute_tensor_range = torch.randperm(features_matrix.shape[0])[:self.num_classes]
            centroids = features_matrix[permute_tensor_range]
            centroids = centroids.reshape(self.num_classes, features_matrix.shape[1])
            return centroids

    def _calculate_cluster_centers(self, features_matrix, cluster_labels):
        for i in range(0, self.num_classes):
            data = features_matrix[cluster_labels == i]
            self._centroids[i, :] = torch.mean(data, dim=0)

    def _calculate_cluster_label(self, features_matrix, centroids=None):
        centroids = self._centroids if centroids is None else centroids
        distance = self._get_euclidean_distances(centroids=centroids, datapoints=features_matrix)
        cluster_labels = torch.argmin(distance, dim=0)
        return cluster_labels

    def _get_euclidean_distances(self, centroids, datapoints):
        if centroids.ndim == 1:
            centroids = centroids.unqueeze(0)
        centroids = centroids.unsqueeze(1)
        num_centroids = centroids.shape[0]
        num_datapoints = datapoints.shape[0]
        distance = centroids - datapoints
        distance = torch.linalg.norm(distance, dim=-1)
        assert distance.shape == (num_centroids, num_datapoints), "distance.shape != (num_centroids, num_datapoints)"
        return distance

    def loss_function(self, features_matrix, centroids=None):
        purity = 0
        centroids = self._centroids if centroids is None else centroids
        distance = self._get_euclidean_distances(centroids=centroids, datapoints=features_matrix)
        cluster_labels = torch.argmin(distance, dim=0)
        for i in range(0, self.num_classes):
            data = features_matrix[cluster_labels == i]
            purity += torch.norm(data - centroids[i, :], p=2, dim=1).mean()
        return purity

    def fit(self, features_matrix, centroids=None):
        if centroids is None:
            self._centroids = self._get_initial_centroids(features_matrix)
        else:
            self._centroids = centroids

        for i in range(0, self.max_iter):
            old_centroids = copy.deepcopy(self._centroids)
            cluster_labels = self._calculate_cluster_label(features_matrix, self._centroids)
            self._calculate_cluster_centers(features_matrix, cluster_labels)
            if torch.norm((old_centroids - self._centroids), dim=1).max() < self._tol:
                break
        purity = self.loss_function(features_matrix, self._centroids)
        return purity

    def predict_labels(self, features_matrix):

        if features_matrix.ndim == 1:
            features_matrix = features_matrix.unqueeze(0)

        distance = self._get_euclidean_distances(centroids=self._centroids, datapoints=features_matrix)
        cluster_labels = torch.argmin(distance, dim=0)

        return cluster_labels


def fit_kmeans_many_times(features_matrix, fit_times=50, **kwargs):
    optimal_loss = 1e9
    optimal_model = None
    if "num_classes" in kwargs:
        num_classes = kwargs["num_classes"]
    else:
        num_classes = 5
    if "max_iter" in kwargs:
        max_iter = kwargs["max_iter"]
    else:
        max_iter = 1000
    if "init_times" in kwargs:
        init_times = kwargs["init_times"]
    else:
        init_times = 50
    for i in range(fit_times):
        model = KMeans(num_classes=num_classes, max_iter=max_iter, init_times=init_times)
        loss = model.fit(features_matrix)
        if loss < optimal_loss:
            optimal_loss = loss
            optimal_model = model
        else:
            del model
    return optimal_model, optimal_loss

