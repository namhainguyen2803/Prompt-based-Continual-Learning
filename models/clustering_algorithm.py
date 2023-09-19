import copy

import torch
import numpy as np

class KMeans():

    def __init__(self, num_classes=5, max_iter=1000):

        self._centroids = None
        self.num_classes = num_classes
        self.max_iter = max_iter
        self._tol = 1e-8

    def get_centroids(self):
        return self._centroids

    def _get_initial_centroids(self, features_matrix):
        permute_tensor_range = torch.randperm(features_matrix.shape[0])[:self.num_classes]
        centroids = features_matrix[permute_tensor_range]
        self._centroids = centroids.reshape(self.num_classes, features_matrix.shape[1])

    def _calculate_cluster_centers(self, features_matrix, cluster_labels):
        for i in range(0, self.num_classes):
            data = features_matrix[cluster_labels == i]
            self._centroids[i, :] = torch.mean(data, dim=0)

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

    def fit(self, features_matrix):
        self._get_initial_centroids(features_matrix)
        for i in range(0, self.max_iter):
            old_centroids = copy.deepcopy(self._centroids)
            distance = self._get_euclidean_distances(centroids=self._centroids, datapoints=features_matrix)
            cluster_labels = torch.argmin(distance, dim=0)
            self._calculate_cluster_centers(features_matrix, cluster_labels)
            if torch.max(torch.abs((self._centroids - old_centroids) / old_centroids)) < self._tol:
                break

    def predict_labels(self, features_matrix):

        if features_matrix.ndim == 1:
            features_matrix = features_matrix.unqueeze(0)

        distance = self._get_euclidean_distances(centroids=self._centroids, datapoints=features_matrix)
        cluster_labels = torch.argmin(distance, dim=0)

        return cluster_labels
