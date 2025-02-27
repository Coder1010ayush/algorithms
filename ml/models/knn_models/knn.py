# ------------------------------ utf-8 encoding -------------------------------
# this file contains all kinds of clustering algorithm such as k-means (all varients ) and density - based clustering etc
import math
import numpy as np
from models.knn_models.centroid import Centroid, InitializeCentroid
from models.basemodel import BaseModel
from scipy.spatial.distance import cdist
from typing import Literal


class KMeansClusterClassification(BaseModel):

    def __init__(
        self,
        degree: int = 2,
        max_iteration: int = 500,
        init_type: Literal["uniform", "kmeans++", "random"] = "uniform",
        tolerence: float = 1e-8,
        metric: str = "euclidean",
    ):
        super().__init__()
        self.degree = degree
        self.max_iters = max_iteration
        self.init_type = init_type
        self.tolerence = tolerence
        self.metric = metric

    def forward(self, x: np.ndarray, y):
        centroid_initializer = InitializeCentroid(
            X=x, k=self.degree, init_type=self.init_type
        )
        self.centroids = centroid_initializer.centroids
        iteration = 0
        while True:
            distances = cdist(x, self.centroids, metric=self.metric)
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.array(
                [
                    (
                        x[self.labels == i].mean(axis=0)
                        if np.any(self.labels == i)
                        else self.centroids[i]
                    )
                    for i in range(self.degree)
                ]
            )
            if np.linalg.norm(self.centroids - new_centroids) < self.tolerence:
                break
            self.centroids = new_centroids
            iteration += 1
            if self.max_iters is not None and iteration >= self.max_iters:
                break
        return self.labels

    def predict(self, x: np.ndarray):
        distances = cdist(x, self.centroids, metric=self.metric)
        return np.argmin(distances, axis=1)

    def compute_loss(self, y, y_pred):
        pass

    def compute_gradient(self, X, y, y_pred):
        pass

    def update_parameters(self, grad_w, grad_b, lr):
        pass


class KMedoids:
    def __init__(self, degree: int, max_iters: int = None, metric: str = "euclidean"):
        self.degree = degree
        self.max_iters = max_iters
        self.metric = metric
        self.medoids = None
        self.labels = None

    def forward(self, X: np.ndarray, y):
        n_samples = X.shape[0]
        medoid_indices = np.random.choice(n_samples, self.degree, replace=False)
        self.medoids = X[medoid_indices]
        iteration = 0

        while True:
            distances = cdist(X, self.medoids, metric=self.metric)
            self.labels = np.argmin(distances, axis=1)
            new_medoids = self.medoids.copy()
            for i in range(self.degree):
                cluster_points = X[self.labels == i]
                if len(cluster_points) == 0:
                    continue
                pairwise_distances = cdist(
                    cluster_points, cluster_points, metric=self.metric
                )
                total_distances = pairwise_distances.sum(axis=1)
                new_medoids[i] = cluster_points[np.argmin(total_distances)]
            if np.all(new_medoids == self.medoids):
                break
            self.medoids = new_medoids
            iteration += 1
            if self.max_iters is not None and iteration >= self.max_iters:
                break
        return self.labels

    def predict(self, X: np.ndarray):
        distances = cdist(X, self.medoids, metric=self.metric)
        return np.argmin(distances, axis=1)

    def compute_loss(self, y, y_pred):
        pass

    def compute_gradient(self, X, y, y_pred):
        pass

    def update_parameters(self, grad_w, grad_b, lr):
        pass


# this is not my complete work for aglomerative , i have taken help from my semester prof.
class AgglomerativeClustering(BaseModel):
    def __init__(
        self,
        n_clusters=2,
        linkage: Literal["single", "average", "complete", "centroid"] = "single",
        metric="euclidean",
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.labels_ = None
        self.cluster_centers_ = None
        self.X_train = None

    def forward(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        clusters = {i: [i] for i in range(n_samples)}
        distance_matrix = cdist(X, X, metric=self.metric)
        np.fill_diagonal(distance_matrix, np.inf)

        while len(clusters) > self.n_clusters:
            i, j = self._find_closest_clusters(distance_matrix, clusters)
            self._merge_clusters(i, j, clusters)
            self._update_distances(i, j, clusters, X, distance_matrix)
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, indices in enumerate(clusters.values()):
            self.labels_[indices] = cluster_id
        self.cluster_centers_ = np.array(
            [X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)]
        )

    def predict(self, X_test):
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fit before predicting.")
        distances = cdist(X_test, self.cluster_centers_, metric=self.metric)
        return np.argmin(distances, axis=1)

    def _find_closest_clusters(self, distance_matrix, clusters):
        min_dist = np.inf
        best_pair = None
        cluster_indices = list(clusters.keys())
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                c1, c2 = cluster_indices[i], cluster_indices[j]
                dist = self._compute_cluster_distance(
                    clusters[c1], clusters[c2], distance_matrix
                )
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (c1, c2)
        return best_pair

    def _compute_cluster_distance(self, cluster1, cluster2, distance_matrix):
        if self.linkage == "single":
            return np.min([distance_matrix[i, j] for i in cluster1 for j in cluster2])
        elif self.linkage == "complete":
            return np.max([distance_matrix[i, j] for i in cluster1 for j in cluster2])
        elif self.linkage == "average":
            return np.mean([distance_matrix[i, j] for i in cluster1 for j in cluster2])
        elif self.linkage == "centroid":
            centroid1 = np.mean([self.X_train[i] for i in cluster1], axis=0)
            centroid2 = np.mean([self.X_train[j] for j in cluster2], axis=0)
            return np.linalg.norm(centroid1 - centroid2)
        else:
            raise ValueError("Invalid linkage method")

    def _merge_clusters(self, i, j, clusters):
        clusters[i].extend(clusters[j])
        del clusters[j]

    def _update_distances(self, i, j, clusters, X, distance_matrix):
        for k in clusters.keys():
            if k != i:
                distance_matrix[i, k] = distance_matrix[k, i] = (
                    self._compute_cluster_distance(
                        clusters[i], clusters[k], distance_matrix
                    )
                )
        distance_matrix[j, :] = np.inf
        distance_matrix[:, j] = np.inf

    def compute_loss(self, y, y_pred):
        pass

    def compute_gradient(self, X, y, y_pred):
        pass

    def update_parameters(self, grad_w, grad_b, lr):
        pass

    # def predict(self, X_test):
    #     distances = cdist(X_test, self.X_train, metric=self.metric)
    #     nearest_neighbors = np.argmin(distances, axis=1)
    #     return self.labels_[nearest_neighbors]
