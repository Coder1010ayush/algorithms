# ------------------------------ utf-8 encoding -------------------------------
# this file contains all kinds of clustering algorithm such as k-means (all varients ) and density - based clustering etc
import math
import numpy as np
from models.knn_models.centroid import Centroid, InitializeCentroid
from models.basemodel import BaseModel
from scipy.spatial.distance import cdist


class KMeansClusterClassification(BaseModel):
    def __init__(
        self,
        degree: int = 2,
        max_iteration: int = 500,
        init_type: str = "uniform",
        tolerence: float = 1e-8,
        metric: str = "euclidean",
    ):
        super().__init__()
        self.degree = degree
        self.max_iteration = max_iteration
        self.init_type = init_type
        self.tolerence = tolerence
        self.metric = metric

    def forward(self, x: np.ndarray):
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
                    for i in range(self.k)
                ]
            )
            if np.linalg.norm(self.centroids - new_centroids) < self.tolerance:
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
