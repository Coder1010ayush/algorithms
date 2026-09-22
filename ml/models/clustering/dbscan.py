from collections import deque

import numpy as np

from ml.base import BaseModel
from ml.utils.distance import DistanceName, pairwise


class DBSCAN(BaseModel):
    """Density-based clustering. Noise points get label -1."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: DistanceName = "euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_: np.ndarray | None = None
        self.core_points_: np.ndarray | None = None
        self.core_labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None) -> "DBSCAN":
        X = np.asarray(X, dtype=float)
        n = len(X)
        neighbours = pairwise(X, X, self.metric) <= self.eps
        is_core = neighbours.sum(axis=1) >= self.min_samples
        labels = np.full(n, -1, dtype=int)
        cluster = 0
        for i in range(n):
            if labels[i] != -1 or not is_core[i]:
                continue
            labels[i] = cluster
            queue = deque([i])
            while queue:
                j = queue.popleft()
                if not is_core[j]:
                    continue
                for k in np.flatnonzero(neighbours[j]):
                    if labels[k] == -1:
                        labels[k] = cluster
                        queue.append(k)
            cluster += 1
        self.labels_ = labels
        self.core_points_ = X[is_core]
        self.core_labels_ = labels[is_core]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.core_points_ is None:
            raise RuntimeError("call fit first")
        if len(self.core_points_) == 0:
            return np.full(len(X), -1, dtype=int)
        D = pairwise(np.asarray(X, dtype=float), self.core_points_, self.metric)
        nearest = np.argmin(D, axis=1)
        labels = self.core_labels_[nearest]
        labels[D[np.arange(len(D)), nearest] > self.eps] = -1
        return labels
