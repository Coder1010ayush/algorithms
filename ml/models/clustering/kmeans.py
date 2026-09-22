from typing import Literal

import numpy as np

from ml.base import BaseModel
from ml.models.clustering.centroid import InitType, initialize_centroids
from ml.utils.distance import DistanceName, pairwise


class KMeans(BaseModel):
    def __init__(self, n_clusters: int = 2, max_iter: int = 300, init: InitType = "kmeans++", tol: float = 1e-6, metric: DistanceName = "euclidean", random_state: int | None = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.tol = tol
        self.metric = metric
        self.rng = np.random.default_rng(random_state)
        self.centroids: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None) -> "KMeans":
        X = np.asarray(X, dtype=float)
        self.centroids = initialize_centroids(X, self.n_clusters, self.init, self.rng)
        for _ in range(self.max_iter):
            self.labels_ = np.argmin(pairwise(X, self.centroids, self.metric), axis=1)
            new = np.array([X[self.labels_ == i].mean(axis=0) if np.any(self.labels_ == i) else self.centroids[i] for i in range(self.n_clusters)])
            shift = np.linalg.norm(new - self.centroids)
            self.centroids = new
            if shift < self.tol:
                break
        self.labels_ = np.argmin(pairwise(X, self.centroids, self.metric), axis=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmin(pairwise(np.asarray(X, dtype=float), self.centroids, self.metric), axis=1)


class KMedoids(BaseModel):
    def __init__(self, n_clusters: int = 2, max_iter: int = 300, metric: DistanceName = "euclidean", random_state: int | None = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric = metric
        self.rng = np.random.default_rng(random_state)
        self.medoids: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None) -> "KMedoids":
        X = np.asarray(X, dtype=float)
        self.medoids = X[self.rng.choice(len(X), self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            self.labels_ = np.argmin(pairwise(X, self.medoids, self.metric), axis=1)
            new = self.medoids.copy()
            for i in range(self.n_clusters):
                pts = X[self.labels_ == i]
                if len(pts):
                    new[i] = pts[np.argmin(pairwise(pts, pts, self.metric).sum(axis=1))]
            if np.allclose(new, self.medoids):
                break
            self.medoids = new
        self.labels_ = np.argmin(pairwise(X, self.medoids, self.metric), axis=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmin(pairwise(np.asarray(X, dtype=float), self.medoids, self.metric), axis=1)


class AgglomerativeClustering(BaseModel):
    """Bottom-up hierarchical clustering; ``predict`` assigns by nearest cluster centre."""

    def __init__(self, n_clusters: int = 2, linkage: Literal["single", "complete", "average", "centroid"] = "single", metric: DistanceName = "euclidean"):
        if linkage not in ("single", "complete", "average", "centroid"):
            raise ValueError(f"unsupported linkage {linkage!r}")
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.labels_: np.ndarray | None = None
        self.cluster_centers_: np.ndarray | None = None

    def _link(self, a: list[int], b: list[int], D: np.ndarray, X: np.ndarray) -> float:
        block = D[np.ix_(a, b)]
        if self.linkage == "single":
            return float(block.min())
        if self.linkage == "complete":
            return float(block.max())
        if self.linkage == "average":
            return float(block.mean())
        return float(np.linalg.norm(X[a].mean(axis=0) - X[b].mean(axis=0)))

    def fit(self, X: np.ndarray, y=None) -> "AgglomerativeClustering":
        X = np.asarray(X, dtype=float)
        n = len(X)
        D = pairwise(X, X, self.metric)
        clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
        link = D.copy()
        np.fill_diagonal(link, np.inf)
        while len(clusters) > self.n_clusters:
            keys = list(clusters)
            sub = link[np.ix_(keys, keys)]
            i, j = np.unravel_index(np.argmin(sub), sub.shape)
            a, b = keys[i], keys[j]
            clusters[a].extend(clusters.pop(b))
            link[b, :] = link[:, b] = np.inf
            for k in clusters:
                if k != a:
                    link[a, k] = link[k, a] = self._link(clusters[a], clusters[k], D, X)
        self.labels_ = np.empty(n, dtype=int)
        for label, members in enumerate(clusters.values()):
            self.labels_[members] = label
        self.cluster_centers_ = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmin(pairwise(np.asarray(X, dtype=float), self.cluster_centers_, self.metric), axis=1)
