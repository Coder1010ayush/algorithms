from typing import Literal

import numpy as np

from ml.base import BaseModel
from ml.models.clustering.kmeans import KMeans
from ml.utils.distance import pairwise


class SpectralClustering(BaseModel):
    """Ng-Jordan-Weiss: k-means on the row-normalised top eigenvectors of the normalised affinity."""

    def __init__(
        self,
        n_clusters: int = 2,
        affinity: Literal["rbf", "nearest_neighbors"] = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        random_state: int | None = None,
    ):
        if affinity not in ("rbf", "nearest_neighbors"):
            raise ValueError(f"unsupported affinity {affinity!r}")
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.labels_: np.ndarray | None = None
        self.embedding_: np.ndarray | None = None
        self._X: np.ndarray | None = None

    def _affinity(self, X: np.ndarray) -> np.ndarray:
        D = pairwise(X, X)
        if self.affinity == "rbf":
            return np.exp(-self.gamma * D**2)
        idx = np.argsort(D, axis=1)[:, 1 : self.n_neighbors + 1]
        W = np.zeros_like(D)
        np.put_along_axis(W, idx, 1.0, axis=1)
        return np.maximum(W, W.T)

    def fit(self, X: np.ndarray, y=None) -> "SpectralClustering":
        X = np.asarray(X, dtype=float)
        W = self._affinity(X)
        np.fill_diagonal(W, 0.0)
        d = np.maximum(W.sum(axis=1), 1e-12)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        L = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
        _, vecs = np.linalg.eigh(L)
        U = vecs[:, -self.n_clusters :]
        U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        self.embedding_ = U
        km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(U)
        self.labels_ = km.labels_
        self._X = X
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Out-of-sample points take the label of their nearest training point."""
        if self._X is None:
            raise RuntimeError("call fit first")
        return self.labels_[np.argmin(pairwise(np.asarray(X, dtype=float), self._X), axis=1)]
