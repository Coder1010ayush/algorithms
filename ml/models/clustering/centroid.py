from typing import Literal

import numpy as np

from ml.utils.distance import pairwise

InitType = Literal["random", "kmeans++", "uniform"]


class Centroid:
    def __init__(self, kind: Literal["mean", "medoid", "weighted"] = "mean"):
        if kind not in ("mean", "medoid", "weighted"):
            raise ValueError(f"unsupported centroid kind {kind!r}")
        self.kind = kind

    def compute(self, points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        if self.kind == "mean":
            return points.mean(axis=0)
        if self.kind == "medoid":
            return points[np.argmin(pairwise(points, points).sum(axis=1))]
        if weights is None:
            raise ValueError("weights required for weighted centroid")
        return np.average(points, axis=0, weights=weights)


def initialize_centroids(X: np.ndarray, k: int, init: InitType = "kmeans++", rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if init == "random":
        return X[rng.choice(len(X), k, replace=False)]
    if init == "uniform":
        return np.linspace(X.min(axis=0), X.max(axis=0), k)
    if init == "kmeans++":
        centroids = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min(pairwise(X, np.array(centroids)), axis=1) ** 2
            centroids.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        return np.array(centroids)
    raise ValueError(f"unsupported init {init!r}")
