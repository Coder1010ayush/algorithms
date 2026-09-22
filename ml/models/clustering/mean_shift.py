import numpy as np

from ml.base import BaseModel
from ml.utils.distance import pairwise


def estimate_bandwidth(X: np.ndarray, quantile: float = 0.3) -> float:
    """Median over points of the distance to the `quantile` fraction nearest neighbour."""
    D = np.sort(pairwise(X, X), axis=1)
    k = max(1, int(quantile * len(X)))
    return float(np.median(D[:, min(k, len(X) - 1)]))


class MeanShift(BaseModel):
    """Flat-kernel mean shift: each seed climbs to the mean of the points within `bandwidth`."""

    def __init__(self, bandwidth: float | None = None, max_iter: int = 300, tol: float = 1e-3, bin_seeding: bool = False):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.bin_seeding = bin_seeding
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def _seeds(self, X: np.ndarray, bandwidth: float) -> np.ndarray:
        if not self.bin_seeding:
            return X.copy()
        bins = np.unique(np.round(X / bandwidth), axis=0)
        return bins * bandwidth

    def fit(self, X: np.ndarray, y=None) -> "MeanShift":
        X = np.asarray(X, dtype=float)
        bw = estimate_bandwidth(X) if self.bandwidth is None else self.bandwidth
        centers = self._seeds(X, bw)
        for _ in range(self.max_iter):
            within = pairwise(centers, X) <= bw
            counts = within.sum(axis=1)
            new = np.where(counts[:, None] > 0, within @ X / np.maximum(counts, 1)[:, None], centers)
            shift = np.linalg.norm(new - centers, axis=1).max()
            centers = new
            if shift < self.tol * bw:
                break
        self.cluster_centers_ = self._merge(centers, bw)
        self.labels_ = self.predict(X)
        return self

    @staticmethod
    def _merge(centers: np.ndarray, bw: float) -> np.ndarray:
        centers = np.unique(np.round(centers, 6), axis=0)
        keep = np.ones(len(centers), dtype=bool)
        D = pairwise(centers, centers)
        for i in range(len(centers)):
            if keep[i]:
                close = (D[i] < bw) & (np.arange(len(centers)) > i)
                keep[close] = False
        return centers[keep]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("call fit first")
        return np.argmin(pairwise(np.asarray(X, dtype=float), self.cluster_centers_), axis=1)
