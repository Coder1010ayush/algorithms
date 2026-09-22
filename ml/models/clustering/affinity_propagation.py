import numpy as np

from ml.base import BaseModel
from ml.utils.distance import pairwise


class AffinityPropagation(BaseModel):
    """Message passing between points to elect exemplars; the number of clusters is not fixed."""

    def __init__(self, damping: float = 0.5, max_iter: int = 200, convergence_iter: int = 15, preference: float | None = None):
        if not 0.5 <= damping < 1.0:
            raise ValueError("damping must be in [0.5, 1)")
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.cluster_centers_indices_: np.ndarray | None = None
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None) -> "AffinityPropagation":
        X = np.asarray(X, dtype=float)
        n = len(X)
        S = -pairwise(X, X) ** 2
        pref = np.median(S[~np.eye(n, dtype=bool)]) if self.preference is None else self.preference
        np.fill_diagonal(S, pref)
        R = np.zeros((n, n))
        A = np.zeros((n, n))
        idx = np.arange(n)
        stable = 0
        prev = None
        for _ in range(self.max_iter):
            AS = A + S
            top = np.argmax(AS, axis=1)
            best = AS[idx, top]
            AS[idx, top] = -np.inf
            second = AS.max(axis=1)
            R_new = S - best[:, None]
            R_new[idx, top] = S[idx, top] - second
            R = self.damping * R + (1 - self.damping) * R_new

            Rp = np.maximum(R, 0)
            Rp[idx, idx] = R[idx, idx]
            col = Rp.sum(axis=0)
            A_new = col[None, :] - Rp
            diag = A_new[idx, idx].copy()
            A_new = np.minimum(A_new, 0)
            A_new[idx, idx] = diag
            A = self.damping * A + (1 - self.damping) * A_new

            exemplars = np.flatnonzero(np.diag(A + R) > 0)
            if prev is not None and len(exemplars) == len(prev) and np.array_equal(exemplars, prev):
                stable += 1
                if stable >= self.convergence_iter:
                    break
            else:
                stable = 0
            prev = exemplars

        if len(exemplars) == 0:
            exemplars = np.array([int(np.argmax(np.diag(A + R)))])
        labels = np.argmax(S[:, exemplars], axis=1)
        labels[exemplars] = np.arange(len(exemplars))
        self.cluster_centers_indices_ = exemplars
        self.cluster_centers_ = X[exemplars]
        self.labels_ = labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("call fit first")
        return np.argmin(pairwise(np.asarray(X, dtype=float), self.cluster_centers_), axis=1)
