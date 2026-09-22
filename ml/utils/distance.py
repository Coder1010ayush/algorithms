from typing import Literal

import numpy as np

DistanceName = Literal["euclidean", "manhattan", "cosine"]


def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.square(x - y))))


def manhattan(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.abs(x - y)))


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(x, y) / denom)


def pairwise(X: np.ndarray, Y: np.ndarray, metric: DistanceName = "euclidean") -> np.ndarray:
    """Distance matrix of shape (len(X), len(Y)) in pure NumPy."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if metric == "euclidean":
        sq = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2.0 * X @ Y.T
        return np.sqrt(np.maximum(sq, 0.0))
    if metric == "manhattan":
        return np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
    if metric == "cosine":
        xn = np.linalg.norm(X, axis=1, keepdims=True)
        yn = np.linalg.norm(Y, axis=1, keepdims=True)
        return 1.0 - (X @ Y.T) / np.maximum(xn * yn.T, 1e-12)
    raise ValueError(f"unsupported metric {metric!r}")


_FUNCTIONS = {"euclidean": euclidean, "manhattan": manhattan, "cosine": cosine}


class Distance:
    def __init__(self, metric: DistanceName = "euclidean"):
        if metric not in _FUNCTIONS:
            raise ValueError(f"unsupported metric {metric!r}")
        self.metric = metric

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return _FUNCTIONS[self.metric](np.asarray(x, dtype=float), np.asarray(y, dtype=float))

    forward = __call__
