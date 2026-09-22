from typing import Literal

import numpy as np

from ml.base import BaseModel
from ml.utils.distance import DistanceName, pairwise


class KNearestNeighbour(BaseModel):
    def __init__(
        self,
        n_neighbours: int = 3,
        metric: DistanceName = "euclidean",
        task: Literal["classification", "regression"] = "classification",
    ):
        if task not in ("classification", "regression"):
            raise ValueError(f"unsupported task {task!r}")
        self.n_neighbours = n_neighbours
        self.metric = metric
        self.task = task
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbour":
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _neighbours(self, X: np.ndarray) -> np.ndarray:
        d = pairwise(np.asarray(X, dtype=float), self.X_train, self.metric)
        return np.argsort(d, axis=1)[:, : self.n_neighbours]

    def predict(self, X: np.ndarray) -> np.ndarray:
        labels = self.y_train[self._neighbours(X)]
        if self.task == "regression":
            return labels.mean(axis=1)
        out = np.empty(len(labels), dtype=self.y_train.dtype)
        for i, row in enumerate(labels):
            values, counts = np.unique(row, return_counts=True)
            out[i] = values[np.argmax(counts)]
        return out
