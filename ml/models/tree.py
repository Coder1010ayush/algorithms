from dataclasses import dataclass
from typing import Literal

import numpy as np

from ml.base import BaseModel


@dataclass
class Node:
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    value: float | int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def gini(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return float(1.0 - np.sum(p**2))


def entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return float(-np.sum(p * np.log2(p)))


def variance(y: np.ndarray) -> float:
    return float(np.var(y)) if len(y) else 0.0


def majority(y: np.ndarray):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


class _Tree(BaseModel):
    """Binary tree grown greedily by minimising a weighted impurity over all feature thresholds."""

    def __init__(self, min_samples_split: int = 5, max_depth: int = 10, max_features: int | None = None, random_state: int | None = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.root: Node | None = None

    def _impurity(self, y: np.ndarray) -> float:
        raise NotImplementedError

    def _leaf_value(self, y: np.ndarray):
        raise NotImplementedError

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None]:
        n, d = X.shape
        features = np.arange(d)
        if self.max_features is not None and self.max_features < d:
            features = self.rng.choice(d, self.max_features, replace=False)
        best_score, best = np.inf, (None, None)
        for f in features:
            col = X[:, f]
            for t in np.unique(col)[1:]:
                left = col < t
                n_left = left.sum()
                score = (n_left * self._impurity(y[left]) + (n - n_left) * self._impurity(y[~left])) / n
                if score < best_score:
                    best_score, best = score, (int(f), float(t))
        return best

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        if len(y) < self.min_samples_split or depth >= self.max_depth or len(np.unique(y)) == 1:
            return Node(value=self._leaf_value(y))
        f, t = self._best_split(X, y)
        if f is None:
            return Node(value=self._leaf_value(y))
        left = X[:, f] < t
        return Node(f, t, self._grow(X[left], y[left], depth + 1), self._grow(X[~left], y[~left], depth + 1))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.root = self._grow(X, y, 0)
        return self

    def predict_sample(self, x: np.ndarray):
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature] < node.threshold else node.right
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_sample(x) for x in np.asarray(X, dtype=float)])


class DecisionTreeID3(_Tree):
    """Classification tree using information gain (entropy)."""

    def _impurity(self, y):
        return entropy(y)

    def _leaf_value(self, y):
        return majority(y)


class DecisionTreeCART(_Tree):
    """Classification (Gini) or regression (variance) tree."""

    def __init__(self, min_samples_split: int = 5, max_depth: int = 10, task: Literal["classification", "regression"] = "classification", **kwargs):
        super().__init__(min_samples_split, max_depth, **kwargs)
        if task not in ("classification", "regression"):
            raise ValueError(f"unsupported task {task!r}")
        self.task = task

    def _impurity(self, y):
        return gini(y) if self.task == "classification" else variance(y)

    def _leaf_value(self, y):
        return majority(y) if self.task == "classification" else float(np.mean(y))


class DecisionTreeRegression(_Tree):
    def _impurity(self, y):
        return variance(y)

    def _leaf_value(self, y):
        return float(np.mean(y))
