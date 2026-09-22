from dataclasses import dataclass

import numpy as np

from ml.base import BaseModel
from ml.utils.activation import softmax


@dataclass
class XGNode:
    feature: int | None = None
    threshold: float | None = None
    left: "XGNode | None" = None
    right: "XGNode | None" = None
    value: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class XGTree:
    """Second-order tree: split gain and leaf weights come from gradients g and hessians h."""

    def __init__(self, max_depth: int = 6, min_samples_split: int = 5, reg_lambda: float = 1.0, gamma: float = 0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root: XGNode | None = None

    def _leaf(self, g: np.ndarray, h: np.ndarray) -> XGNode:
        return XGNode(value=float(-g.sum() / (h.sum() + self.reg_lambda)))

    def _best_split(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> tuple[int | None, float | None]:
        G, H = g.sum(), h.sum()
        base = G**2 / (H + self.reg_lambda)
        best_gain, best = 0.0, (None, None)
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f])
            xs, gs, hs = X[order, f], g[order], h[order]
            GL = np.cumsum(gs)[:-1]
            HL = np.cumsum(hs)[:-1]
            valid = xs[1:] != xs[:-1]
            gain = 0.5 * (GL**2 / (HL + self.reg_lambda) + (G - GL) ** 2 / (H - HL + self.reg_lambda) - base) - self.gamma
            gain = np.where(valid, gain, -np.inf)
            i = int(np.argmax(gain))
            if gain[i] > best_gain:
                best_gain, best = float(gain[i]), (f, float(xs[i + 1]))
        return best

    def _grow(self, X, g, h, depth) -> XGNode:
        if len(g) < self.min_samples_split or depth >= self.max_depth:
            return self._leaf(g, h)
        f, t = self._best_split(X, g, h)
        if f is None:
            return self._leaf(g, h)
        left = X[:, f] < t
        return XGNode(f, t, self._grow(X[left], g[left], h[left], depth + 1), self._grow(X[~left], g[~left], h[~left], depth + 1))

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> "XGTree":
        self.root = self._grow(X, g, h, 0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X))
        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf:
                node = node.left if x[node.feature] < node.threshold else node.right
            out[i] = node.value
        return out


class XGBoostRegressor(BaseModel):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 6, min_samples_split: int = 5, reg_lambda: float = 1.0, gamma: float = 0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tree_kwargs = dict(max_depth=max_depth, min_samples_split=min_samples_split, reg_lambda=reg_lambda, gamma=gamma)
        self.trees: list[XGTree] = []
        self.init_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.init_ = float(y.mean())
        pred = np.full(len(y), self.init_)
        self.trees = []
        for _ in range(self.n_estimators):
            tree = XGTree(**self.tree_kwargs).fit(X, pred - y, np.ones_like(y))
            pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        pred = np.full(len(X), self.init_)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred


class XGBoostClassifier(BaseModel):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 6, min_samples_split: int = 5, reg_lambda: float = 1.0, gamma: float = 0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tree_kwargs = dict(max_depth=max_depth, min_samples_split=min_samples_split, reg_lambda=reg_lambda, gamma=gamma)
        self.rounds: list[list[XGTree]] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        onehot = np.eye(k)[np.searchsorted(self.classes_, y)]
        logits = np.zeros((len(y), k))
        self.rounds = []
        for _ in range(self.n_estimators):
            p = softmax(logits, axis=1)
            g, h = p - onehot, p * (1.0 - p)
            trees = [XGTree(**self.tree_kwargs).fit(X, g[:, c], h[:, c]) for c in range(k)]
            for c, tree in enumerate(trees):
                logits[:, c] += self.learning_rate * tree.predict(X)
            self.rounds.append(trees)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        logits = np.zeros((len(X), len(self.classes_)))
        for trees in self.rounds:
            for c, tree in enumerate(trees):
                logits[:, c] += self.learning_rate * tree.predict(X)
        return softmax(logits, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
