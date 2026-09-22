import numpy as np

from ml.base import BaseModel
from ml.models.tree import DecisionTreeCART


class AdaBoostClassifier(BaseModel):
    """SAMME AdaBoost with decision stumps; supports any label set (binary or multi-class)."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 1, min_samples_split: int = 2, random_state: int | None = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = np.random.default_rng(random_state)
        self.trees: list[DecisionTreeCART] = []
        self.alphas: list[float] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        n = len(y)
        w = np.full(n, 1.0 / n)
        self.trees, self.alphas = [], []
        for _ in range(self.n_estimators):
            idx = self.rng.choice(n, n, replace=True, p=w)
            tree = DecisionTreeCART(self.min_samples_split, self.max_depth, task="classification").fit(X[idx], y[idx])
            pred = tree.predict(X)
            wrong = pred != y
            err = np.sum(w[wrong])
            if err >= 1.0 - 1.0 / k:
                continue
            alpha = np.log((1.0 - err) / max(err, 1e-10)) + np.log(k - 1.0)
            w = w * np.exp(alpha * wrong)
            w /= w.sum()
            self.trees.append(tree)
            self.alphas.append(float(alpha))
            if err < 1e-10:
                break
        return self

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(X), len(self.classes_)))
        for tree, alpha in zip(self.trees, self.alphas):
            pred = tree.predict(X)
            scores[np.arange(len(X)), np.searchsorted(self.classes_, pred)] += alpha
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_scores(np.asarray(X, dtype=float)), axis=1)]


class AdaBoostRegressor(BaseModel):
    """AdaBoost.R2 with regression trees and weighted-median prediction."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 3, min_samples_split: int = 2, random_state: int | None = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = np.random.default_rng(random_state)
        self.trees: list[DecisionTreeCART] = []
        self.betas: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostRegressor":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n = len(y)
        w = np.full(n, 1.0 / n)
        self.trees, self.betas = [], []
        for _ in range(self.n_estimators):
            idx = self.rng.choice(n, n, replace=True, p=w)
            tree = DecisionTreeCART(self.min_samples_split, self.max_depth, task="regression").fit(X[idx], y[idx])
            err_i = np.abs(tree.predict(X) - y)
            max_err = err_i.max()
            if max_err == 0:
                self.trees.append(tree)
                self.betas.append(1e-10)
                break
            err_i /= max_err
            err = np.sum(w * err_i)
            if err >= 0.5:
                break
            beta = err / (1.0 - err)
            w = w * beta ** (1.0 - err_i)
            w /= w.sum()
            self.trees.append(tree)
            self.betas.append(float(beta))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        preds = np.array([t.predict(X) for t in self.trees])
        weights = np.log(1.0 / np.array(self.betas))
        order = np.argsort(preds, axis=0)
        sorted_preds = np.take_along_axis(preds, order, axis=0)
        cum = np.cumsum(weights[order], axis=0)
        median_idx = np.argmax(cum >= 0.5 * weights.sum(), axis=0)
        return sorted_preds[median_idx, np.arange(X.shape[0])]
