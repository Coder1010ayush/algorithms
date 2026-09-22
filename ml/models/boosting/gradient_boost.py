import numpy as np

from ml.base import BaseModel
from ml.models.tree import DecisionTreeRegression
from ml.utils.activation import softmax


class GradientBoostRegressor(BaseModel):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, min_samples_split: int = 5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees: list[DecisionTreeRegression] = []
        self.init_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostRegressor":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.init_ = float(y.mean())
        pred = np.full(len(y), self.init_)
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegression(self.min_samples_split, self.max_depth).fit(X, y - pred)
            pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        pred = np.full(len(X), self.init_)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred


class GradientBoostClassifier(BaseModel):
    """Multi-class gradient boosting: one regression tree per class per round on softmax residuals."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, min_samples_split: int = 5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rounds: list[list[DecisionTreeRegression]] = []
        self.classes_: np.ndarray | None = None
        self.init_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostClassifier":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)
        onehot = np.eye(k)[y_idx]
        self.init_ = np.log(np.bincount(y_idx, minlength=k) / len(y) + 1e-9)
        logits = np.tile(self.init_, (len(y), 1))
        self.rounds = []
        for _ in range(self.n_estimators):
            residual = onehot - softmax(logits, axis=1)
            trees = [DecisionTreeRegression(self.min_samples_split, self.max_depth).fit(X, residual[:, c]) for c in range(k)]
            for c, tree in enumerate(trees):
                logits[:, c] += self.learning_rate * tree.predict(X)
            self.rounds.append(trees)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        logits = np.tile(self.init_, (len(X), 1))
        for trees in self.rounds:
            for c, tree in enumerate(trees):
                logits[:, c] += self.learning_rate * tree.predict(X)
        return softmax(logits, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
