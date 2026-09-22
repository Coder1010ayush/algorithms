import copy
from typing import Sequence

import numpy as np

from ml.base import BaseModel
from ml.models.ensemble import proba_matrix


class SelfTrainingClassifier(BaseModel):
    """Iteratively labels unlabeled rows (marked ``-1``) whose predicted probability exceeds ``threshold``."""

    def __init__(self, base_estimator: BaseModel, threshold: float = 0.75, max_iter: int = 10):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.max_iter = max_iter
        self.classes_: np.ndarray | None = None
        self.transduction_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SelfTrainingClassifier":
        X, y = np.asarray(X), np.asarray(y).copy()
        self.classes_ = np.unique(y[y != -1])
        for _ in range(self.max_iter):
            labeled = y != -1
            self.base_estimator.fit(X[labeled], y[labeled])
            if labeled.all():
                break
            probs = proba_matrix(self.base_estimator, X[~labeled], self.classes_)
            confident = probs.max(axis=1) >= self.threshold
            if not confident.any():
                break
            idx = np.flatnonzero(~labeled)[confident]
            y[idx] = self.classes_[np.argmax(probs[confident], axis=1)]
        self.transduction_ = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.base_estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return proba_matrix(self.base_estimator, np.asarray(X), self.classes_)


class CoTraining(BaseModel):
    """Two estimators on disjoint feature views each label the other's most confident unlabeled rows."""

    def __init__(
        self,
        estimator_a: BaseModel,
        estimator_b: BaseModel,
        feature_split: tuple[Sequence[int], Sequence[int]],
        k: int = 5,
        n_iter: int = 10,
    ):
        self.estimator_a = estimator_a
        self.estimator_b = estimator_b
        self.view_a, self.view_b = (np.asarray(v, dtype=int) for v in feature_split)
        self.k = k
        self.n_iter = n_iter
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CoTraining":
        X, y = np.asarray(X), np.asarray(y).copy()
        self.classes_ = np.unique(y[y != -1])
        for _ in range(self.n_iter):
            labeled = y != -1
            self.estimator_a.fit(X[labeled][:, self.view_a], y[labeled])
            self.estimator_b.fit(X[labeled][:, self.view_b], y[labeled])
            unlabeled = np.flatnonzero(~labeled)
            if len(unlabeled) == 0:
                break
            for model, view in ((self.estimator_a, self.view_a), (self.estimator_b, self.view_b)):
                unlabeled = np.flatnonzero(y == -1)
                if len(unlabeled) == 0:
                    break
                probs = proba_matrix(model, X[unlabeled][:, view], self.classes_)
                top = np.argsort(-probs.max(axis=1))[: self.k]
                y[unlabeled[top]] = self.classes_[np.argmax(probs[top], axis=1)]
        labeled = y != -1
        self.estimator_a.fit(X[labeled][:, self.view_a], y[labeled])
        self.estimator_b.fit(X[labeled][:, self.view_b], y[labeled])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        pa = proba_matrix(self.estimator_a, X[:, self.view_a], self.classes_)
        pb = proba_matrix(self.estimator_b, X[:, self.view_b], self.classes_)
        return (pa + pb) / 2.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
