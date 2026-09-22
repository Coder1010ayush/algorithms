import copy
from typing import Literal, Sequence

import numpy as np

from ml.base import BaseModel


def proba_matrix(model, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Class-probability matrix aligned with ``classes`` for any model with predict_proba or predict."""
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X), dtype=float)
        if p.ndim == 1:
            p = np.c_[1.0 - p, p]
        model_classes = getattr(model, "classes_", None)
        if model_classes is None or len(model_classes) == p.shape[1] and np.array_equal(model_classes, classes):
            return p
        out = np.zeros((len(X), len(classes)))
        for j, c in enumerate(model_classes):
            out[:, np.searchsorted(classes, c)] = p[:, j]
        return out
    pred = np.asarray(model.predict(X))
    return (pred[:, None] == classes[None, :]).astype(float)


class VotingClassifier(BaseModel):
    def __init__(
        self,
        estimators: Sequence[tuple[str, BaseModel]],
        voting: Literal["hard", "soft"] = "hard",
        weights: Sequence[float] | None = None,
    ):
        if voting not in ("hard", "soft"):
            raise ValueError("voting must be 'hard' or 'soft'")
        self.estimators = list(estimators)
        self.voting = voting
        self.weights = np.ones(len(self.estimators)) if weights is None else np.asarray(weights, dtype=float)
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingClassifier":
        self.classes_ = np.unique(y)
        for _, model in self.estimators:
            model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = [w * proba_matrix(m, X, self.classes_) for w, (_, m) in zip(self.weights, self.estimators)]
        return np.sum(probs, axis=0) / self.weights.sum()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.voting == "soft":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        votes = np.zeros((len(X), len(self.classes_)))
        for w, (_, m) in zip(self.weights, self.estimators):
            votes += w * (np.asarray(m.predict(X))[:, None] == self.classes_[None, :])
        return self.classes_[np.argmax(votes, axis=1)]


class StackingClassifier(BaseModel):
    """Base estimators feed out-of-fold class probabilities to a final estimator."""

    def __init__(
        self,
        estimators: Sequence[tuple[str, BaseModel]],
        final_estimator: BaseModel,
        cv: int = 5,
        random_state: int | None = None,
    ):
        self.estimators = list(estimators)
        self.final_estimator = final_estimator
        self.cv = cv
        self.rng = np.random.default_rng(random_state)
        self.classes_: np.ndarray | None = None

    def _meta_features(self, X: np.ndarray, models: Sequence[BaseModel]) -> np.ndarray:
        return np.hstack([proba_matrix(m, X, self.classes_)[:, 1:] for m in models])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingClassifier":
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        folds = np.array_split(self.rng.permutation(len(y)), self.cv)
        meta = np.zeros((len(y), len(self.estimators) * (len(self.classes_) - 1)))
        for k in range(self.cv):
            val = folds[k]
            train = np.concatenate([f for j, f in enumerate(folds) if j != k])
            fold_models = [copy.deepcopy(m).fit(X[train], y[train]) for _, m in self.estimators]
            meta[val] = self._meta_features(X[val], fold_models)
        for _, m in self.estimators:
            m.fit(X, y)
        self.final_estimator.fit(meta, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.final_estimator.predict(self._meta_features(np.asarray(X), [m for _, m in self.estimators]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta = self._meta_features(np.asarray(X), [m for _, m in self.estimators])
        return proba_matrix(self.final_estimator, meta, self.classes_)
