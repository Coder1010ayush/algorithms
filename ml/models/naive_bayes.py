from typing import Literal

import numpy as np

from ml.base import BaseModel


class NaiveBayes(BaseModel):
    def __init__(self, model_type: Literal["gaussian", "multinomial", "bernoulli"] = "gaussian", smoothing: float = 1.0):
        if model_type not in ("gaussian", "multinomial", "bernoulli"):
            raise ValueError(f"unsupported model_type {model_type!r}")
        self.model_type = model_type
        self.smoothing = smoothing
        self.classes_: np.ndarray | None = None
        self.log_priors: np.ndarray | None = None
        self.params: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayes":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.log_priors = np.log(counts / len(y))
        if self.model_type == "gaussian":
            self.params["mean"] = np.array([X[y == c].mean(axis=0) for c in self.classes_])
            self.params["var"] = np.array([X[y == c].var(axis=0) for c in self.classes_]) + 1e-9
        elif self.model_type == "multinomial":
            counts = np.array([X[y == c].sum(axis=0) for c in self.classes_]) + self.smoothing
            self.params["log_prob"] = np.log(counts / counts.sum(axis=1, keepdims=True))
        else:
            prob = np.array(
                [(X[y == c].sum(axis=0) + self.smoothing) / (np.sum(y == c) + 2 * self.smoothing) for c in self.classes_]
            )
            self.params["log_prob"] = np.log(prob)
            self.params["log_neg"] = np.log(1.0 - prob)
        return self

    def _log_posterior(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.model_type == "gaussian":
            mean, var = self.params["mean"], self.params["var"]
            diff = X[:, None, :] - mean[None, :, :]
            ll = -0.5 * np.sum(diff**2 / var + np.log(2.0 * np.pi * var), axis=2)
        elif self.model_type == "multinomial":
            ll = X @ self.params["log_prob"].T
        else:
            ll = X @ self.params["log_prob"].T + (1.0 - X) @ self.params["log_neg"].T
        return ll + self.log_priors

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lp = self._log_posterior(X)
        lp -= lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._log_posterior(X), axis=1)]
