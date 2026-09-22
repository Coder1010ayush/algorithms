from typing import Literal

import numpy as np

from ml.base import BaseModel


class LinearDiscriminantAnalysis(BaseModel):
    """Fisher discriminant projection plus a Gaussian class-conditional classifier with shared covariance."""

    def __init__(self, n_components: int | None = None, solver: Literal["eigen"] = "eigen", reg: float = 1e-6):
        if solver != "eigen":
            raise ValueError(f"unsupported solver {solver!r}")
        self.n_components = n_components
        self.solver = solver
        self.reg = reg
        self.classes_: np.ndarray | None = None
        self.priors_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.scalings_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearDiscriminantAnalysis":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        n, d = X.shape
        self.priors_ = counts / n
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        overall = X.mean(axis=0)

        Sw = np.zeros((d, d))
        Sb = np.zeros((d, d))
        for c, mu, nc in zip(self.classes_, self.means_, counts):
            diff = X[y == c] - mu
            Sw += diff.T @ diff
            gap = (mu - overall)[:, None]
            Sb += nc * gap @ gap.T
        Sw += self.reg * np.eye(d)
        self.covariance_ = Sw / n

        vals, vecs = np.linalg.eigh(Sw)
        whiten = vecs / np.sqrt(vals)
        Sb_w = whiten.T @ Sb @ whiten
        evals, evecs = np.linalg.eigh(Sb_w)
        order = np.argsort(evals)[::-1]
        max_k = min(d, len(self.classes_) - 1)
        k = max_k if self.n_components is None else min(self.n_components, max_k)
        self.scalings_ = whiten @ evecs[:, order[:k]]
        pos = np.maximum(evals[order], 0)
        self.explained_variance_ratio_ = pos[:k] / max(pos.sum(), 1e-12)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scalings_ is None:
            raise RuntimeError("call fit first")
        return np.asarray(X, dtype=float) @ self.scalings_

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        inv = np.linalg.inv(self.covariance_)
        lin = X @ inv @ self.means_.T
        const = -0.5 * np.einsum("ij,jk,ik->i", self.means_, inv, self.means_) + np.log(self.priors_)
        return lin + const

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        s = self.decision_function(X)
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
