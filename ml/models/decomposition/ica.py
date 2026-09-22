from typing import Literal

import numpy as np

from ml.base import BaseModel


def _logcosh(u: np.ndarray):
    g = np.tanh(u)
    return g, 1.0 - g**2


def _exp(u: np.ndarray):
    e = np.exp(-(u**2) / 2)
    return u * e, (1 - u**2) * e


def _cube(u: np.ndarray):
    return u**3, 3 * u**2


_CONTRASTS = {"logcosh": _logcosh, "exp": _exp, "cube": _cube}


def _sym_decorrelate(W: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(W @ W.T)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T @ W


class FastICA(BaseModel):
    """Symmetric FastICA: whiten, then rotate to maximise non-Gaussianity."""

    def __init__(
        self,
        n_components: int | None = None,
        max_iter: int = 200,
        tol: float = 1e-4,
        fun: Literal["logcosh", "exp", "cube"] = "logcosh",
        whiten: bool = True,
        random_state: int | None = None,
    ):
        if fun not in _CONTRASTS:
            raise ValueError(f"unsupported fun {fun!r}")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.fun = fun
        self.whiten = whiten
        self.rng = np.random.default_rng(random_state)
        self.mean_: np.ndarray | None = None
        self.whitening_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.mixing_: np.ndarray | None = None
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y=None) -> "FastICA":
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = d if self.n_components is None else self.n_components
        self.mean_ = X.mean(axis=0)
        Xc = (X - self.mean_).T  # (d, n)
        if self.whiten:
            u, s, _ = np.linalg.svd(Xc, full_matrices=False)
            self.whitening_ = (u[:, :k] / s[:k]).T * np.sqrt(n)
        else:
            self.whitening_ = np.eye(k, d)
        Z = self.whitening_ @ Xc  # (k, n), unit covariance

        contrast = _CONTRASTS[self.fun]
        W = _sym_decorrelate(self.rng.normal(size=(k, k)))
        for it in range(1, self.max_iter + 1):
            g, dg = contrast(W @ Z)
            W_new = _sym_decorrelate(g @ Z.T / n - dg.mean(axis=1)[:, None] * W)
            change = np.max(np.abs(np.abs(np.einsum("ij,ij->i", W_new, W)) - 1))
            W = W_new
            if change < self.tol:
                break
        self.n_iter_ = it
        self.components_ = W @ self.whitening_
        self.mixing_ = np.linalg.pinv(self.components_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("call fit first")
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, S: np.ndarray) -> np.ndarray:
        return np.asarray(S, dtype=float) @ self.mixing_.T + self.mean_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)
