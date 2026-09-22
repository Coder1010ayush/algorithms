from typing import Literal

import numpy as np

from ml.base import BaseModel
from ml.models.clustering.kmeans import KMeans


def _log_gaussian(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mean).T)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (np.sum(z**2, axis=0) + log_det + d * np.log(2.0 * np.pi))


class GaussianMixtureModel(BaseModel):
    """EM-fitted mixture of full-covariance Gaussians."""

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-6,
        init: Literal["random", "kmeans"] = "kmeans",
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.reg_covar = reg_covar
        self.rng = np.random.default_rng(random_state)
        self.means: np.ndarray | None = None
        self.covariances: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.log_likelihood_: float = -np.inf
        self.n_iter_: int = 0

    def _initialize(self, X: np.ndarray) -> None:
        n, d = X.shape
        if self.init == "kmeans":
            seed = int(self.rng.integers(2**31))
            self.means = KMeans(self.n_components, random_state=seed).fit(X).centroids.copy()
        else:
            self.means = X[self.rng.choice(n, self.n_components, replace=False)].copy()
        self.covariances = np.array([np.cov(X.T) + self.reg_covar * np.eye(d) for _ in range(self.n_components)])
        self.weights = np.full(self.n_components, 1.0 / self.n_components)

    def _log_resp(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        log_p = np.column_stack([_log_gaussian(X, self.means[k], self.covariances[k]) for k in range(self.n_components)])
        log_p += np.log(self.weights)
        log_norm = np.logaddexp.reduce(log_p, axis=1, keepdims=True)
        return log_p - log_norm, float(log_norm.sum())

    def fit(self, X: np.ndarray, y=None) -> "GaussianMixtureModel":
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self._initialize(X)
        prev = -np.inf
        for it in range(1, self.max_iter + 1):
            log_resp, ll = self._log_resp(X)
            resp = np.exp(log_resp)
            nk = resp.sum(axis=0) + 1e-12
            self.weights = nk / n
            self.means = (resp.T @ X) / nk[:, None]
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances[k] = (diff.T * resp[:, k]) @ diff / nk[k] + self.reg_covar * np.eye(d)
            self.log_likelihood_, self.n_iter_ = ll, it
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self._log_resp(np.asarray(X, dtype=float))[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
