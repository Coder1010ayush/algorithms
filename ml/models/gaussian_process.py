from typing import Callable, Literal

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fn
from scipy.special import kv

from ml.base import BaseModel
from ml.utils.activation import sigmoid, softmax
from ml.utils.distance import pairwise

KernelName = Literal["rbf", "matern", "periodic", "linear", "rational_quadratic"]


def rbf_kernel(X1, X2, sigma: float = 1.0, length_scale: float = 1.0) -> np.ndarray:
    d2 = pairwise(X1, X2) ** 2
    return sigma**2 * np.exp(-0.5 * d2 / length_scale**2)


def matern_kernel(X1, X2, sigma: float = 1.0, length_scale: float = 1.0, nu: float = 1.5) -> np.ndarray:
    d = pairwise(X1, X2) / length_scale
    if nu == 0.5:
        k = np.exp(-d)
    elif nu == 1.5:
        k = (1.0 + np.sqrt(3) * d) * np.exp(-np.sqrt(3) * d)
    elif nu == 2.5:
        k = (1.0 + np.sqrt(5) * d + 5.0 * d**2 / 3.0) * np.exp(-np.sqrt(5) * d)
    else:
        s = np.sqrt(2 * nu) * np.maximum(d, 1e-12)
        k = (2 ** (1 - nu) / gamma_fn(nu)) * s**nu * kv(nu, s)
        k[d == 0] = 1.0
    return sigma**2 * k


def periodic_kernel(X1, X2, sigma: float = 1.0, length_scale: float = 1.0, period: float = 1.0) -> np.ndarray:
    d = pairwise(X1, X2)
    return sigma**2 * np.exp(-2.0 * np.sin(np.pi * d / period) ** 2 / length_scale**2)


def linear_kernel(X1, X2, sigma_b: float = 1.0, sigma_v: float = 1.0) -> np.ndarray:
    return sigma_b**2 + sigma_v**2 * np.asarray(X1) @ np.asarray(X2).T


def rational_quadratic_kernel(X1, X2, sigma: float = 1.0, length_scale: float = 1.0, alpha: float = 1.0) -> np.ndarray:
    d2 = pairwise(X1, X2) ** 2
    return sigma**2 * (1.0 + d2 / (2.0 * alpha * length_scale**2)) ** (-alpha)


KERNELS: dict[str, Callable] = {
    "rbf": rbf_kernel,
    "matern": matern_kernel,
    "periodic": periodic_kernel,
    "linear": linear_kernel,
    "rational_quadratic": rational_quadratic_kernel,
}


class _GPBase(BaseModel):
    def __init__(self, kernel: KernelName = "rbf", noise: float = 1e-5, kernel_params: dict | None = None):
        if kernel not in KERNELS:
            raise ValueError(f"unsupported kernel {kernel!r}")
        self.kernel = kernel
        self.noise = noise
        self.kernel_params = dict(kernel_params or {})
        self.X_train: np.ndarray | None = None

    def _K(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return KERNELS[self.kernel](A, B, **self.kernel_params)


class GaussianProcessRegression(_GPBase):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegression":
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        K = self._K(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        self.L = cholesky(K, lower=True)
        self.alpha = cho_solve((self.L, True), self.y_train)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        X = np.asarray(X, dtype=float)
        Ks = self._K(self.X_train, X)
        mu = Ks.T @ self.alpha
        if not return_std:
            return mu
        v = solve_triangular(self.L, Ks, lower=True)
        var = np.diag(self._K(X, X)) + self.noise - np.sum(v**2, axis=0)
        return mu, np.sqrt(np.maximum(var, 0.0))

    def log_marginal_likelihood(self) -> float:
        n = len(self.X_train)
        return float(-0.5 * self.y_train @ self.alpha - np.sum(np.log(np.diag(self.L))) - 0.5 * n * np.log(2 * np.pi))

    def optimize_hyperparameters(self, names: tuple[str, ...] = ("length_scale", "sigma")) -> "GaussianProcessRegression":
        def objective(log_params):
            self.kernel_params.update({k: float(np.exp(v)) for k, v in zip(names, log_params)})
            self.fit(self.X_train, self.y_train)
            return -self.log_marginal_likelihood()

        x0 = np.log([self.kernel_params.get(k, 1.0) for k in names])
        res = minimize(objective, x0, method="L-BFGS-B")
        objective(res.x)
        return self


class GaussianProcess(GaussianProcessRegression):
    """Alias kept for backward compatibility."""


class GaussianProcessClassification(_GPBase):
    """Binary GP classifier with a Laplace approximation to the logistic posterior."""

    def __init__(self, kernel: KernelName = "rbf", noise: float = 1e-5, kernel_params: dict | None = None, max_iter: int = 20):
        super().__init__(kernel, noise, kernel_params)
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessClassification":
        self.X_train = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        t = np.where(np.asarray(y) == self.classes_[-1], 1.0, 0.0)
        n = len(t)
        K = self._K(self.X_train, self.X_train) + self.noise * np.eye(n)
        f = np.zeros(n)
        for _ in range(self.max_iter):
            pi = sigmoid(f)
            W = pi * (1.0 - pi)
            sqrt_W = np.sqrt(W)
            B = np.eye(n) + sqrt_W[:, None] * K * sqrt_W[None, :]
            L = cholesky(B, lower=True)
            b = W * f + (t - pi)
            a = b - sqrt_W * cho_solve((L, True), sqrt_W * (K @ b))
            f_new = K @ a
            if np.max(np.abs(f_new - f)) < 1e-8:
                f = f_new
                break
            f = f_new
        self.f_ = f
        self.grad_ = t - sigmoid(f)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Ks = self._K(self.X_train, np.asarray(X, dtype=float))
        return sigmoid(Ks.T @ self.grad_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.predict_proba(X) >= 0.5, self.classes_[-1], self.classes_[0])


class MultiClassGaussianProcessClassification(_GPBase):
    """One-vs-rest wrapper around the binary Laplace GP classifier."""

    def __init__(self, kernel: KernelName = "rbf", noise: float = 1e-5, kernel_params: dict | None = None, max_iter: int = 20):
        super().__init__(kernel, noise, kernel_params)
        self.max_iter = max_iter
        self.models: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiClassGaussianProcessClassification":
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models = {
            c: GaussianProcessClassification(self.kernel, self.noise, self.kernel_params, self.max_iter).fit(X, (y == c).astype(int))
            for c in self.classes_
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.column_stack([self.models[c].predict_proba(X) for c in self.classes_])
        return softmax(np.log(np.clip(scores, 1e-12, 1.0)), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
