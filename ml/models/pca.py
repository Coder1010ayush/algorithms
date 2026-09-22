import numpy as np

from ml.base import BaseModel


class PCA(BaseModel):
    def __init__(self, n_components: int | None = None):
        if n_components is not None and n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.explained_variance: np.ndarray | None = None
        self.explained_variance_ratio: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None) -> "PCA":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        n, d = X.shape
        k = d if self.n_components is None else self.n_components
        if k > d:
            raise ValueError(f"n_components={k} exceeds n_features={d}")
        self.mean = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - self.mean, full_matrices=False)
        var = s**2 / max(n - 1, 1)
        self.components = vt[:k]
        self.explained_variance = var[:k]
        self.explained_variance_ratio = var[:k] / var.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components is None:
            raise RuntimeError("call fit first")
        return (np.asarray(X, dtype=float) - self.mean) @ self.components.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=float) @ self.components + self.mean

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)
