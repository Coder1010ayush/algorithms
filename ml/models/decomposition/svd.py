import numpy as np

from ml.base import BaseModel


class TruncatedSVD(BaseModel):
    """Rank-k factorisation X ≈ U S Vᵀ without centring, so it works on sparse-like count data."""

    def __init__(self, n_components: int = 2):
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None) -> "TruncatedSVD":
        X = np.asarray(X, dtype=float)
        k = self.n_components
        if k > min(X.shape):
            raise ValueError(f"n_components={k} exceeds min(X.shape)={min(X.shape)}")
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = vt[:k]
        self.singular_values_ = s[:k]
        Z = X @ self.components_.T
        self.explained_variance_ = Z.var(axis=0)
        self.explained_variance_ratio_ = self.explained_variance_ / max(X.var(axis=0).sum(), 1e-12)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("call fit first")
        return np.asarray(X, dtype=float) @ self.components_.T

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=float) @ self.components_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)
