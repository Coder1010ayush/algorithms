from typing import Literal

import numpy as np

Method = Literal[
    "min_max",
    "z_score",
    "max_abs",
    "unit_vector",
    "log",
    "mean",
    "box_cox",
    "yeo_johnson",
    "quantile",
]


def _box_cox(x: np.ndarray, lam: float) -> np.ndarray:
    return np.log(x) if lam == 0 else (x**lam - 1.0) / lam


def _yeo_johnson(x: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    if lam != 0:
        out[pos] = ((x[pos] + 1.0) ** lam - 1.0) / lam
    else:
        out[pos] = np.log1p(x[pos])
    if lam != 2:
        out[~pos] = -((-x[~pos] + 1.0) ** (2.0 - lam) - 1.0) / (2.0 - lam)
    else:
        out[~pos] = -np.log1p(-x[~pos])
    return out


def _mle_lambda(x: np.ndarray, transform, grid: np.ndarray, log_jacobian) -> float:
    best_lam, best_ll = grid[0], -np.inf
    n = x.size
    for lam in grid:
        z = transform(x, lam)
        var = np.var(z)
        if var <= 0:
            continue
        ll = -0.5 * n * np.log(var) + log_jacobian(x, lam)
        if ll > best_ll:
            best_lam, best_ll = lam, ll
    return float(best_lam)


class Normalization:
    """Column-wise feature scaling. ``fit`` learns statistics, ``transform`` applies them."""

    def __init__(self, method: Method = "z_score", n_quantiles: int = 100):
        self.method = method
        self.n_quantiles = n_quantiles
        self.params: dict = {}

    def fit(self, X: np.ndarray) -> "Normalization":
        X = np.asarray(X, dtype=float)
        m = self.method
        if m == "min_max":
            self.params = {"min": X.min(axis=0), "max": X.max(axis=0)}
        elif m == "z_score":
            self.params = {"mean": X.mean(axis=0), "std": X.std(axis=0)}
        elif m == "max_abs":
            self.params = {"max_abs": np.abs(X).max(axis=0)}
        elif m == "mean":
            self.params = {"mean": X.mean(axis=0), "min": X.min(axis=0), "max": X.max(axis=0)}
        elif m == "box_cox":
            if np.any(X <= 0):
                raise ValueError("box_cox requires strictly positive values")
            grid = np.linspace(-3, 3, 121)
            lams = [
                _mle_lambda(col, _box_cox, grid, lambda c, l: (l - 1.0) * np.sum(np.log(c)))
                for col in X.T
            ]
            self.params = {"lambda": np.array(lams)}
        elif m == "yeo_johnson":
            grid = np.linspace(-3, 3, 121)
            lams = [
                _mle_lambda(
                    col, _yeo_johnson, grid, lambda c, l: (l - 1.0) * np.sum(np.sign(c) * np.log1p(np.abs(c)))
                )
                for col in X.T
            ]
            self.params = {"lambda": np.array(lams)}
        elif m == "quantile":
            q = np.linspace(0, 1, self.n_quantiles)
            self.params = {"quantiles": np.quantile(X, q, axis=0), "grid": q}
        elif m in ("unit_vector", "log"):
            self.params = {}
        else:
            raise ValueError(f"unsupported method {m!r}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        p, m = self.params, self.method
        if m == "min_max":
            return (X - p["min"]) / np.where(p["max"] - p["min"] == 0, 1.0, p["max"] - p["min"])
        if m == "z_score":
            return (X - p["mean"]) / np.where(p["std"] == 0, 1.0, p["std"])
        if m == "max_abs":
            return X / np.where(p["max_abs"] == 0, 1.0, p["max_abs"])
        if m == "unit_vector":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            return X / np.where(norms == 0, 1.0, norms)
        if m == "log":
            return np.log1p(X)
        if m == "mean":
            return (X - p["mean"]) / np.where(p["max"] - p["min"] == 0, 1.0, p["max"] - p["min"])
        if m == "box_cox":
            return np.column_stack([_box_cox(c, l) for c, l in zip(X.T, p["lambda"])])
        if m == "yeo_johnson":
            return np.column_stack([_yeo_johnson(c, l) for c, l in zip(X.T, p["lambda"])])
        if m == "quantile":
            cols = [np.interp(c, np.unique(qc), np.linspace(0, 1, len(np.unique(qc)))) for c, qc in zip(X.T, p["quantiles"].T)]
            return np.column_stack(cols)
        raise ValueError(f"unsupported method {m!r}")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
