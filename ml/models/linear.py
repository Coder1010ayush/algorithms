from typing import Literal

import numpy as np

from ml.base import BaseModel
from ml.optim.gradient_descent import GradientDescent, Penalty
from ml.utils.activation import sigmoid
from ml.utils.metrics import log_loss, mse


def _linear(X: np.ndarray, coeff: np.ndarray, intercept: float) -> np.ndarray:
    return X @ coeff + intercept


def _mse_grad(X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> tuple[np.ndarray, float]:
    err = y_hat - y
    return 2.0 * X.T @ err / len(y), float(2.0 * np.mean(err))


def _logistic(X: np.ndarray, coeff: np.ndarray, intercept: float) -> np.ndarray:
    return sigmoid(X @ coeff + intercept)


def _log_loss_grad(X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> tuple[np.ndarray, float]:
    err = y_hat - y
    return X.T @ err / len(y), float(np.mean(err))


class _GDModel(BaseModel):
    def __init__(
        self,
        optimizer: Literal["batch", "stochastic"] = "batch",
        epochs: int = 100,
        learning_rate: float = 1e-2,
        tolerance: float = 1e-8,
        penalty: Penalty = "none",
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        elastic_ratio: float = 0.5,
        random_state: int | None = None,
    ):
        self.optimizer = GradientDescent(
            mode=optimizer,
            epochs=epochs,
            learning_rate=learning_rate,
            tolerance=tolerance,
            penalty=penalty,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            elastic_ratio=elastic_ratio,
            random_state=random_state,
        )
        self.coeff: np.ndarray | None = None
        self.intercept: float = 0.0
        self.loss_history: list[float] = []

    def _fit(self, X, y, model_fn, loss_fn, grad_fn):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.coeff, self.intercept, self.loss_history = self.optimizer.optimize(X, y, model_fn, loss_fn, grad_fn)
        return self


class LinearRegression(_GDModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        return self._fit(X, y, _linear, mse, _mse_grad)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _linear(X.reshape(-1, 1) if X.ndim == 1 else X, self.coeff, self.intercept)


class LogisticRegression(_GDModel):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        return self._fit(X, y, _logistic, log_loss, _log_loss_grad)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _logistic(X.reshape(-1, 1) if X.ndim == 1 else X, self.coeff, self.intercept)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)
