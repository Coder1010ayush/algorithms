from typing import Callable, Literal

import numpy as np

Penalty = Literal["none", "l1", "l2", "elastic"]
ModelFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]
LossFn = Callable[[np.ndarray, np.ndarray], float]
GradFn = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, float]]


class GradientDescent:
    """Batch or stochastic gradient descent for models of the form f(X @ coeff + intercept).

    ``model_fn(X, coeff, intercept)`` returns predictions, ``loss_fn(y, y_hat)`` a scalar loss
    and ``grad_fn(X, y, y_hat)`` the gradients ``(d_coeff, d_intercept)``.
    """

    def __init__(
        self,
        mode: Literal["batch", "stochastic"] = "batch",
        epochs: int = 100,
        learning_rate: float = 1e-2,
        tolerance: float = 1e-8,
        penalty: Penalty = "none",
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        elastic_ratio: float = 0.5,
        max_grad_norm: float | None = None,
        random_state: int | None = None,
    ):
        if mode not in ("batch", "stochastic"):
            raise ValueError("mode must be 'batch' or 'stochastic'")
        if penalty not in ("none", "l1", "l2", "elastic"):
            raise ValueError(f"unsupported penalty {penalty!r}")
        self.mode = mode
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.penalty = penalty
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.elastic_ratio = elastic_ratio
        self.max_grad_norm = max_grad_norm
        self.rng = np.random.default_rng(random_state)
        self.coeff: np.ndarray | None = None
        self.intercept: float = 0.0
        self.loss_history: list[float] = []

    def _penalty_grad(self, coeff: np.ndarray) -> np.ndarray:
        if self.penalty == "l1":
            return self.l1_lambda * np.sign(coeff)
        if self.penalty == "l2":
            return 2.0 * self.l2_lambda * coeff
        if self.penalty == "elastic":
            l1 = self.elastic_ratio * self.l1_lambda * np.sign(coeff)
            l2 = (1.0 - self.elastic_ratio) * 2.0 * self.l2_lambda * coeff
            return l1 + l2
        return np.zeros_like(coeff)

    def _step(self, X, y, model_fn, loss_fn, grad_fn) -> tuple[float, bool]:
        y_hat = model_fn(X, self.coeff, self.intercept)
        d_coeff, d_intercept = grad_fn(X, y, y_hat)
        d_coeff = d_coeff + self._penalty_grad(self.coeff)
        if self.max_grad_norm is not None:
            d_coeff = np.clip(d_coeff, -self.max_grad_norm, self.max_grad_norm)
        self.coeff = self.coeff - self.learning_rate * d_coeff
        self.intercept = self.intercept - self.learning_rate * d_intercept
        converged = np.linalg.norm(d_coeff) < self.tolerance and abs(d_intercept) < self.tolerance
        return float(loss_fn(y, y_hat)), converged

    def optimize(
        self, X: np.ndarray, y: np.ndarray, model_fn: ModelFn, loss_fn: LossFn, grad_fn: GradFn
    ) -> tuple[np.ndarray, float, list[float]]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        self.coeff = self.rng.normal(0.0, 0.01, size=d)
        self.intercept = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            if self.mode == "batch":
                loss, converged = self._step(X, y, model_fn, loss_fn, grad_fn)
            else:
                total, converged = 0.0, False
                for i in self.rng.permutation(n):
                    loss_i, converged = self._step(X[i : i + 1], y[i : i + 1], model_fn, loss_fn, grad_fn)
                    total += loss_i
                    if converged:
                        break
                loss = total / n
            self.loss_history.append(loss)
            if converged:
                break
        return self.coeff, self.intercept, self.loss_history
