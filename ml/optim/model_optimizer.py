from typing import Literal, Protocol

import numpy as np


class GradientModel(Protocol):
    weights: np.ndarray
    bias: float

    def forward(self, X: np.ndarray) -> np.ndarray: ...
    def compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float: ...
    def compute_gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, float]: ...
    def update_parameters(self, grad_w: np.ndarray, grad_b: float, lr: float) -> None: ...


class GradientOptimizer:
    """Drives any model that exposes forward / compute_loss / compute_gradient / update_parameters."""

    def __init__(
        self,
        model: GradientModel,
        learning_rate: float = 0.01,
        epochs: int = 100,
        tolerance: float = 1e-6,
        penalty: Literal["none", "l1", "l2", "elastic"] = "none",
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        elastic_ratio: float = 0.5,
        random_state: int | None = None,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.penalty = penalty
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.elastic_ratio = elastic_ratio
        self.rng = np.random.default_rng(random_state)
        self.loss_history: list[float] = []

    def _regularize(self, grad_w: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if self.penalty == "l1":
            return grad_w + self.l1_lambda * np.sign(weights)
        if self.penalty == "l2":
            return grad_w + 2.0 * self.l2_lambda * weights
        if self.penalty == "elastic":
            l1 = self.elastic_ratio * self.l1_lambda * np.sign(weights)
            l2 = (1.0 - self.elastic_ratio) * 2.0 * self.l2_lambda * weights
            return grad_w + l1 + l2
        return grad_w

    def _update(self, X: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
        y_pred = self.model.forward(X)
        loss = float(self.model.compute_loss(y, y_pred))
        grad_w, grad_b = self.model.compute_gradient(X, y, y_pred)
        grad_w = self._regularize(grad_w, self.model.weights)
        self.model.update_parameters(grad_w, grad_b, self.learning_rate)
        converged = np.linalg.norm(grad_w) < self.tolerance and abs(grad_b) < self.tolerance
        return loss, converged

    def optimize_batch(self, X: np.ndarray, y: np.ndarray) -> GradientModel:
        self.loss_history = []
        for _ in range(self.epochs):
            loss, converged = self._update(X, y)
            self.loss_history.append(loss)
            if converged:
                break
        return self.model

    def optimize_stochastic(self, X: np.ndarray, y: np.ndarray) -> GradientModel:
        self.loss_history = []
        n = X.shape[0]
        for _ in range(self.epochs):
            total, converged = 0.0, False
            for i in self.rng.permutation(n):
                loss, converged = self._update(X[i : i + 1], y[i : i + 1])
                total += loss
            self.loss_history.append(total / n)
            if converged:
                break
        return self.model
