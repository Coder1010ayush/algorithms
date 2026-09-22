from typing import Literal

import numpy as np

from ml.base import BaseModel
from ml.optim.model_optimizer import GradientOptimizer

Kernel = Literal["linear", "rbf", "poly"]


def kernel_matrix(X1: np.ndarray, X2: np.ndarray, kernel: Kernel, gamma: float, degree: int, coef0: float) -> np.ndarray:
    if kernel == "linear":
        return X1 @ X2.T
    if kernel == "rbf":
        sq = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2.0 * X1 @ X2.T
        return np.exp(-gamma * np.maximum(sq, 0.0))
    if kernel == "poly":
        return (gamma * X1 @ X2.T + coef0) ** degree
    raise ValueError(f"unsupported kernel {kernel!r}")


class _SVMBase(BaseModel):
    """Primal SVM trained with (sub)gradient descent on the hinge / epsilon-insensitive loss.

    Non-linear kernels are handled by mapping ``X`` to kernel features against the training set.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: Kernel = "linear",
        gamma: float = 0.1,
        degree: int = 3,
        coef0: float = 1.0,
        learning_rate: float = 1e-2,
        epochs: int = 200,
        random_state: int | None = None,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.X_train: np.ndarray | None = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.kernel == "linear":
            return X
        return kernel_matrix(X, self.X_train, self.kernel, self.gamma, self.degree, self.coef0)

    def forward(self, X: np.ndarray) -> np.ndarray:
        F = self._features(X)
        if self.weights is None:
            self.weights = np.zeros(F.shape[1])
            self.bias = 0.0
        return F @ self.weights + self.bias

    def update_parameters(self, grad_w: np.ndarray, grad_b: float, lr: float) -> None:
        self.weights = self.weights - lr * grad_w
        self.bias = self.bias - lr * grad_b

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.X_train = X
        self.weights = None
        y = self._targets(np.asarray(y))
        GradientOptimizer(self, learning_rate=self.learning_rate, epochs=self.epochs).optimize_batch(X, y)
        return self

    def _targets(self, y: np.ndarray) -> np.ndarray:
        return y.astype(float)


class SVMClassifier(_SVMBase):
    """Binary classifier; accepts labels in {0, 1} or {-1, 1} and predicts in the same convention."""

    def _targets(self, y: np.ndarray) -> np.ndarray:
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVMClassifier is binary")
        return np.where(y == self.classes_[1], 1.0, -1.0)

    def compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        hinge = np.maximum(0.0, 1.0 - y * y_pred)
        return float(np.mean(hinge) + 0.5 * self.C * np.sum(self.weights**2))

    def compute_gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, float]:
        F = self._features(X)
        mask = (y * y_pred) < 1.0
        grad_w = -F.T @ (mask * y) / len(y) + self.C * self.weights
        grad_b = float(-np.mean(mask * y))
        return grad_w, grad_b

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])


class SVMRegression(_SVMBase):
    def __init__(self, epsilon: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        loss = np.maximum(0.0, np.abs(y - y_pred) - self.epsilon)
        return float(np.mean(loss) + 0.5 * self.C * np.sum(self.weights**2))

    def compute_gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, float]:
        F = self._features(X)
        diff = y_pred - y
        active = (np.abs(diff) > self.epsilon) * np.sign(diff)
        grad_w = F.T @ active / len(y) + self.C * self.weights
        return grad_w, float(np.mean(active))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
