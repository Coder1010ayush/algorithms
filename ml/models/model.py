# ------------------------------------- utf-8 encoding ----------------------------------
# this is the main file which contains a general architecture for defining optimiser and different-different models

from abc import ABC, abstractmethod
import numpy as np
from models.basemodel import BaseModel
from utils.activation import ActivationFunction
from utils.metrics import RegressionMetric, ClassificationMetric
from utils.distant_matric import Distant
from optimizer.gradient_optimiser import GradientOptimizer


class LinearRegression(BaseModel):
    def __init__(self):
        self.weights = None
        self.bias = None
        self.loss_fn = RegressionMetric(method="mse")

    def init_w_b(self, coeff_shape: tuple):
        self.weights = np.random.normal(loc=0, scale=1, size=coeff_shape)
        self.bias = 0

    def forward(self, X):
        if self.weights is None and self.bias is None:
            self.init_w_b(coeff_shape=(X.shape[1], 1))
        return np.dot(X, self.weights) + self.bias

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias

    def compute_loss(self, y, y_pred):
        return self.loss_fn.forward(y_true=y, y_hat=y_pred)

    def compute_gradient(self, X, y, y_pred):
        grad_w = -2 * np.dot(X.T, (y - y_pred)) / len(y)
        grad_b = -2 * np.mean(y - y_pred)
        return grad_w, grad_b, self.weights, self.bias

    def update_parameters(self, grad_w, grad_b, lr):
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b


class LogisticModel(BaseModel):
    def __init__(self):
        self.weights = None
        self.bias = None
        self.act = ActivationFunction(activation_fnc="sigmoid")
        self.loss_fn = ClassificationMetric(method="log_loss")

    def init_w_b(self, coeff_shape: tuple):
        self.weights = np.random.normal(loc=0, scale=1, size=coeff_shape)
        self.bias = 0.0

    def forward(self, X):
        if self.weights is None and self.bias is None:
            self.init_w_b(coeff_shape=(X.shape[1], 1))
        return self.act.forward(x=np.dot(X, self.weights) + self.bias)

    def predict(self, x):
        return self.act.forward(x=np.dot(x, self.weights) + self.bias)

    def compute_loss(self, y, y_pred):
        return self.loss_fn.forward(y_true=y, y_hat=y_pred)

    def compute_gradient(self, X, y, y_pred):
        grad_w = np.dot(X.T, (y_pred - y)) / len(y)
        grad_b = np.mean(y_pred - y)
        return grad_w, grad_b, self.weights, self.bias

    def update_parameters(self, grad_w, grad_b, lr):
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b


class SVMRegression(BaseModel):
    def __init__(
        self, epsilon=0.1, C=1.0, kernel="linear", gamma=0.1, degree=3, coef0=1
    ):
        self.weights = None
        self.bias = None
        self.epsilon = epsilon
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.loss_fn = RegressionMetric(method="mse")

    def init_w_b(self, coeff_shape: tuple):
        self.weights = np.random.normal(loc=0, scale=1, size=coeff_shape)
        self.bias = 0.0

    def kernel_function(self, X1, X2):
        if self.kernel == "linear":
            return np.dot(X1, X2.T)
        elif self.kernel == "rbf":
            return np.exp(-self.gamma * np.linalg.norm(X1 - X2, axis=1) ** 2)
        elif self.kernel == "poly":
            return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        else:
            raise ValueError("Unsupported kernel")

    def forward(self, X):
        if self.weights is None and self.bias is None:
            self.init_w_b(coeff_shape=(X.shape[1], 1))
        return np.dot(X, self.weights) + self.bias

    def compute_loss(self, y, y_pred):
        loss = np.maximum(0, np.abs(y - y_pred) - self.epsilon)
        return np.mean(loss) + (self.C / 2) * np.sum(self.weights**2)

    def compute_gradient(self, X, y, y_pred):
        diff = y_pred - y
        mask = np.abs(diff) > self.epsilon
        grad_w = np.dot(X.T, mask * np.sign(diff)) / len(y) + self.C * self.weights
        grad_b = np.mean(mask * np.sign(diff))
        return grad_w, grad_b, self.weights, self.bias

    def update_parameters(self, grad_w, grad_b, lr):
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b


class SVMClassifier(BaseModel):
    def __init__(self, C=1.0, kernel="linear", gamma=0.1, degree=3, coef0=1):
        self.weights = None
        self.bias = None
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.loss_fn = ClassificationMetric(method="hinge_loss")

    def init_w_b(self, coeff_shape: tuple):
        self.weights = np.random.normal(loc=0, scale=1, size=coeff_shape)
        self.bias = 0.0

    def kernel_function(self, X1, X2):
        if self.kernel == "linear":
            return np.dot(X1, X2.T)
        elif self.kernel == "rbf":
            return np.exp(-self.gamma * np.linalg.norm(X1 - X2, axis=1) ** 2)
        elif self.kernel == "poly":
            return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        else:
            raise ValueError("Unsupported kernel")

    def forward(self, X):
        if self.weights is None and self.bias is None:
            self.init_w_b(coeff_shape=(X.shape[1], 1))
        return np.dot(X, self.weights) + self.bias

    def compute_loss(self, y, y_pred):
        hinge_loss = np.maximum(0, 1 - y * y_pred)
        return np.mean(hinge_loss) + (self.C / 2) * np.sum(self.weights**2)

    def compute_gradient(self, X, y, y_pred):
        margin = y * y_pred
        mask = margin < 1
        grad_w = -np.dot(X.T, (mask * y)) / len(y) + self.C * self.weights
        grad_b = -np.mean(mask * y)
        return grad_w, grad_b, self.weights, self.bias

    def update_parameters(self, grad_w, grad_b, lr):
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b


# if __name__ == "__main__":
#     x = np.random.rand(50, 5)
#     y = np.random.rand(50, 1)
#     model = LinearRegression()
#     optimer = GradientOptimizer(model=model, epochs=2)
#     model = optimer.optimize_batch(X=x, y=y)
