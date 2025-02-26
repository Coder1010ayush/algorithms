# -------------------------------- utf-8 encoding ----------------------
# all the model functions and its derivative function will be implemented here

import numpy as np
import math
from typing import Callable, Tuple, Union, Literal, List


def linear_function(x_val: np.ndarray, coeff: np.ndarray, intercept: float):
    return np.dot(x_val, coeff.T) + intercept


def linear_derivative_function(x_train, y_train, y_hat, coeff, intercept):
    beta_not = -2 * np.mean(y_hat - y_train)
    beta = (-2 * (np.dot((y_train - y_hat), x_train))) / x_train.shape[0]
    return beta_not, beta


def svm_function(
    x_val: np.ndarray,
    coeff: np.ndarray,
    intercept: float,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_train: np.ndarray,
):
    return kernel(x_val, x_train) @ coeff.T + intercept


def svm_derivative_function(
    x_train: np.ndarray,
    y_train: np.ndarray,
    coeff: np.ndarray,
    intercept: float,
    C: float,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Tuple[float, np.ndarray]:
    n_samples = x_train.shape[0]
    K = kernel(x_train, x_train)
    margins = y_train * (K @ coeff + intercept)
    mask = margins < 1

    beta_not = -C * np.sum(y_train[mask]) / n_samples
    beta = coeff - (C * (y_train[mask] @ K[mask]) / n_samples)
    return beta_not, beta


def linear_kernel(x1: np.ndarray, x2: np.ndarray):
    return np.dot(x1, x2.T)


def polynomial_kernel(x1: np.ndarray, x2: np.ndarray, degree: int = 3):
    return (np.dot(x1, x2.T) + 1) ** degree


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float = 0.1):
    pairwise_sq_dists = (
        np.sum(x1**2, axis=1)[:, None] + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
    )
    return np.exp(-gamma * pairwise_sq_dists)
