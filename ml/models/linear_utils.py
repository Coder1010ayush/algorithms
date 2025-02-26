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
