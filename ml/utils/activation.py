# ------------------------------ utf-8 encoding ---------------------------
# this file cotains all kind of activation functions

import math
import numpy as np
from typing import Literal


class ActivationFunction:

    def __init__(self, activation_fnc=Literal["relu", "sigmoid", "tanh", "softmax"]):
        self.activation = activation_fnc

    def forward(self, x: np.ndarray):
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "sigmoid":
            out = 1 / (1 + np.exp(-x))
            return out
        elif self.activation == "tanh":
            x = x - np.max(x)
            out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
            return out
        elif self.activation == "softmax":
            x_shifted = x - np.max(x)
            out = np.exp(x_shifted) / np.sum(np.exp(x_shifted))
            return out
        else:
            raise ValueError(
                f"Unsupported activation function {self.activation} is provided!"
            )
