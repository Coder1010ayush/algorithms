from typing import Literal

import numpy as np

ActivationName = Literal["relu", "sigmoid", "tanh", "softmax"]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


_FUNCTIONS = {"relu": relu, "sigmoid": sigmoid, "tanh": tanh, "softmax": softmax}


class Activation:
    def __init__(self, name: ActivationName = "sigmoid"):
        if name not in _FUNCTIONS:
            raise ValueError(f"unsupported activation {name!r}")
        self.name = name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return _FUNCTIONS[self.name](np.asarray(x, dtype=float))

    forward = __call__
