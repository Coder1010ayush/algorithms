from typing import Literal

import numpy as np

RegressionName = Literal["mse", "rmse", "mae", "r2_score", "mape"]
ClassificationName = Literal[
    "accuracy", "precision", "recall", "f1_score", "log_loss", "mcc", "cross_entropy", "hinge_loss"
]
_EPS = 1e-8


def mse(y_true, y_hat) -> float:
    return float(np.mean(np.square(np.asarray(y_true, float) - np.asarray(y_hat, float))))


def rmse(y_true, y_hat) -> float:
    return float(np.sqrt(mse(y_true, y_hat)))


def mae(y_true, y_hat) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_hat, float))))


def r2_score(y_true, y_hat) -> float:
    y_true, y_hat = np.asarray(y_true, float), np.asarray(y_hat, float)
    ss_res = np.sum(np.square(y_true - y_hat))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return float(1.0 - ss_res / (ss_tot + _EPS))


def mape(y_true, y_hat) -> float:
    y_true, y_hat = np.asarray(y_true, float), np.asarray(y_hat, float)
    return float(100.0 * np.mean(np.abs((y_true - y_hat) / np.clip(np.abs(y_true), _EPS, None))))


def accuracy(y_true, y_hat) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_hat)))


def _confusion(y_true, y_hat, positive=1):
    y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
    tp = np.sum((y_true == positive) & (y_hat == positive))
    fp = np.sum((y_true != positive) & (y_hat == positive))
    fn = np.sum((y_true == positive) & (y_hat != positive))
    tn = np.sum((y_true != positive) & (y_hat != positive))
    return tp, fp, fn, tn


def precision(y_true, y_hat) -> float:
    tp, fp, _, _ = _confusion(y_true, y_hat)
    return float(tp / (tp + fp + _EPS))


def recall(y_true, y_hat) -> float:
    tp, _, fn, _ = _confusion(y_true, y_hat)
    return float(tp / (tp + fn + _EPS))


def f1_score(y_true, y_hat) -> float:
    p, r = precision(y_true, y_hat), recall(y_true, y_hat)
    return float(2.0 * p * r / (p + r + _EPS))


def mcc(y_true, y_hat) -> float:
    tp, fp, fn, tn = (float(v) for v in _confusion(y_true, y_hat))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + _EPS
    return float((tp * tn - fp * fn) / denom)


def log_loss(y_true, y_prob) -> float:
    y_true, p = np.asarray(y_true, float), np.clip(np.asarray(y_prob, float), _EPS, 1 - _EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def cross_entropy(y_true, y_prob) -> float:
    """Multi-class cross entropy with one-hot ``y_true`` and probabilities ``y_prob``."""
    y_true, p = np.asarray(y_true, float), np.clip(np.asarray(y_prob, float), _EPS, 1.0)
    return float(-np.sum(y_true * np.log(p)) / max(len(np.atleast_2d(y_true)), 1))


def hinge_loss(y_true, y_score) -> float:
    return float(np.mean(np.maximum(0.0, 1.0 - np.asarray(y_true, float) * np.asarray(y_score, float))))


_REGRESSION = {"mse": mse, "rmse": rmse, "mae": mae, "r2_score": r2_score, "mape": mape}
_CLASSIFICATION = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score,
    "log_loss": log_loss,
    "mcc": mcc,
    "cross_entropy": cross_entropy,
    "hinge_loss": hinge_loss,
}


class RegressionMetric:
    def __init__(self, method: RegressionName = "mse"):
        if method not in _REGRESSION:
            raise ValueError(f"unsupported method {method!r}")
        self.method = method

    def __call__(self, y_true, y_hat) -> float:
        return _REGRESSION[self.method](y_true, y_hat)

    forward = __call__


class ClassificationMetric:
    def __init__(self, method: ClassificationName = "accuracy"):
        if method not in _CLASSIFICATION:
            raise ValueError(f"unsupported method {method!r}")
        self.method = method

    def __call__(self, y_true, y_hat) -> float:
        return _CLASSIFICATION[self.method](y_true, y_hat)

    forward = __call__
