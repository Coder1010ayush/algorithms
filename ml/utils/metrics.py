# ------------------------------- utf-8 encoding ---------------------------------
# this file contains all kinds of evaluation metrics

import numpy as np
import pandas as pd
from typing import Union, List, Literal


class RegressionMetric:
    def __init__(self, method: Literal["mse", "rmse", "mae", "r2_score", "mape"]):
        self.method = method

    def forward(self, y_true: np.ndarray, y_hat: np.ndarray):
        y_true, y_hat = np.array(y_true), np.array(y_hat)

        if self.method == "mae":
            return np.mean(np.abs(y_true - y_hat), axis=0)

        elif self.method == "mse":
            return np.mean(np.square(y_true - y_hat), axis=0)

        elif self.method == "rmse":
            return np.sqrt(np.mean(np.square(y_true - y_hat), axis=0))

        elif self.method == "r2_score":
            ss_total = np.sum(np.square(y_true - np.mean(y_true, axis=0)), axis=0)
            ss_residual = np.sum(np.square(y_true - y_hat), axis=0)
            return 1 - (ss_residual / (ss_total + 1e-8))  # Avoid division by zero

        elif self.method == "mape":
            epsilon = 1e-8
            return 100 * np.mean(
                np.abs((y_true - y_hat) / np.clip(np.abs(y_true), epsilon, None)),
                axis=0,
            )

        else:
            raise ValueError(f"Unsupported method '{self.method}' provided")


class ClassificationMetric:

    def __init__(
        self,
        method: Literal[
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "log_loss",
            "mcc",
            "cross_entropy",
            "hinge_loss",
        ],
    ):
        self.method = method

    def forward(self, y_true: np.ndarray, y_hat: np.ndarray):
        if self.method == "accuracy":
            return np.mean(y_true == y_hat)

        elif self.method == "precision":
            tp = np.sum((y_true == 1) & (y_hat == 1))
            fp = np.sum((y_true == 0) & (y_hat == 1))
            return tp / (tp + fp + 1e-8)

        elif self.method == "recall":
            tp = np.sum((y_true == 1) & (y_hat == 1))
            fn = np.sum((y_true == 1) & (y_hat == 0))
            return tp / (tp + fn + 1e-8)

        elif self.method == "f1_score":
            precision = ClassificationMetric("precision").forward(y_true, y_hat)
            recall = ClassificationMetric("recall").forward(y_true, y_hat)
            return 2 * (precision * recall) / (precision + recall + 1e-8)

        elif self.method == "log_loss":
            y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8)
            return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

        elif self.method == "mcc":
            tp = np.sum((y_true == 1) & (y_hat == 1))
            tn = np.sum((y_true == 0) & (y_hat == 0))
            fp = np.sum((y_true == 0) & (y_hat == 1))
            fn = np.sum((y_true == 1) & (y_hat == 0))
            numerator = tp * tn - fp * fn
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
            return numerator / denominator

        elif self.method == "hinge_loss":
            return np.mean(np.maximum(0, 1 - y_true * y_hat))

        elif self.method == "cross_entropy":
            return -np.mean(y_true * np.log10(y_hat))

        else:
            raise ValueError(f"Unsupported method '{self.method}' provided")
