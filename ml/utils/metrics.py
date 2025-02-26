# ------------------------------- utf-8 encoding ---------------------------------
# this file contains all kinds of evaluation metrics

import numpy as np
import pandas as pd
from typing import Union, List, Literal


class RegressionMetric:
    def __init__(self, method: Literal["mse", "rmse", "mae", "r2_score", "mape"]):
        self.method = method

    def forward(self, y_true: np.ndarray, y_hat: np.ndarray):
        if self.method == "mae":
            return np.mean(np.abs(y_true - y_hat))

        elif self.method == "mse":
            return np.mean(np.square(y_true - y_hat))

        elif self.method == "rmse":
            return np.sqrt(np.mean(np.square(y_true - y_hat)))

        elif self.method == "r2_score":
            ss_total = np.sum(np.square(y_true - np.mean(y_true)))
            ss_residual = np.sum(np.square(y_true - y_hat))
            return 1 - (ss_residual / ss_total)

        elif self.method == "mape":
            epsilon = 1e-8
            return 100 * np.mean(
                np.abs((y_true - y_hat) / np.maximum(np.abs(y_true), epsilon))
            )
        else:
            raise ValueError(f"Unsupported method '{self.method}' provided")


class ClassificationMetric:

    def __init__(
        self,
        method: Literal[
            "accuracy", "precision", "recall", "f1_score", "log_loss", "mcc"
        ],
    ):
        pass
