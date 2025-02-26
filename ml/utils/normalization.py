# ------------------------ utf-8 encoding ------------------
import numpy as np
import math
import pandas as pd
from typing import Union, Literal
from scipy import stats


# all the major kinds of normalization technique is implemented here
class Normalization:

    def __init__(
        self,
        methode=Literal[
            "min_max",
            "z_score",
            "max_abs",
            "median_iqr",
            "unit_vector_scale",
            "log_transform",
            "power_transform_yeo_johnson",
            "power_transform_boxcox",
            "mean_normalization",
            "binary_transform",
            "quantile_transform",
        ],
    ):
        self.methode = methode

    def quantile_transform(self, x_val: np.ndarray):
        x_val = x_val.flatten()
        sorted_x = np.sort(x_val)

        # Generate quantiles
        quantiles = np.linspace(0, 1, self.n_quantiles)
        ranks = np.argsort(np.argsort(x_val)) / len(x_val)

        transformed_x = np.interp(
            ranks, quantiles, np.linspace(-3, 3, self.n_quantiles)
        )
        return transformed_x.reshape(-1, 1)

    def forward(self, x_val: Union[np.ndarray, pd.DataFrame]):
        if isinstance(x_val, pd.DataFrame):
            x_val = x_val.to_numpy()

        if self.methode == "min_max":
            max_v = np.max(x_val)
            min_v = np.min(x_val)
            x_val = (x_val - min_v) / (max_v - min_v)
        elif self.methode == "z_score":
            mean = np.mean(x_val)
            alpha = np.std(x_val)
            x_val = (x_val - mean) / alpha
        elif self.methode == "max_abs":
            max_v = np.max(np.abs(x_val))
            x_val = x_val / max_v
        elif self.methode == "unit_vector_scale":
            norm = np.linalg.norm(x_val, keepdims=True)
            x_val = x_val / norm
        elif self.methode == "log_transform":
            x_val = np.log2(x_val + 1)
        elif self.methode == "mean_normalization":
            mean = np.mean(x_val)
            max_v = np.max(x_val)
            min_v = np.min(x_val)
            x_val = (x_val - mean) / (max_v - min_v)
        elif self.method == "power_transform_boxcox":
            if np.any(x_val <= 0):
                raise ValueError(
                    "Box-Cox transformation requires all values to be positive."
                )
            x_val, _ = stats.boxcox(x_val.flatten())
            x_val = x_val.reshape(-1, 1)
        elif self.method == "power_transform_yeo_johnson":
            x_val, _ = stats.yeojohnson(x_val.flatten())
            x_val = x_val.reshape(-1, 1)
        elif self.methode == "quantile_transform":
            return self.quantile_transform(x_val=x_val)

        return x_val
