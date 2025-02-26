# ------------------------------ utf-8 encoding ---------------------------
# this file cotains all kind of distance metric function used in majorly clustering problems

import math
import numpy as np
from typing import Literal, List, Union


class Distant:
    def __init__(self, method=Literal["eucledian", "manhattan", "cosine"]):
        self.method = method

    def forward(self, x: np.ndarray, y: np.ndarray):

        if self.method == "eucledian":
            val = np.sqrt(np.sum(np.square(x - y)))
            return val
        elif self.method == "manhattan":
            val = np.sum(np.abs(x - y))
            return val
        elif self.method == "cosine":
            dot_product = np.dot(x, y)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            return 1 - (dot_product / (norm_x * norm_y))
        else:
            raise ValueError(f"Unsupported distance method: {self.method}")
