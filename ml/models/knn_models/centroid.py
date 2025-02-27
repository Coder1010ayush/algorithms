# ---------------------------------- utf-8 encoding ------------------------------
# this file contains major and all the conventional approach to define or declare centroids

import numpy as np
import math
from typing import Literal
from scipy.spatial.distance import cdist


class Centroid:

    def __init__(self, type_centroid: str = Literal["mean", "medoid", "probabilistic"]):
        assert type_centroid in {"mean", "medoid", "probabilistic"}, ValueError(
            f"unsupported type of centroid {type_centroid} is provided"
        )
        self.type_centroid = type_centroid

    def compute(self, points: np.ndarray, weights: np.ndarray = None):
        if self.type_centroid == "mean":
            return np.mean(points, axis=0)

        elif self.type_centroid == "medoid":
            distances = cdist(points, points, metric="euclidean")
            medoid_idx = np.argmin(distances.sum(axis=1))
            return points[medoid_idx]

        elif self.type_centroid == "probabilistic":
            assert weights is not None, "Weights required for probabilistic centroid"
            return np.average(points, axis=0, weights=weights)

        else:
            raise ValueError(f"Unknown centroid type {self.type_centroid}")
