# --------------------------------- utf-8 encoding --------------------------------
# this file contains implementation of pca (principal componant analysis ) algorithm

import numpy as np
import math


class PCA:

    def __init__(self, num_feature: int = 3):
        self.num_features = num_feature
        self.component = None
        self.mean = None

    def forward(self, x: np.ndarray):

        self.mean = np.mean(x, axis=0)
        x_std = x - self.mean
        cov = np.cov(x_std, rowvar=False)
        eigen_value, eigen_vectors = np.linalg.eig(cov)

        indices = np.argsort(eigen_value)[::-1]
        self.component = eigen_vectors[:, indices][: self.num_features]

    def transform(self, x: np.ndarray):
        x_std = x - self.mean
        out = np.dot(x_std, self.component)
        return out
