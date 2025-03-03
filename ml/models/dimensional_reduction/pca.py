# --------------------------------- utf-8 encoding --------------------------------
# this file contains implementation of pca (principal componant analysis ) algorithm

import numpy as np
import math


class PCA:
    def __init__(self, num_features: int = None):
        if num_features is not None and num_features <= 0:
            raise ValueError("num_features must be a positive integer.")

        self.num_features = num_features
        self.components = None
        self.mean = None

    def forward(self, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input data must be a NumPy array.")

        if x.ndim != 2:
            raise ValueError("Input data must be a 2D array.")

        n_samples, n_features = x.shape
        if self.num_features is None:
            self.num_features = n_features

        if self.num_features > n_features:
            raise ValueError(
                f"num_features ({self.num_features}) must be <= number of features ({n_features})."
            )

        self.mean = np.mean(x, axis=0)
        x_std = x - self.mean

        cov_matrix = np.cov(x_std, rowvar=False)

        if np.linalg.matrix_rank(cov_matrix) < min(cov_matrix.shape):
            raise ValueError(
                "Covariance matrix is singular, PCA may not be applicable."
            )

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = np.real(eigenvectors[:, : self.num_features])

    def transform(self, x: np.ndarray):
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted. Call 'forward' first.")

        x_std = x - self.mean
        return np.dot(x_std, self.components)

    def fit_transform(self, x: np.ndarray):
        self.forward(x)
        return self.transform(x)
