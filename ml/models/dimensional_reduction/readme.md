# Principal Component Analysis (PCA) Implementation

## Overview
This file contains an implementation of **Principal Component Analysis (PCA)** from scratch using NumPy. PCA is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional form while retaining as much variance as possible.

## Features
- Implements PCA from scratch without using libraries like scikit-learn.
- Computes the principal components using eigenvalue decomposition.
- Supports dimensionality reduction by selecting a subset of the principal components.

## Installation
Ensure you have Python installed along with NumPy. You can install NumPy using:

```bash
pip install numpy
```

## Usage
### Import the PCA Class
```python
import numpy as np
from models.dimensional_reduction.pca import PCA
```

### Example Usage
```python
X = np.array([
    [2.5, 2.4, 1.5],
    [0.5, 0.7, 0.9],
    [2.2, 2.9, 1.3],
    [1.9, 2.2, 1.7],
    [3.1, 3.0, 2.5]
])

pca = PCA(num_feature=2)

pca.forward(X)

X_reduced = pca.transform(X)

print("Reduced Data:")
print(X_reduced)
```

## How It Works
1. **Standardization**: The data is mean-centered.
2. **Covariance Matrix Calculation**: The covariance matrix of the standardized data is computed.
3. **Eigen Decomposition**: Eigenvalues and eigenvectors of the covariance matrix are found.
4. **Sorting & Selection**: The top `num_feature` eigenvectors corresponding to the largest eigenvalues are selected.
5. **Projection**: The data is projected onto the new feature space.

## Class Implementation
### PCA Class
```python
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
```
