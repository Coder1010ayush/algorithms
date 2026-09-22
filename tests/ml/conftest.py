import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def blobs2(rng):
    """Two well-separated Gaussian blobs, labels in {0, 1}."""
    X = np.vstack([rng.normal(-3, 1, (40, 2)), rng.normal(3, 1, (40, 2))])
    y = np.r_[np.zeros(40, dtype=int), np.ones(40, dtype=int)]
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


@pytest.fixture
def blobs3(rng):
    centres = np.array([[-5, 0], [5, 0], [0, 6]])
    X = np.vstack([rng.normal(c, 0.8, (30, 2)) for c in centres])
    y = np.repeat(np.arange(3), 30)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


@pytest.fixture
def linear_data(rng):
    """y = 2 x0 - 3 x1 + 1 + small noise."""
    X = rng.uniform(-2, 2, (100, 2))
    y = 2 * X[:, 0] - 3 * X[:, 1] + 1 + rng.normal(0, 0.05, 100)
    return X, y
