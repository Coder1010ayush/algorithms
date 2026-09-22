import numpy as np
import pytest

from ml.models import KNearestNeighbour, NaiveBayes


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_knn_classification(blobs3, metric):
    X, y = blobs3
    model = KNearestNeighbour(n_neighbours=5, metric=metric).fit(X, y)
    assert np.mean(model.predict(X) == y) >= 0.95


def test_knn_regression():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    y = 3 * X[:, 0]
    model = KNearestNeighbour(n_neighbours=3, task="regression").fit(X, y)
    assert np.allclose(model.predict(X), y, atol=0.1)


def test_gaussian_nb(blobs3):
    X, y = blobs3
    model = NaiveBayes("gaussian").fit(X, y)
    assert np.mean(model.predict(X) == y) >= 0.95
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_bernoulli_and_multinomial_nb(rng):
    X = np.vstack([rng.binomial(1, 0.9, (40, 6)), rng.binomial(1, 0.1, (40, 6))])
    y = np.r_[np.zeros(40, int), np.ones(40, int)]
    assert np.mean(NaiveBayes("bernoulli").fit(X, y).predict(X) == y) >= 0.9
    counts = np.vstack([rng.poisson([5, 1, 1], (40, 3)), rng.poisson([1, 1, 5], (40, 3))])
    assert np.mean(NaiveBayes("multinomial").fit(counts, y).predict(counts) == y) >= 0.9
