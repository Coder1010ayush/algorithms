import numpy as np
import pytest

from ml.models import AgglomerativeClustering, GaussianMixtureModel, KMeans, KMedoids


def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    total = 0
    for c in np.unique(labels):
        _, counts = np.unique(truth[labels == c], return_counts=True)
        total += counts.max()
    return total / len(truth)


@pytest.mark.parametrize("init", ["random", "kmeans++", "uniform"])
def test_kmeans(blobs3, init):
    X, y = blobs3
    model = KMeans(n_clusters=3, init=init, random_state=0).fit(X)
    assert model.centroids.shape == (3, 2)
    assert purity(model.predict(X), y) >= 0.95


def test_kmedoids(blobs3):
    X, y = blobs3
    model = KMedoids(n_clusters=3, random_state=0).fit(X)
    assert all(any(np.array_equal(m, x) for x in X) for m in model.medoids)
    assert purity(model.predict(X), y) >= 0.95


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "centroid"])
def test_agglomerative(blobs3, linkage):
    X, y = blobs3
    model = AgglomerativeClustering(n_clusters=3, linkage=linkage).fit(X)
    assert purity(model.labels_, y) >= 0.95
    assert purity(model.predict(X), y) >= 0.95


def test_gmm(blobs3):
    X, y = blobs3
    model = GaussianMixtureModel(n_components=3, init="kmeans", random_state=0).fit(X)
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.isclose(model.weights.sum(), 1.0)
    assert purity(model.predict(X), y) >= 0.95


def test_gmm_random_init_increases_likelihood(blobs3):
    X, _ = blobs3
    model = GaussianMixtureModel(n_components=3, init="random", max_iter=1, random_state=0).fit(X)
    first = model.log_likelihood_
    model = GaussianMixtureModel(n_components=3, init="random", max_iter=50, random_state=0).fit(X)
    assert np.isfinite(model.log_likelihood_) and model.log_likelihood_ >= first
