import numpy as np
import pytest

from ml.models.clustering import DBSCAN, AffinityPropagation, KMeans, MeanShift, SpectralClustering


def _purity(labels, y):
    total = 0
    for c in np.unique(labels):
        if c == -1:
            continue
        total += np.bincount(y[labels == c]).max()
    return total / len(y)


@pytest.fixture
def rings(rng):
    """Two concentric rings, inner label 0, outer label 1."""
    t = rng.uniform(0, 2 * np.pi, 120)
    inner = np.c_[np.cos(t[:60]), np.sin(t[:60])] * 1.0
    outer = np.c_[np.cos(t[60:]), np.sin(t[60:])] * 4.0
    X = np.vstack([inner, outer]) + rng.normal(0, 0.05, (120, 2))
    y = np.r_[np.zeros(60, dtype=int), np.ones(60, dtype=int)]
    return X, y


def test_dbscan_finds_blobs_and_flags_noise(blobs3):
    X, y = blobs3
    X = np.vstack([X, [[30.0, 30.0]]])
    model = DBSCAN(eps=1.2, min_samples=4).fit(X)
    labels = model.labels_
    assert labels[-1] == -1
    assert len(set(labels[:-1]) - {-1}) == 3
    assert _purity(labels[:-1], y) > 0.95
    assert np.array_equal(model.predict(X[:5]), labels[:5])
    assert model.predict(np.array([[100.0, 100.0]]))[0] == -1


def test_dbscan_separates_rings(rings):
    X, y = rings
    labels = DBSCAN(eps=1.5, min_samples=3).fit_predict(X)
    assert len(set(labels) - {-1}) == 2
    assert _purity(labels, y) > 0.95


def test_mean_shift_finds_three_centres(blobs3):
    X, y = blobs3
    model = MeanShift(bandwidth=2.5).fit(X)
    assert len(model.cluster_centers_) == 3
    assert _purity(model.labels_, y) > 0.95
    assert np.array_equal(model.predict(X), model.labels_)
    auto = MeanShift().fit(X)
    assert 2 <= len(auto.cluster_centers_) <= 4
    binned = MeanShift(bandwidth=2.5, bin_seeding=True).fit(X)
    assert len(binned.cluster_centers_) == 3


def test_spectral_separates_rings_where_kmeans_cannot(rings):
    X, y = rings
    spectral = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=8, random_state=0).fit(X)
    assert _purity(spectral.labels_, y) > 0.95
    assert _purity(KMeans(n_clusters=2, random_state=0).fit(X).labels_, y) < 0.8
    rbf = SpectralClustering(n_clusters=2, affinity="rbf", gamma=2.0, random_state=0).fit(X)
    assert _purity(rbf.labels_, y) > 0.95
    assert np.array_equal(spectral.predict(X[:10]), spectral.labels_[:10])


def test_affinity_propagation_on_blobs(blobs3):
    X, y = blobs3
    model = AffinityPropagation(damping=0.7, preference=-50.0).fit(X)
    assert len(model.cluster_centers_indices_) == 3
    assert _purity(model.labels_, y) > 0.95
    assert np.array_equal(model.predict(X), model.labels_)
    assert np.all(model.labels_[model.cluster_centers_indices_] == np.arange(3))
