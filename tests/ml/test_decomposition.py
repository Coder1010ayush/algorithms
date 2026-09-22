import numpy as np
import pytest

from ml.models.decomposition import TSNE, FastICA, LinearDiscriminantAnalysis, TruncatedSVD
from ml.utils.distance import pairwise


def _knn_agreement(Z, y, k=5):
    D = pairwise(Z, Z)
    np.fill_diagonal(D, np.inf)
    neigh = np.argsort(D, axis=1)[:, :k]
    return np.mean(y[neigh] == y[:, None])


def test_tsne_keeps_clusters_separated(blobs3):
    X, y = blobs3
    model = TSNE(n_components=2, perplexity=10, n_iter=250, learning_rate=50, random_state=0)
    Z = model.fit_transform(X)
    assert Z.shape == (len(X), 2)
    assert _knn_agreement(Z, y) > 0.8
    assert model.kl_divergence_ >= 0
    with pytest.raises(ValueError):
        TSNE(perplexity=500).fit(X)


def test_fastica_unmixes_signals(rng):
    t = np.linspace(0, 8, 400)
    S = np.c_[np.sin(3 * t), np.sign(np.cos(7 * t))] + rng.normal(0, 0.02, (400, 2))
    A = np.array([[1.0, 0.5], [0.3, 2.0]])
    X = S @ A.T
    ica = FastICA(n_components=2, random_state=0).fit(X)
    recovered = ica.transform(X)
    corr = np.abs(np.corrcoef(recovered.T, S.T)[:2, 2:])
    assert corr.max(axis=1).min() > 0.9
    assert ica.components_.shape == (2, 2) and ica.mixing_.shape == (2, 2)
    np.testing.assert_allclose(ica.inverse_transform(recovered), X, atol=1e-8)
    assert FastICA(fun="cube", random_state=0).fit_transform(X).shape == (400, 2)


def test_lda_projects_and_classifies(blobs3):
    X, y = blobs3
    lda = LinearDiscriminantAnalysis().fit(X, y)
    assert lda.transform(X).shape == (len(X), 2)
    assert LinearDiscriminantAnalysis(n_components=1).fit(X, y).transform(X).shape == (len(X), 1)
    assert (lda.predict(X) == y).mean() > 0.95
    proba = lda.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(proba.argmax(axis=1), lda.predict(X))
    assert lda.explained_variance_ratio_.sum() == pytest.approx(1.0)


def test_truncated_svd_matches_numpy(rng):
    X = rng.normal(size=(30, 6)) @ rng.normal(size=(6, 6))
    svd = TruncatedSVD(n_components=3).fit(X)
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    np.testing.assert_allclose(svd.singular_values_, s[:3])
    np.testing.assert_allclose(np.abs(svd.components_), np.abs(vt[:3]))
    Z = svd.transform(X)
    assert Z.shape == (30, 3)
    assert 0 < svd.explained_variance_ratio_.sum() <= 1.0 + 1e-9
    assert np.linalg.norm(svd.inverse_transform(Z) - X) < np.linalg.norm(X)
    with pytest.raises(ValueError):
        TruncatedSVD(n_components=10).fit(X)
