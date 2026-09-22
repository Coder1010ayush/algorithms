import numpy as np
import pytest

from ml.models import PCA, GaussianProcessClassification, GaussianProcessRegression, MultiClassGaussianProcessClassification


def test_pca_recovers_dominant_direction(rng):
    t = rng.normal(0, 3, 200)
    X = np.c_[t, 2 * t, rng.normal(0, 0.01, 200)]
    pca = PCA(n_components=2).fit(X)
    direction = pca.components[0] / np.sign(pca.components[0][0])
    assert np.allclose(direction, np.array([1, 2, 0]) / np.sqrt(5), atol=1e-2)
    assert pca.explained_variance_ratio[0] > 0.99
    Z = pca.transform(X)
    assert Z.shape == (200, 2)
    assert np.allclose(pca.inverse_transform(Z), X, atol=0.1)


def test_pca_rejects_too_many_components():
    with pytest.raises(ValueError):
        PCA(n_components=5).fit(np.zeros((10, 3)))


def test_gp_regression_interpolates_and_gives_uncertainty():
    X = np.linspace(0, 5, 20).reshape(-1, 1)
    y = np.sin(X[:, 0])
    gp = GaussianProcessRegression(kernel="rbf", noise=1e-6).fit(X, y)
    mu, std = gp.predict(X, return_std=True)
    assert np.allclose(mu, y, atol=1e-3)
    assert np.all(std < 1e-2)
    far_mu, far_std = gp.predict(np.array([[50.0]]), return_std=True)
    assert far_std[0] > 0.5


def test_gp_hyperparameter_optimisation_improves_likelihood():
    X = np.linspace(0, 5, 20).reshape(-1, 1)
    y = np.sin(X[:, 0])
    gp = GaussianProcessRegression(kernel="rbf", noise=1e-3, kernel_params={"length_scale": 0.1, "sigma": 1.0}).fit(X, y)
    before = gp.log_marginal_likelihood()
    gp.optimize_hyperparameters()
    assert gp.log_marginal_likelihood() >= before


@pytest.mark.parametrize("kernel", ["matern", "periodic", "rational_quadratic", "linear"])
def test_gp_other_kernels_run(kernel):
    X = np.linspace(0, 5, 15).reshape(-1, 1)
    gp = GaussianProcessRegression(kernel=kernel, noise=1e-3).fit(X, X[:, 0])
    assert gp.predict(X).shape == (15,)


def test_gp_binary_classification(blobs2):
    X, y = blobs2
    model = GaussianProcessClassification(kernel="rbf", kernel_params={"length_scale": 2.0}).fit(X, y)
    assert np.mean(model.predict(X) == y) >= 0.95


def test_gp_multiclass_classification(blobs3):
    X, y = blobs3
    model = MultiClassGaussianProcessClassification(kernel="rbf", kernel_params={"length_scale": 2.0}).fit(X, y)
    assert np.mean(model.predict(X) == y) >= 0.95
