import numpy as np
import pytest

from ml.utils import Activation, ClassificationMetric, Distance, Normalization, RegressionMetric
from ml.utils.distance import pairwise
from ml.utils.metrics import cross_entropy


def test_regression_metrics_on_scalars_and_vectors():
    assert RegressionMetric("mse")(1.0, 3.0) == 4.0
    assert RegressionMetric("mae")([1, 2], [2, 4]) == 1.5
    assert RegressionMetric("r2_score")([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert RegressionMetric("rmse")([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_classification_metrics():
    y, p = np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1])
    assert ClassificationMetric("accuracy")(y, p) == 0.75
    assert ClassificationMetric("precision")(y, p) == pytest.approx(1.0)
    assert ClassificationMetric("recall")(y, p) == pytest.approx(2 / 3)
    assert ClassificationMetric("f1_score")(y, p) == pytest.approx(0.8, abs=1e-6)
    assert ClassificationMetric("log_loss")([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))


def test_cross_entropy_uses_natural_log():
    assert cross_entropy([[1, 0]], [[0.5, 0.5]]) == pytest.approx(np.log(2))


def test_activations():
    x = np.array([-1.0, 0.0, 2.0])
    assert np.array_equal(Activation("relu")(x), [0, 0, 2])
    assert Activation("sigmoid")(np.array([0.0]))[0] == 0.5
    assert np.isclose(Activation("softmax")(x).sum(), 1.0)
    with pytest.raises(ValueError):
        Activation("nope")


def test_distances_and_pairwise(rng):
    a, b = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    assert Distance("euclidean")(a, b) == 5.0
    assert Distance("manhattan")(a, b) == 7.0
    assert Distance("cosine")(b, 2 * b) == pytest.approx(0.0)
    X, Y = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    D = pairwise(X, Y)
    expected = np.array([[np.linalg.norm(x - y) for y in Y] for x in X])
    assert np.allclose(D, expected)


@pytest.mark.parametrize("method", ["min_max", "z_score", "max_abs", "mean", "unit_vector", "log", "quantile", "yeo_johnson"])
def test_normalization_methods_run(rng, method):
    X = np.abs(rng.normal(size=(20, 3))) + 0.1
    Z = Normalization(method).fit_transform(X)
    assert Z.shape == X.shape and np.all(np.isfinite(Z))


def test_z_score_and_min_max_ranges(rng):
    X = rng.normal(5, 2, (50, 2))
    Z = Normalization("z_score").fit_transform(X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9) and np.allclose(Z.std(axis=0), 1)
    M = Normalization("min_max").fit_transform(X)
    assert M.min() == 0 and M.max() == 1


def test_box_cox_requires_positive():
    with pytest.raises(ValueError):
        Normalization("box_cox").fit(np.array([[1.0], [-1.0]]))
    Z = Normalization("box_cox").fit_transform(np.exp(np.random.default_rng(1).normal(size=(30, 1))))
    assert Z.shape == (30, 1)
