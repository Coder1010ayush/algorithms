import numpy as np
import pytest

from ml.models import DecisionTreeCART, DecisionTreeID3, DecisionTreeRegression
from ml.models.tree import entropy, gini


def test_impurities():
    assert gini(np.array([1, 1, 1])) == 0
    assert gini(np.array([0, 1])) == pytest.approx(0.5)
    assert entropy(np.array([0, 1])) == pytest.approx(1.0)


@pytest.mark.parametrize("cls", [DecisionTreeCART, DecisionTreeID3])
def test_classification_trees_fit_blobs(blobs2, cls):
    X, y = blobs2
    model = cls(min_samples_split=2, max_depth=5).fit(X, y)
    assert np.mean(model.predict(X) == y) == 1.0


def test_tree_accepts_ndarray_and_list_labels(blobs2):
    X, y = blobs2
    a = DecisionTreeCART().fit(X, y).predict(X)
    b = DecisionTreeCART().fit(X, list(y)).predict(X)
    assert np.array_equal(a, b)


def test_tree_max_depth_limits_growth(blobs2):
    X, y = blobs2
    model = DecisionTreeCART(max_depth=1, min_samples_split=2).fit(X, y)
    assert model.root.left.is_leaf and model.root.right.is_leaf


@pytest.mark.parametrize("cls", [DecisionTreeRegression, lambda **k: DecisionTreeCART(task="regression", **k)])
def test_regression_tree_fits_step_function(cls):
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.where(X[:, 0] < 5, 1.0, 5.0)
    model = cls(min_samples_split=2, max_depth=3).fit(X, y)
    assert np.mean((model.predict(X) - y) ** 2) < 1e-6


def test_max_features_subsampling(blobs2):
    X, y = blobs2
    X = np.hstack([X, np.zeros((len(X), 5))])
    model = DecisionTreeCART(max_features=2, random_state=0).fit(X, y)
    assert model.predict(X).shape == y.shape
