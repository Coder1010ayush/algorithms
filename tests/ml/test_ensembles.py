import numpy as np
import pytest

from ml.models import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostClassifier,
    GradientBoostRegressor,
    RandomForest,
    XGBoostClassifier,
    XGBoostRegressor,
)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_random_forest_classification(blobs2, n_jobs):
    X, y = blobs2
    model = RandomForest(n_trees=10, n_jobs=n_jobs, oob_score=True, random_state=0).fit(X, y)
    assert np.mean(model.predict(X) == y) >= 0.95
    assert model.oob_score_ is not None and model.oob_score_ >= 0.9


def test_random_forest_regression(linear_data):
    X, y = linear_data
    model = RandomForest(n_trees=20, task="regression", max_features=None, random_state=0).fit(X, y)
    assert np.corrcoef(model.predict(X), y)[0, 1] > 0.95


@pytest.mark.parametrize("labels", [(0, 1), (-1, 1)])
def test_adaboost_classifier(blobs2, labels):
    X, y = blobs2
    y = np.where(y == 1, labels[1], labels[0])
    model = AdaBoostClassifier(n_estimators=10, random_state=0).fit(X, y)
    pred = model.predict(X)
    assert set(np.unique(pred)) <= set(labels)
    assert np.mean(pred == y) >= 0.95


def test_adaboost_multiclass(blobs3):
    X, y = blobs3
    model = AdaBoostClassifier(n_estimators=20, random_state=0).fit(X, y)
    assert np.mean(model.predict(X) == y) >= 0.9


def test_adaboost_regressor(linear_data):
    X, y = linear_data
    model = AdaBoostRegressor(n_estimators=20, max_depth=4, random_state=0).fit(X, y)
    assert np.corrcoef(model.predict(X), y)[0, 1] > 0.95


def test_gradient_boost_regressor(linear_data):
    X, y = linear_data
    model = GradientBoostRegressor(n_estimators=50, learning_rate=0.2, max_depth=3).fit(X, y)
    assert np.mean((model.predict(X) - y) ** 2) < 0.3


def test_gradient_boost_classifier(blobs3):
    X, y = blobs3
    model = GradientBoostClassifier(n_estimators=10, learning_rate=0.5, max_depth=2).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.mean(model.predict(X) == y) >= 0.95


def test_xgboost_regressor(linear_data):
    X, y = linear_data
    model = XGBoostRegressor(n_estimators=50, learning_rate=0.2, max_depth=3).fit(X, y)
    assert np.mean((model.predict(X) - y) ** 2) < 0.3


def test_xgboost_classifier(blobs3):
    X, y = blobs3
    model = XGBoostClassifier(n_estimators=10, learning_rate=0.5, max_depth=2).fit(X, y)
    assert len(model.rounds) == 10 and len(model.rounds[0]) == 3
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)
    assert np.mean(model.predict(X) == y) >= 0.95
