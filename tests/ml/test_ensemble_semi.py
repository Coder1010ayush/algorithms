import numpy as np
import pytest

from ml.models import DecisionTreeCART, KNearestNeighbour, LogisticRegression, NaiveBayes
from ml.models.ensemble import StackingClassifier, VotingClassifier
from ml.models.linear import QuantileRegression
from ml.models.semi_supervised import CoTraining, SelfTrainingClassifier


def _estimators():
    return [
        ("lr", LogisticRegression(epochs=100, learning_rate=0.1, random_state=0)),
        ("knn", KNearestNeighbour(n_neighbours=5)),
        ("tree", DecisionTreeCART(max_depth=4)),
    ]


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_voting_classifier(blobs2, voting):
    X, y = blobs2
    model = VotingClassifier(_estimators(), voting=voting, weights=[1, 2, 1]).fit(X, y)
    assert np.mean(model.predict(X) == y) > 0.9
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2) and np.allclose(proba.sum(axis=1), 1.0)


def test_stacking_classifier(blobs2):
    X, y = blobs2
    model = StackingClassifier(_estimators(), final_estimator=LogisticRegression(epochs=200, learning_rate=0.5), cv=4, random_state=0).fit(X, y)
    assert np.mean(model.predict(X) == y) > 0.9
    assert model.predict_proba(X).shape == (len(y), 2)


def test_self_training_with_few_labels(blobs2, rng):
    X, y = blobs2
    y_partial = y.copy()
    y_partial[rng.random(len(y)) > 0.1] = -1
    assert 0 < (y_partial != -1).sum() < len(y)
    model = SelfTrainingClassifier(NaiveBayes(), threshold=0.7).fit(X, y_partial)
    assert np.mean(model.predict(X) == y) > 0.85
    assert np.all(model.transduction_ != -1)


def test_co_training(blobs2, rng):
    X, y = blobs2
    X = np.c_[X, X + rng.normal(0, 0.5, X.shape)]
    y_partial = y.copy()
    y_partial[rng.random(len(y)) > 0.15] = -1
    model = CoTraining(NaiveBayes(), NaiveBayes(), feature_split=([0, 1], [2, 3]), k=4, n_iter=10).fit(X, y_partial)
    assert np.mean(model.predict(X) == y) > 0.8


def test_quantile_regression_upper_quantile(rng):
    X = rng.uniform(-2, 2, (400, 1))
    y = 2 * X[:, 0] + 1 + rng.normal(0, 1.0, 400)
    high = QuantileRegression(quantile=0.9, epochs=1500, learning_rate=0.05, random_state=0).fit(X, y)
    frac_below = np.mean(y <= high.predict(X))
    assert 0.85 <= frac_below <= 0.95
    median = QuantileRegression(quantile=0.5, epochs=1500, learning_rate=0.05, random_state=0).fit(X, y)
    assert 0.45 <= np.mean(y <= median.predict(X)) <= 0.55
    with pytest.raises(ValueError):
        QuantileRegression(quantile=1.5)
