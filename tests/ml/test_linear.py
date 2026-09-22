import numpy as np
import pytest

from ml.models import LinearRegression, LogisticRegression, SVMClassifier, SVMRegression
from ml.optim import GradientDescent


@pytest.mark.parametrize("optimizer", ["batch", "stochastic"])
def test_linear_regression_recovers_coefficients(linear_data, optimizer):
    X, y = linear_data
    model = LinearRegression(optimizer=optimizer, epochs=300, learning_rate=0.05, random_state=0).fit(X, y)
    assert np.allclose(model.coeff, [2, -3], atol=0.1)
    assert abs(model.intercept - 1) < 0.1
    assert model.loss_history[-1] < model.loss_history[0]


def test_linear_regression_accepts_1d_input():
    x = np.linspace(-1, 1, 50)
    model = LinearRegression(epochs=500, learning_rate=0.1).fit(x, 3 * x + 2)
    assert np.allclose(model.predict(x), 3 * x + 2, atol=0.05)


@pytest.mark.parametrize("optimizer", ["batch", "stochastic"])
def test_logistic_regression_separates_blobs(blobs2, optimizer):
    X, y = blobs2
    model = LogisticRegression(optimizer=optimizer, epochs=200, learning_rate=0.1, random_state=0).fit(X, y)
    assert np.mean(model.predict(X) == y) >= 0.95
    proba = model.predict_proba(X)
    assert proba.shape == (len(y),) and np.all((proba >= 0) & (proba <= 1))


def test_logistic_default_optimizer_is_usable(blobs2):
    X, y = blobs2
    assert np.mean(LogisticRegression(epochs=100, learning_rate=0.1).fit(X, y).predict(X) == y) >= 0.9


def test_gradient_descent_always_returns_three_values():
    X, y = np.ones((5, 1)), np.zeros(5)
    gd = GradientDescent(mode="stochastic", epochs=3, learning_rate=0.5, tolerance=10.0)
    out = gd.optimize(X, y, lambda X, c, b: X @ c + b, lambda y, yh: float(np.mean((y - yh) ** 2)), lambda X, y, yh: (X.T @ (yh - y), float(np.mean(yh - y))))
    assert len(out) == 3


def test_penalty_shrinks_coefficients(linear_data):
    X, y = linear_data
    plain = LinearRegression(epochs=200, learning_rate=0.05, random_state=0).fit(X, y)
    ridge = LinearRegression(epochs=200, learning_rate=0.05, penalty="l2", l2_lambda=1.0, random_state=0).fit(X, y)
    assert np.linalg.norm(ridge.coeff) < np.linalg.norm(plain.coeff)


@pytest.mark.parametrize("labels", [(0, 1), (-1, 1)])
def test_svm_classifier_binary_labels(blobs2, labels):
    X, y = blobs2
    y = np.where(y == 1, labels[1], labels[0])
    model = SVMClassifier(C=0.01, learning_rate=0.05, epochs=200).fit(X, y)
    pred = model.predict(X)
    assert set(np.unique(pred)) <= set(labels)
    assert np.mean(pred == y) >= 0.95


def test_svm_rbf_kernel_on_nonlinear_data(rng):
    r = np.r_[rng.uniform(0, 1, 60), rng.uniform(2.5, 3.5, 60)]
    theta = rng.uniform(0, 2 * np.pi, 120)
    X = np.c_[r * np.cos(theta), r * np.sin(theta)]
    y = np.r_[np.zeros(60, int), np.ones(60, int)]
    model = SVMClassifier(C=1e-3, kernel="rbf", gamma=0.5, learning_rate=0.1, epochs=300).fit(X, y)
    assert np.mean(model.predict(X) == y) >= 0.9


def test_svm_regression(linear_data):
    X, y = linear_data
    model = SVMRegression(C=1e-4, epsilon=0.05, learning_rate=0.05, epochs=500).fit(X, y)
    assert np.mean(np.abs(model.predict(X) - y)) < 0.5
