# -------------------------------- utf-8 encoding --------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs


def generate_linear_data(
    n_samples=100, noise=0.1, slope=2, intercept=5, x_range=(-5, 5)
):
    X = np.linspace(x_range[0], x_range[1], n_samples)
    y = slope * X + intercept + np.random.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y


def generate_parabolic_data(n_samples=100, noise=0.1, a=1, b=0, c=0, x_range=(-5, 5)):
    X = np.linspace(x_range[0], x_range[1], n_samples)
    y = a * X**2 + b * X + c + np.random.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y


def generate_polynomial_data(
    n_samples=100, noise=0.1, coefficients=[1, 0, 0], x_range=(-5, 5)
):
    X = np.linspace(x_range[0], x_range[1], n_samples)
    y = sum(c * X**i for i, c in enumerate(coefficients)) + np.random.normal(
        0, noise, n_samples
    )
    return X.reshape(-1, 1), y


def generate_classification_data(
    n_samples=100, n_features=2, n_classes=2, noise=0.1, dataset_type="blobs"
):
    if dataset_type == "blobs":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_classes,
            n_features=n_features,
            random_state=42,
        )
    elif dataset_type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=42)
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_clusters_per_class=1,
            n_redundant=0,
            random_state=42,
        )
    X += np.random.normal(0, noise, X.shape)  # Add noise
    return X, y


def plot_regression_data(X, y, title="Regression Data"):
    plt.scatter(X, y, color="blue", alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


def plot_classification_data(X, y, title="Classification Data"):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.7)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.show()
