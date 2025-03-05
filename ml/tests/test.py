import numpy as np
import matplotlib.pyplot as plt
from models.linear import Linear
from models.logistic import LogisticModel
from models.knn_model.knn import KNearestNeighbour
from models.kmeans_model.k_means import (
    KMeansClusterClassification,
    KMedoids,
    AgglomerativeClustering,
)
from models.boosting_model.adaboost import AdaboostClassification, AdaboostRegression
from models.boosting_model.gradient_boost import (
    GradientBoostClassification,
    GradientBoostRegression,
)
from models.boosting_model.xgboost import XgBoostModelClassifier, XgBoostModelRegressor
from models.decision_tree.decision_tree import (
    DecisionTreeCART,
    DecisionTreeID3,
    DecisionTreeRegression,
)
from models.dimensional_reduction.pca import PCA
from models.mixture_model.gmm import GaussianMixtureModel, GaussianMixtureModelFast
from tests.data_generator import (
    generate_linear_data,
    generate_classification_data,
    generate_polynomial_data,
    plot_classification_data,
    plot_regression_data,
)
from utils.metrics import RegressionMetric, ClassificationMetric


def plot_data_with_labels(X, y, labels, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", marker="o", label="True Labels")
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="coolwarm",
        marker="x",
        label="Predicted Labels",
    )
    plt.title(title)
    plt.legend()
    plt.show()


def test_linear_regression():
    model = Linear(epochs=200, optimizer="batch", learning_rate=0.01)
    x, y = generate_linear_data(n_samples=50)
    model.forward(X=x, y=y)
    model.show_loss()
    x_test = x[10:50, :]
    y_out = model.predict(X=x_test)
    plot_regression_data(X=x_test, y=y_out, title="Linear Regression")


def test_logistic_classification():
    model = LogisticModel(optimizer="stochastic")
    x, y = generate_classification_data(n_features=2, n_classes=2)
    model.forward(x, y)
    model.show_loss()
    predictions = model.predict(x)
    plot_data_with_labels(x, y, predictions, "Logistic Classification")


def test_knn():
    model = KNearestNeighbour(num_neighbours=3)
    x, y = generate_classification_data(n_features=2, n_classes=2)
    model.forward(x, y)
    predictions = model.predict(x)
    plot_data_with_labels(x, y, predictions, "KNN Classification")


def test_kmeans():
    model = KMeansClusterClassification(degree=3)
    x, _ = generate_classification_data(n_features=2, n_classes=3)
    model.forward(x)
    labels = model.predict(x)
    plot_data_with_labels(x, _, labels, "K-Means Clustering")


def test_decision_tree():
    model = DecisionTreeCART()
    x, y = generate_classification_data(n_features=2, n_classes=2)
    model.forward(x, y)
    predictions = model.predict(x)
    plot_data_with_labels(x, y, predictions, "Decision Tree Classification")


def test_adaboost():
    model = AdaboostClassification()
    x, y = generate_classification_data(n_features=2, n_classes=2)
    model.forward(x, y)
    predictions = model.predict(x)
    plot_data_with_labels(x, y, predictions, "Adaboost Classification")


def test_gradient_boost():
    model = GradientBoostClassification()
    x, y = generate_classification_data(n_features=2, n_classes=2)
    model.forward(x, y)
    predictions = model.predict(x)
    plot_data_with_labels(x, y, predictions, "Gradient Boost Classification")


def test_xgboost():
    model = XgBoostModelClassifier(n_classes=2)
    x, y = generate_classification_data(n_features=2, n_classes=2)
    model.forward(x, y)
    predictions = model.predict(x)
    plot_data_with_labels(x, y, predictions, "XGBoost Classification")


def test_pca():
    model = PCA(num_features=2)
    x, _ = generate_classification_data(n_features=5, n_classes=3)
    reduced_x = model.fit_transform(x)
    plt.scatter(reduced_x[:, 0], reduced_x[:, 1])
    plt.title("PCA Dimensionality Reduction")
    plt.show()


def test_gmm():
    model = GaussianMixtureModel(num_cluster=3)
    x, y = generate_classification_data(n_features=2, n_classes=3)
    model.forward(x, y)
    labels = model.predict(x)
    plot_data_with_labels(x, y, labels, "Gaussian Mixture Model")


def run_all_tests():
    # test_linear_regression()
    # test_logistic_classification()
    test_knn()
    print(f"KNN testing is completed")
    test_kmeans()
    print(f"KMeans testing is completed")
    test_decision_tree()
    print(f"Decision Tree testing is completed")
    test_adaboost()
    print(f"Adaboost testing is completed")
    test_gradient_boost()
    print(f"Gradient Boosting testing is completed")
    test_xgboost()
    print(f"XGBoost testing is completed")
    test_pca()
    print(f"PCA testing is completed")
    test_gmm()
    print(f"Gaussian Mixture Model testing is completed")


if __name__ == "__main__":
    run_all_tests()
