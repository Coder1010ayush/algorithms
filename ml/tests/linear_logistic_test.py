# --------------------------------------- utf-8 encoding --------------------------------
import numpy as np
from models.linear import Linear
from models.logistic import LogisticModel
import matplotlib.pyplot as plt
from tests.data_generator import (
    generate_linear_data,
    generate_classification_data,
    generate_parabolic_data,
    generate_polynomial_data,
    plot_classification_data,
    plot_regression_data,
)


def test_linear_regression():
    model = Linear(epochs=500, optimizer="batch", learning_rate=0.01)
    x, y = generate_linear_data(n_samples=50)
    model.fit(X=x, y=y)
    model.show_loss()


if __name__ == "__main__":
    test_linear_regression()
