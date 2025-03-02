import numpy as np
from models.model import LinearRegression
from optimizer.gradient_optimiser import GradientOptimizer
from models.kmeans_model.centroid import Centroid
from models.decision_tree.decision_tree import DecisionTreeID3
from models.knn_model.knn import KNearestNeighbour


def modify_array(y: np.ndarray, y_pred: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.where(y == y_pred, values * np.exp(0.1), values * np.exp(-0.1))


if __name__ == "__main__":
    # x = np.random.rand(50, 5)
    y1 = np.random.randint(low=0, high=1, size=(5,))
    y2 = np.random.randint(low=0, high=2, size=(5,))

    # dc = DecisionTreeID3(min_sample_split=5, max_depth=5)
    # dc.forward(x=x, y=y)

    # x_t = np.random.rand(5, 5)
    # op1 = dc.predict(x=x_t)
    # print(op1)
    # print()

    # knn = KNearestNeighbour(task="classification")
    # knn.forward(x=x, y=y)
    # x_t = np.random.rand(10, 5)
    # y_t = knn.predict(x_test=x_t)
    # print(y_t)

    y = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 3, 3, 2, 5])
    values = np.array([10, 20, 30, 40, 50])

    result = modify_array(y, y_pred, values)
    print(result)
