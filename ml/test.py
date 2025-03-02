import numpy as np
from models.model import LinearRegression
from optimizer.gradient_optimiser import GradientOptimizer
from models.kmeans_model.centroid import Centroid
from models.decision_tree.decision_tree import DecisionTreeID3
from models.knn_model.knn import KNearestNeighbour

if __name__ == "__main__":
    x = np.random.rand(50, 5)
    y = np.random.randint(low=0, high=3, size=(50,))

    # dc = DecisionTreeID3(min_sample_split=5, max_depth=5)
    # dc.forward(x=x, y=y)

    # x_t = np.random.rand(5, 5)
    # op1 = dc.predict(x=x_t)
    # print(op1)
    # print()

    knn = KNearestNeighbour(task="classification")
    knn.forward(x=x, y=y)
    x_t = np.random.rand(10, 5)
    y_t = knn.predict(x_test=x_t)
    print(y_t)
