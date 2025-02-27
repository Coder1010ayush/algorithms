import numpy as np
from models.model import LinearRegression
from optimizer.gradient_optimiser import GradientOptimizer
from models.knn_models.centroid import Centroid

if __name__ == "__main__":
    # x = np.random.rand(50, 5)
    # y = np.random.rand(50, 1)
    # model = LinearRegression()
    # optimer = GradientOptimizer(model=model, epochs=5)
    # # model = optimer.optimize_batch(X=x, y=y)
    # model = optimer.optimize_stochastic(X=x, y=y)
    # optimer.show_loss_hist()

    # x = np.random.rand(10, 2)
    # weights = np.random.rand(
    #     10,
    # )
    # centroid = Centroid(type_centroid="probabilistic")
    # out = centroid.compute(points=x, weights=weights)
    # print(out)
    # print(out.shape)

    import numpy as np
    from scipy.spatial.distance import cdist

    XA = np.array([[1, 2], [3, 4], [5, 6]])  # 3 points
    XB = np.array([[0, 0], [7, 8]])  # 2 points

    dist_matrix = cdist(XA, XB, metric="euclidean")
    print(dist_matrix)
    # print(dist_matrix.shape)
    # print(XA.shape)
    # print(XB.shape)
    print()

    v = np.argmin(dist_matrix, axis=1)
    print(v)
