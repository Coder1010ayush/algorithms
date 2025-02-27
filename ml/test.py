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

    x = np.random.rand(10, 2)
    weights = np.random.rand(
        10,
    )
    centroid = Centroid(type_centroid="probabilistic")
    out = centroid.compute(points=x, weights=weights)
    print(out)
    print(out.shape)
