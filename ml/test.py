import numpy as np
from models.model import GradientOptimizer, LinearRegression


if __name__ == "__main__":
    x = np.random.rand(50, 5)
    y = np.random.rand(50, 1)
    model = LinearRegression()
    optimer = GradientOptimizer(model=model, epochs=5)
    # model = optimer.optimize_batch(X=x, y=y)
    model = optimer.optimize_stochastic(X=x, y=y)
    optimer.show_loss_hist()
