# -------------------- utf-8 encoding ----------------------------
# gradient calculation function is implemented
import math
import numpy as np


class GradientDescent:

    def __init__(self, epochs=50, learning_rate=1e-3, tolerance=1e-8, debug_mode="off"):
        self.epochs = epochs
        self.lr = learning_rate
        self.tolerance = tolerance
        self.slop = 0.0
        self.intercept = 0.0
        self.debug_mode = debug_mode
        if self.debug_mode == "on":
            print(f"Number of epochs : {self.epochs}")
            print(f"Learning Rate is {self.lr}")

    def forward(self, x_train: np.ndarray, y_train: np.ndarray):
        n = len(x_train)
        for ep in range(self.epochs):
            y_pred = self.slop * x_train + self.intercept

            # gradient computation
            new_slop = -2 * (np.mean((y_train - y_pred) * x_train))
            new_intercept = -2 * (np.mean(y_train - y_pred))

            # gradient update
            self.intercept = self.intercept - self.lr * new_intercept
            self.slop = self.slop - self.lr * new_slop

            if (
                abs(new_intercept - self.intercept) < self.tolerance
                and abs(new_slop - self.slope) < self.tolerance
            ):
                print(f"Converged at iteration {ep}")
                break
        return [self.slop, self.intercept]


if __name__ == "__main__":
    x_train = np.random.rand(10, 5)
    y_train = np.random.rand(10, 1)
    gd = GradientDescent(epochs=10)
    a, b = gd.forward(x_train, y_train)
    print(a)
    print(b)
