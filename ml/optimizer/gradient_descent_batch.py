# -------------------- utf-8 encoding ----------------------------
# gradient calculation function is implemented for batched data
import math
import numpy as np


class GradientDescentBatch:

    def __init__(
        self,
        coeff_shape: tuple,
        epochs=50,
        learning_rate=1e-3,
        tolerence=1e-6,
        debug_mode="off",
        debug_step = 10
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tolerance = tolerence
        self.debug_mode = debug_mode
        self.intercept = 0.0
        self.debug_step = debug_step
        self.coeff = np.random.uniform(low=0.0, high=1.0, size=coeff_shape)
        if self.debug_mode == "on":
            print(f"Number of Epochs is : {self.epochs}")
            print(f"Learning is {self.learning_rate}")

    def forward(self, x_train: np.ndarray, y_train: np.ndarray):
        
        for ep in range(self.epochs):
            
            y_hat = np.dot(x_train , self.coeff.T) + self.intercept
            
            # gradient computation
            beta_not = -2 * np.mean(y_hat - y_train)
            beta = (-2 * (np.dot((y_train - y_hat) ,x_train) )  ) / x_train.shape[0]
            
            # parameter update
            self.coeff = self.coeff - self.learning_rate*beta
            self.intercept = self.intercept - self.learning_rate*beta_not
            
            if ep % self.debug_step == 0 and self.debug_mode == "on":
                print(f"Coefficient is {self.coeff}")
                print(f"Intercept is {self.intercept}")
            if np.all(np.abs(beta) < self.tolerance) and abs(beta_not) < self.tolerance:
                print(f"Converged at iteration {ep}")
                break
            
        return [self.coeff , self.intercept]


if __name__ == "__main__":
    x_train = np.random.rand(10, 5)
    y_train = np.random.rand(10, 1)
    gd = GradientDescentBatch(epochs=10 , coeff_shape=(5))
    a, b = gd.forward(x_train, y_train)
    print(a)
    print(b)
