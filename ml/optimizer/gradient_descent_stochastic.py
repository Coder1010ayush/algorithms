# -------------------- utf-8 encoding ----------------------------
# Gradient calculation function for stochastic gradient descent
import numpy as np


class GradientDescentStochastic:
    
    def __init__(self, coeff_shape: tuple, epochs=50, learning_rate=1e-3, tolerance=1e-8, debug_mode="off", debug_step=10 , norm = "" , l1_lambda = 0.2 , l2_lambda = 0.1 , el_param = 0.9):
        # norm could be l1 , l2 and elastic , ""
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.debug_mode = debug_mode
        self.debug_step = max(1, debug_step)
        self.coeff = np.random.normal(loc=0, scale=1.0, size=coeff_shape)
        self.intercept = 0.0
        self.norm = norm
        if self.norm == "l1":
            self.lambad_l1 = l1_lambda
        elif self.norm == "l2":
            self.lambad_l2 = l2_lambda
        elif self.norm == "elastic":
            self.lambad_l1 = l1_lambda
            self.lambad_l2 = l2_lambda
            self.el_param = el_param
        elif self.norm == "":
            pass
        else:
            raise ValueError(f"{self.norm} is unsupported")

        if self.debug_mode == "on":
            print(f"Number of Epochs: {self.epochs}")
            print(f"Learning Rate: {self.learning_rate}")

    def forward(self, x_train: np.ndarray, y_train: np.ndarray):
        n = x_train.shape[0]
        for ep in range(self.epochs):
            # Shuffle indices before each epoch
            indices = np.arange(n)
            np.random.shuffle(indices)

            for rand_idx in indices:
                y_hat = np.dot(x_train[rand_idx], self.coeff.T) + self.intercept
                
                # Compute gradients
                beta_not = -2 * (y_train[rand_idx] - y_hat) 
                beta = -2 * (y_train[rand_idx].item() - y_hat) * x_train[rand_idx]  

                # norm apply
                if self.norm == "":
                    pass

                elif self.norm == "l1":
                    l1_penalty = self.lambad_l1 * np.sign(self.coeff)
                    beta = beta + l1_penalty
                elif self.norm == "l2":
                    l2_penalty = self.lambad_l2 * 2 * self.coeff
                    beta = beta + l2_penalty
                elif self.norm == "elastic":
                    l1_penalty = self.lambad_l1 * np.sign(self.coeff)
                    l2_penalty = self.lambad_l2 * 2 * self.coeff
                    elastic_penalty = (l1_penalty + l2_penalty ) * self.el_param
                    beta = beta + elastic_penalty
                
                # Update parameters
                self.coeff -= self.learning_rate * beta
                self.intercept -= self.learning_rate * beta_not
                
                # Convergence check
                if np.linalg.norm(beta) < self.tolerance and abs(beta_not) < self.tolerance:
                    print(f"Converged at iteration {ep}")
                    return self.coeff, self.intercept

            # Debugging output
            if self.debug_mode == "on" and self.debug_step > 0 and ep % self.debug_step == 0:
                print(f"Epoch {ep}: Coefficients = {self.coeff}, Intercept = {self.intercept}")

        return self.coeff, self.intercept


if __name__ == "__main__":
    x_train = np.random.rand(10, 5)
    y_train = np.random.rand(10, 1)

    gd = GradientDescentStochastic(coeff_shape=(5,) , norm="elastic")
    coeffs, intercept = gd.forward(x_train, y_train)

    print("Final Coefficients:", coeffs)
    print("Final Intercept:", intercept)
