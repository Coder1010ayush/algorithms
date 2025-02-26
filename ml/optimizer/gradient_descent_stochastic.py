# -------------------- utf-8 encoding ----------------------------
# Gradient calculation function for stochastic gradient descent
import numpy as np


class GradientDescentStochastic:

    def __init__(
        self,
        epochs=50,
        learning_rate=1e-3,
        tolerance=1e-8,
        debug_mode="off",
        debug_step=10,
        norm="",
        l1_lambda=0.2,
        l2_lambda=0.1,
        el_param=0.9,
    ):
        # norm could be l1 , l2 and elastic , ""
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.debug_mode = debug_mode
        self.debug_step = max(1, debug_step)
        self.coeff = None
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
        self.loss_history = []

    def forward(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        model_function,
        loss_function,
        derivative_function,
    ):
        n = x_train.shape[0]
        self.coeff = np.random.uniform(low=0.0, high=1.0, size=(x_train.shape[1],))
        for ep in range(self.epochs):
            epoch_loss = 0
            indices = np.arange(n)
            np.random.shuffle(indices)

            for rand_idx in indices:
                # y_hat = np.dot(x_train[rand_idx], self.coeff.T) + self.intercept
                y_hat = model_function(x_train[rand_idx], self.coeff, self.intercept)

                # Compute gradients
                # beta_not = -2 * (y_train[rand_idx] - y_hat)
                # beta = -2 * (y_train[rand_idx].item() - y_hat) * x_train[rand_idx]
                beta_not, beta = derivative_function(
                    x_train[rand_idx],
                    y_train[rand_idx],
                    y_hat,
                    self.coeff,
                    self.intercept,
                )

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
                    elastic_penalty = (l1_penalty + l2_penalty) * self.el_param
                    beta = beta + elastic_penalty

                # loss computation
                epoch_loss += loss_function(y_train[rand_idx], y_hat)

                # Update parameters
                self.coeff -= self.learning_rate * beta
                self.intercept -= self.learning_rate * beta_not

                # Convergence check
                if (
                    np.linalg.norm(beta) < self.tolerance
                    and abs(beta_not) < self.tolerance
                ):
                    print(f"Converged at iteration {ep}")
                    return self.coeff, self.intercept

            avg_loss = epoch_loss / n
            self.loss_history.append(avg_loss)

            # Debugging output
            if (
                self.debug_mode == "on"
                and self.debug_step > 0
                and ep % self.debug_step == 0
            ):
                print(
                    f"Epoch {ep}: Loss = {avg_loss:.6f}, Coefficients Shape = {self.coeff.shape}, Intercept = {self.intercept}"
                )

        return [self.coeff, self.intercept, self.loss_history]


if __name__ == "__main__":
    x_train = np.random.rand(10, 5)
    y_train = np.random.rand(10, 1)

    gd = GradientDescentStochastic(norm="elastic")
    coeffs, intercept = gd.forward(x_train, y_train)

    print("Final Coefficients:", coeffs)
    print("Final Intercept:", intercept)
