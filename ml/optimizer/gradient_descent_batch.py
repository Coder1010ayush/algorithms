# -------------------- utf-8 encoding ----------------------------
# gradient calculation function is implemented for batched data
import math
import numpy as np


class GradientDescentBatch:

    def __init__(
        self,
        epochs=50,
        learning_rate=1e-3,
        tolerence=1e-6,
        debug_mode="off",
        debug_step=10,
        norm="",
        l1_lambda=0.2,
        l2_lambda=0.1,
        el_param=0.9,
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tolerance = tolerence
        self.debug_mode = debug_mode
        self.intercept = 0.0
        self.debug_step = max(1, debug_step)
        self.coeff = None
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
            print(f"Number of Epochs is : {self.epochs}")
            print(f"Learning is {self.learning_rate}")
        self.loss_history = []

    def forward(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        model_function,
        loss_function,
        derivative_function,
    ):
        max_grad_norm = 2.0
        self.coeff = np.random.randn(
            x_train.shape[1],
        )
        epoch_loss = 0
        for ep in range(self.epochs):

            # y_hat = np.dot(x_train , self.coeff.T) + self.intercept
            y_hat = model_function(x_train, self.coeff, self.intercept)

            # gradient computation
            # beta_not = -2 * np.mean(y_hat - y_train)
            # beta = (-2 * (np.dot((y_train - y_hat), x_train))) / x_train.shape[0]
            beta_not, beta = derivative_function(
                x_train, y_train, y_hat, self.coeff, self.intercept
            )
            beta = np.clip(beta, -max_grad_norm, max_grad_norm)
            if self.norm == "l1":
                l1_penalty = self.lambad_l1 * np.sign(self.coeff)
                beta = beta + l1_penalty
            elif self.norm == "l2":
                l2_penalty = self.lambad_l2 * 2 * self.coeff
                beta = beta + l2_penalty
            elif self.norm == "elastic":
                l1_penalty = self.lambad_l1 * np.sign(self.coeff)
                l2_penalty = self.lambad_l2 * 2 * self.coeff
                penalty = (l1_penalty + l2_penalty) * self.el_param
                beta = beta + penalty

            # parameter update
            self.coeff = self.coeff - self.learning_rate * beta
            self.intercept = self.intercept - self.learning_rate * beta_not

            # loss computation
            loss = loss_function(y_train, y_hat)
            epoch_loss += loss
            self.loss_history.append(loss)

            if ep % self.debug_step == 0 and self.debug_mode == "on":
                print(
                    f"Epoch {ep}: Loss = {loss:.6f}, Coefficients Shape = {self.coeff.shape}, Intercept = {self.intercept}"
                )

            if np.all(np.abs(beta) < self.tolerance) and abs(beta_not) < self.tolerance:
                print(f"Converged at iteration {ep}")
                break

        return [self.coeff, self.intercept, self.loss_history]


if __name__ == "__main__":
    x_train = np.random.rand(10, 5)
    y_train = np.random.rand(10, 1)
    gd = GradientDescentBatch(epochs=10, norm="elastic")
    a, b = gd.forward(x_train, y_train)
    print(a)
    print(b)
