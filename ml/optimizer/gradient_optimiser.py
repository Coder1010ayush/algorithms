# ------------------------------------- utf-8 encoding ----------------------------------
# this is the main file which contains a general architecture for defining optimizer
import numpy as np


class GradientOptimizer:
    def __init__(
        self,
        model,
        learning_rate=0.01,
        epochs=100,
        tolerance=1e-6,
        debug=False,
        norm="",
        l1_lambda=0.2,
        l2_lambda=0.1,
        el_param=0.9,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.debug = debug
        self.loss_history = []
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

    def apply_regularization(self, grad_w, coeff):
        if self.norm == "l1":
            return grad_w + self.lambad_l1 * np.sign(coeff)
        elif self.norm == "l2":
            return grad_w + self.lambad_l2 * 2 * coeff
        elif self.norm == "elastic":
            l1_penalty = self.lambad_l1 * np.sign(coeff)
            l2_penalty = self.lambad_l2 * 2 * coeff
            return grad_w + (l1_penalty + l2_penalty) * self.el_param
        return grad_w

    def optimize_batch(self, X, y):
        for epoch in range(self.epochs):
            y_pred = self.model.forward(X)
            loss = self.model.compute_loss(y, y_pred)
            grad_w, grad_b, coeff, intercept = self.model.compute_gradient(X, y, y_pred)

            # applying regulerization
            grad_w = self.apply_regularization(grad_w=grad_w, coeff=coeff)

            # Update model parameters
            self.model.update_parameters(grad_w, grad_b, self.learning_rate)

            self.loss_history.append(loss)

            if self.debug and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={loss:.6f}")

            # Convergence check
            if np.linalg.norm(grad_w) < self.tolerance and abs(grad_b) < self.tolerance:
                print(f"Converged at epoch {epoch}")
                break
        return self.model

    def optimize_stochastic(self, X, y):
        num_samples = X.shape[0]

        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(num_samples):
                X_i = X[i].reshape(1, -1)
                y_i = np.array([y[i]])

                y_pred = self.model.forward(X_i)
                loss = self.model.compute_loss(y_i, y_pred)
                grad_w, grad_b, coeff, intercept = self.model.compute_gradient(
                    X_i, y_i, y_pred
                )

                # applying regulerization
                grad_w = self.apply_regularization(grad_w=grad_w, coeff=coeff)

                # Update parameters after each sample
                self.model.update_parameters(grad_w, grad_b, self.learning_rate)

                total_loss += loss

            avg_loss = total_loss / num_samples
            self.loss_history.append(avg_loss)

            if self.debug and epoch % 10 == 0:
                print(f"Epoch {epoch}: Avg Loss={avg_loss:.6f}")

            # Convergence check based on average gradient norms
            if np.linalg.norm(grad_w) < self.tolerance and abs(grad_b) < self.tolerance:
                print(f"Converged at epoch {epoch}")
                break

        return self.model

    def show_loss_hist(self):
        for idx, itm in enumerate(self.loss_history):
            print(f"Epoch {idx+1} Loss is {itm}")
