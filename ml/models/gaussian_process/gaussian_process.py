# ---------------------------------------- utf-8 encoding ---------------------------------
# this file contains gaussian process implementation [ distribution over function ]
import numpy as np
import math
from scipy.spatial.distance import euclidean
from scipy.special import kv as bessel_kv
from scipy.special import gamma
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit, softmax


def matern_kernel(x: np.ndarray, x_prime: np.ndarray, length_scale=1.0, nu=1.5):
    d = euclidean(x, x_prime)
    if nu == 0.5:
        return np.exp(-d / length_scale)
    elif nu == 1.5:
        return (1 + np.sqrt(3) * d / length_scale) * np.exp(
            -np.sqrt(3) * d / length_scale
        )
    elif nu == 2.5:
        return (
            1 + np.sqrt(5) * d / length_scale + 5 * d**2 / (3 * length_scale**2)
        ) * np.exp(-np.sqrt(5) * d / length_scale)
    else:
        factor = (2 ** (1 - nu)) / gamma(nu)
        scaled_d = np.sqrt(2 * nu) * d / length_scale
        return factor * (scaled_d**nu) * bessel_kv(nu, scaled_d)


def rbf_kernel(x: np.ndarray, x_prime, sigma=1.0, length_scale=1.0):
    return sigma**2 * np.exp(-0.5 * np.linalg.norm(x - x_prime) ** 2 / length_scale**2)


def periodic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, period=1.0
):
    return sigma**2 * np.exp(
        -2 * np.sin(np.pi * np.linalg.norm(x - x_prime) / period) ** 2 / length_scale**2
    )


def linear_kernel(x: np.ndarray, x_prime: np.ndarray, sigma_b=1.0, sigma_v=1.0):
    return sigma_b**2 + sigma_v**2 * np.dot(x, x_prime)


def rational_quadratic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, alpha=1.0
):
    return sigma**2 * (
        1 + np.linalg.norm(x - x_prime) ** 2 / (2 * alpha * length_scale**2)
    ) ** (-alpha)


def white_noise_kernel(x: np.ndarray, x_prime: np.ndarray, sigma=1.0):
    return sigma**2 if np.array_equal(x, x_prime) else 0


class GaussianProcess:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None):
        self.kernel_name = kernel
        self.noise = noise
        self.kernel_params = kernel_params if kernel_params else {}
        self.X_train = None
        self.y_train = None
        self.K = None  # Covariance matrix

    def _select_kernel(self, X1, X2):
        # Selects and computes the kernel matrix based on user choice.
        if self.kernel_name == "rbf":
            return rbf_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_name == "matern":
            return matern_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_name == "periodic":
            return periodic_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_name == "linear":
            return linear_kernel(X1, X2, **self.kernel_params)
        else:
            raise ValueError(
                "Unsupported kernel. Choose from ['rbf', 'matern', 'periodic', 'linear']."
            )

    def fit(self, X, y):
        # Train the Gaussian Process Model with training data X and observations y.
        self.X_train = X
        self.y_train = y
        self.K = self._select_kernel(self.X_train, self.X_train) + self.noise * np.eye(
            len(self.X_train)
        )

        # Compute Cholesky decomposition for stable inversion
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve_triangular(
            self.L.T, solve_triangular(self.L, self.y_train, lower=True)
        )

    def predict(self, X_test, return_std=False):
        # Predict mean and optionally variance at test points X_test.
        X_test = np.asarray(X_test)
        K_s = self._select_kernel(self.X_train, X_test)
        K_ss = self._select_kernel(X_test, X_test) + self.noise * np.eye(len(X_test))

        # Compute predictive mean
        mu = K_s.T @ self.alpha

        # Compute predictive variance
        v = solve_triangular(self.L, K_s, lower=True)
        cov = K_ss - v.T @ v

        return (mu, np.sqrt(np.diag(cov))) if return_std else mu

    def log_marginal_likelihood(self):
        # Compute the log marginal likelihood for model selection.
        return (
            -0.5 * (self.y_train.T @ self.alpha)
            - np.sum(np.log(np.diag(self.L)))
            - len(self.X_train) / 2 * np.log(2 * np.pi)
        )

    def optimize_hyperparameters(self):
        # Optimize hyperparameters using Maximum Likelihood Estimation (MLE).

        def objective(params):
            # Convert params to kernel_params
            self.kernel_params = {
                "length_scale": np.exp(params[0]),
                "sigma": np.exp(params[1]),
            }
            self.fit(self.X_train, self.y_train)
            return -self.log_marginal_likelihood()  # Negative for minimization

        # Initial values
        init_params = np.log([1.0, 1.0])  # Log scale for better optimization
        res = minimize(objective, init_params, method="L-BFGS-B")
        self.kernel_params = {
            "length_scale": np.exp(res.x[0]),
            "sigma": np.exp(res.x[1]),
        }


class GaussianProcessRegression:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None):
        self.kernel_name = kernel
        self.noise = noise
        self.kernel_params = kernel_params if kernel_params else {}
        self.X_train = None
        self.y_train = None
        self.K = None

    def _select_kernel(self, X1, X2):
        if self.kernel_name == "rbf":
            return rbf_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_name == "matern":
            return matern_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_name == "periodic":
            return periodic_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_name == "linear":
            return linear_kernel(X1, X2, **self.kernel_params)
        else:
            raise ValueError("Unsupported kernel.")

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.K = self._select_kernel(self.X_train, self.X_train) + self.noise * np.eye(
            len(self.X_train)
        )
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve_triangular(
            self.L.T, solve_triangular(self.L, self.y_train, lower=True)
        )

    def predict(self, X_test, return_std=False):
        X_test = np.asarray(X_test)
        K_s = self._select_kernel(self.X_train, X_test)
        K_ss = self._select_kernel(X_test, X_test) + self.noise * np.eye(len(X_test))
        mu = K_s.T @ self.alpha
        v = solve_triangular(self.L, K_s, lower=True)
        cov = K_ss - v.T @ v
        return (mu, np.sqrt(np.diag(cov))) if return_std else mu


class GaussianProcessClassification:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None, max_iter=10):
        self.kernel_name = kernel
        self.noise = noise
        self.kernel_params = kernel_params if kernel_params else {}
        self.max_iter = max_iter
        self.X_train = None
        self.y_train = None
        self.K = None

    def _select_kernel(self, X1, X2):
        if self.kernel_name == "rbf":
            return rbf_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_name == "matern":
            return matern_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_name == "periodic":
            return periodic_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_name == "linear":
            return linear_kernel(X1, X2, **self.kernel_params)
        else:
            raise ValueError("Unsupported kernel.")

    def fit(self, X, y):
        """Binary classification: y ∈ {0,1}, transformed to {-1,1}"""
        self.X_train = np.asarray(X)
        self.y_train = 2 * np.asarray(y) - 1  # Convert {0,1} → {-1,1}
        self.K = self._select_kernel(self.X_train, self.X_train) + self.noise * np.eye(
            len(self.X_train)
        )

        # Initialize latent function f = 0
        f = np.zeros(len(y))
        for _ in range(self.max_iter):
            pi = expit(f)
            W = np.diag(pi * (1 - pi))
            L = cholesky(np.eye(len(y)) + np.sqrt(W) @ self.K @ np.sqrt(W), lower=True)
            b = W @ f + self.y_train - pi
            a = solve_triangular(L, np.sqrt(W) @ self.K @ b, lower=True)
            f = self.K @ b - np.sqrt(W) @ solve_triangular(L.T, a, lower=False)

        self.f = f
        self.pi = expit(f)

    def predict(self, X_test):
        """Predict class probabilities using the Laplace approximation."""
        X_test = np.asarray(X_test)
        K_s = self._select_kernel(self.X_train, X_test)

        mean_f = K_s.T @ self.pi
        return expit(mean_f)


class MultiClassGaussianProcessClassification:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None, max_iter=10):
        self.kernel_name = kernel
        self.noise = noise
        self.kernel_params = kernel_params if kernel_params else {}
        self.max_iter = max_iter
        self.classes_ = None
        self.binary_gpcs = {}

    def _select_kernel(self, X1, X2):
        """Kernel function selection."""
        if self.kernel_name == "rbf":
            return rbf_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_name == "matern":
            return matern_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_name == "periodic":
            return periodic_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_name == "linear":
            return linear_kernel(X1, X2, **self.kernel_params)
        else:
            raise ValueError("Unsupported kernel.")

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.binary_gpcs = {}

        for cls in self.classes_:
            y_binary = (y == cls).astype(int)
            gpc = GaussianProcessClassification(
                kernel=self.kernel_name,
                noise=self.noise,
                kernel_params=self.kernel_params,
                max_iter=self.max_iter,
            )
            gpc.fit(X, y_binary)
            self.binary_gpcs[cls] = gpc

    def predict(self, X_test):
        probs = np.zeros((len(X_test), len(self.classes_)))

        for i, cls in enumerate(self.classes_):
            probs[:, i] = self.binary_gpcs[cls].predict(X_test)

        probs = softmax(probs, axis=1)
        predictions = self.classes_[
            np.argmax(probs, axis=1)
        ]  # Get the most likely class
        return predictions, probs
