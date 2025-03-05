# ------------------------------------ utf-8 encoding -------------------------------
# this file contains implementation of gaussian mixture model
import numpy as np
import random
from models.basemodel import BaseModel
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


class GaussianMixtureModelFast(BaseModel):

    def __init__(
        self,
        num_cluster: int = 2,
        max_iteration: int = 100,
        tol: float = 1e-6,
        use_parallel: bool = False,
        init_method: str = "default",
    ):
        """
        Gaussian Mixture Model with Expectation-Maximization.
        Parameters:
        - num_cluster (int): Number of clusters.
        - max_iteration (int): Maximum iterations before stopping.
        - tol (float): Convergence threshold based on log-likelihood.
        - use_parallel (bool): Enable parallel computation for likelihood estimation.
        - init_method (str): Initialization method - "random" or "kmeans" or "default".
        """
        super().__init__()
        self.num_cluster = num_cluster
        self.max_iterations = max_iteration
        self.tol = tol
        self.use_parallel = use_parallel
        self.init_method = init_method.lower()

    def __initialize_parameters(self, x: np.ndarray):
        """Initialize means, covariances, and mixing coefficients."""
        self.num_data_points, self.dimension = x.shape

        if self.init_method == "kmeans":
            # Use K-Means for better initialization
            kmeans = KMeans(n_clusters=self.num_cluster, n_init=10, random_state=42)
            labels = kmeans.fit_predict(x)
            self.mu = kmeans.cluster_centers_
        elif self.init_method == "random":
            # Randomly select points as initial means
            random_indices = np.random.choice(
                self.num_data_points, self.num_cluster, replace=False
            )
            self.mu = x[random_indices]
        elif self.init_method == "default":
            self.theta = np.full(
                shape=self.num_cluster, fill_value=1 / self.num_cluster
            )
            self.weight = np.zeros(x.shape)

            # self.mu = [
            #     x[random_index]
            #     for random_index in random.sample(0, len(x), self.num_cluster)
            # ]
            self.mu = [x[i] for i in random.sample(range(len(x)), self.num_cluster)]

        else:
            raise ValueError("Invalid init_method. Choose 'random' or 'kmeans'.")

        # Initialize covariance matrices (with small regularization)
        self.sigma = [
            np.cov(x.T) + np.eye(self.dimension) * 1e-6 for _ in range(self.num_cluster)
        ]

        # Initialize mixing coefficients equally
        self.theta = np.ones(self.num_cluster) / self.num_cluster

    def __compute_likelihood(self, x: np.ndarray, i: int):
        """Compute likelihood for Gaussian i (supports parallelization)."""
        return multivariate_normal(mean=self.mu[i], cov=self.sigma[i]).pdf(x)

    def predict_probability(self, x: np.ndarray):
        """E-Step: Compute responsibility matrix (gamma)."""
        if self.use_parallel:
            # Compute likelihoods in parallel
            likelihoods = np.array(
                Parallel(n_jobs=-1)(
                    delayed(self.__compute_likelihood)(x, i)
                    for i in range(self.num_cluster)
                )
            ).T
        else:
            # Sequential computation
            likelihoods = np.array(
                [
                    multivariate_normal(mean=self.mu[i], cov=self.sigma[i]).pdf(x)
                    for i in range(self.num_cluster)
                ]
            ).T

        weighted_likelihoods = likelihoods * self.theta
        total_likelihood = weighted_likelihoods.sum(axis=1, keepdims=True)

        # Avoid division errors
        return np.divide(
            weighted_likelihoods, total_likelihood, where=total_likelihood != 0
        )

    def __update_parameters(self, x: np.ndarray):
        """M-Step: Update Gaussian parameters."""
        for i in range(self.num_cluster):
            weight = self.responsibilities[:, [i]]
            sum_weight = weight.sum()

            # Update mean
            self.mu[i] = (x * weight).sum(axis=0) / sum_weight

            # Update covariance (with small regularization)
            diff = x - self.mu[i]
            self.sigma[i] = (diff.T @ (diff * weight)) / sum_weight + np.eye(
                self.dimension
            ) * 1e-6

            # Update mixing coefficients
            self.theta[i] = sum_weight / self.num_data_points

    def forward(self, x: np.ndarray, y: np.ndarray = None):
        """Train the GMM using Expectation-Maximization (EM) until convergence."""
        self.__initialize_parameters(x)
        prev_log_likelihood = -np.inf  # Initial log-likelihood
        iteration = 0
        while True:
            # E-Step: Compute responsibilities
            self.responsibilities = self.predict_probability(x)

            # Compute log-likelihood
            log_likelihood = np.sum(np.log(self.responsibilities.sum(axis=1)))

            # Check for convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"Converged at iteration {iteration}.")
                break
            prev_log_likelihood = log_likelihood

            # M-Step: Update parameters
            self.__update_parameters(x)
            iteration += 1

            if self.max_iterations and self.max_iterations < iteration:
                print(f"Reached maximumum iteration")
                break

    def predict(self, x: np.ndarray):
        """Assigns each point to the most probable Gaussian cluster."""
        return np.argmax(self.predict_probability(x), axis=1)


class GaussianMixtureModel(BaseModel):

    def __init__(
        self, num_cluster: int = 2, max_iteration: int = 100, tolerence: float = 1e-6
    ):
        super().__init__()
        self.num_cluster = num_cluster
        self.max_iterations = max_iteration
        self.tolerence = tolerence

    def __initialize_weights(self, x: np.ndarray):
        self.num_data_points = x.shape[0]
        self.dimension = x.shape[1]

        self.theta = np.full(shape=self.num_cluster, fill_value=1 / self.num_cluster)
        self.weight = np.zeros(x.shape)

        self.mu = [x[i] for i in random.sample(range(len(x)), self.num_cluster)]
        self.sigma = [np.cov(x.T) for _ in range(self.num_cluster)]

    def forward(self, x: np.ndarray, y: np.ndarray):
        self.__initialize_weights(x)
        for idx in range(self.max_iterations):
            self.estimation(x)
            self.maximization(x)

    def predict_probability(self, x: np.ndarray):

        likelihood = np.zeros((len(x), self.num_cluster))
        for i in range(self.num_cluster):
            dist = multivariate_normal(mean=self.mu[i], cov=self.sigma[i])
            likelihood[:, i] = dist.pdf(x)

        op1 = likelihood * self.theta
        op2 = op1.sum(axis=1)[:, np.newaxis]
        weight = op1 / op2
        return weight

    def estimation(self, x: np.ndarray):
        self.weight = self.predict_probability(x)
        self.theta = self.weight.mean(axis=0)

    def maximization(self, x: np.ndarray):

        for i in range(self.num_cluster):
            weight = self.weight[:, [i]]
            net_weight = weight.sum()
            self.mu[i] = (x * weight).sum(axis=0) / net_weight
            self.sigma[i] = np.cov(
                x.T, aweights=(weight / net_weight).flatten(), bias=True
            )

    def predict(self, x: np.ndarray):
        return np.argmax(self.predict_probability(x), axis=1)
