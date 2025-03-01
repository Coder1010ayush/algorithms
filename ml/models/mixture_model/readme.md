# Gaussian Mixture Model (GMM)

## Introduction
Gaussian Mixture Model (GMM) is a probabilistic clustering algorithm that assumes data is generated from a mixture of multiple Gaussian distributions with unknown parameters. It is widely used for clustering tasks and density estimation.

This document describes the implementation of two variants of GMM:
1. **GaussianMixtureModelFast**: An optimized version with parallel computation.
2. **GaussianMixtureModel**: A simpler version without parallelization.

Both implementations use the Expectation-Maximization (EM) algorithm to iteratively estimate the parameters of the Gaussian distributions.

---

## 1. GaussianMixtureModelFast

### Class Definition
```python
class GaussianMixtureModelFast(BaseModel):
```
This is an optimized version of GMM that supports parallel computation for likelihood estimation.

### Parameters
- `num_cluster (int)`: Number of Gaussian components (clusters).
- `max_iteration (int)`: Maximum number of iterations for EM.
- `tol (float)`: Convergence threshold for log-likelihood.
- `use_parallel (bool)`: Enables parallel computation for likelihood estimation.
- `init_method (str)`: Initialization method, can be:
  - `"random"`: Selects random points as initial means.
  - `"kmeans"`: Uses K-Means for better initialization.
  - `"default"`: Uses random sampling for initialization.

### Methods
#### `__initialize_parameters(x: np.ndarray)`
Initializes the mean (`mu`), covariance (`sigma`), and mixing coefficients (`theta`).
- Uses K-Means or random initialization.
- Covariance matrices are initialized with a small regularization term to prevent singularity.

#### `__compute_likelihood(x: np.ndarray, i: int)`
Computes the probability density function (PDF) for Gaussian component `i`.
- Uses `scipy.stats.multivariate_normal`.
- Supports parallel execution for speed optimization.

#### `predict_probability(x: np.ndarray)`
Computes the responsibility matrix (E-step).
- Uses parallelization if `use_parallel=True`.
- Computes weighted likelihoods and normalizes probabilities.

#### `__update_parameters(x: np.ndarray)`
Updates the Gaussian parameters (M-step).
- Updates mean (`mu`), covariance (`sigma`), and mixing coefficients (`theta`).

#### `forward(x: np.ndarray, y: np.ndarray = None)`
Trains the GMM using the EM algorithm.
- Iteratively performs E-step and M-step until convergence or maximum iterations.
- Stops when the log-likelihood change is below `tol`.

#### `predict(x: np.ndarray)`
Assigns each data point to the most probable Gaussian cluster based on the highest responsibility value.

---

## 2. GaussianMixtureModel

### Class Definition
```python
class GaussianMixtureModel(BaseModel):
```
A simpler version of GMM without parallelization.

### Parameters
- `num_cluster (int)`: Number of Gaussian components.
- `max_iteration (int)`: Maximum number of EM iterations.
- `tolerance (float)`: Convergence threshold.

### Methods
#### `__initialize_weights(x: np.ndarray)`
Initializes the parameters:
- Mean (`mu`) using random selection.
- Covariance (`sigma`) using the empirical covariance matrix.
- Mixing coefficients (`theta`).

#### `predict_probability(x: np.ndarray)`
Computes the posterior probability (responsibilities) of each data point belonging to each Gaussian.

#### `estimation(x: np.ndarray)`
Performs the E-step:
- Computes the new responsibilities based on the current parameters.

#### `maximization(x: np.ndarray)`
Performs the M-step:
- Updates the mean (`mu`), covariance (`sigma`), and mixing coefficients (`theta`).

#### `forward(x: np.ndarray, y: np.ndarray)`
Trains the GMM using EM until convergence.
- Alternates between E-step and M-step for `max_iterations`.

#### `predict(x: np.ndarray)`
Predicts the cluster assignment for each data point.

---

## Key Differences Between Variants
| Feature | GaussianMixtureModelFast | GaussianMixtureModel |
|---------|--------------------------|----------------------|
| Parallel Computation | ✅ (Joblib) | ❌ |
| Initialization Options | Random, K-Means, Default | Random |
| Convergence Check | Log-likelihood | Fixed iterations |
| Speed | Faster (Parallelization) | Slower |

---

