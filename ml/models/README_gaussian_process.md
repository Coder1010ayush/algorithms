# Gaussian processes (`ml/models/gaussian_process.py`)

## Kernels

Each kernel maps two matrices to a covariance matrix of shape `(len(X1), len(X2))`.

| Name | Function | Parameters |
|---|---|---|
| `rbf` | `rbf_kernel` | `sigma`, `length_scale` |
| `matern` | `matern_kernel` | `sigma`, `length_scale`, `nu` (0.5, 1.5, 2.5 closed form; otherwise Bessel) |
| `periodic` | `periodic_kernel` | `sigma`, `length_scale`, `period` |
| `linear` | `linear_kernel` | `sigma_b`, `sigma_v` |
| `rational_quadratic` | `rational_quadratic_kernel` | `sigma`, `length_scale`, `alpha` |

## Regression

`GaussianProcessRegression(kernel="rbf", noise=1e-5, kernel_params=None)`

- `fit(X, y)`: Cholesky of `K + noise I`, stores `alpha = K^-1 y`.
- `predict(X_test, return_std=False)`: posterior mean, optionally the predictive standard deviation.
- `log_marginal_likelihood()`.
- `optimize_hyperparameters(names=("length_scale", "sigma"))`: maximises the marginal likelihood in log space with L-BFGS-B.

## Classification

`GaussianProcessClassification` is a binary classifier using a Laplace approximation to the logistic-likelihood posterior (Newton iterations on the latent function). `MultiClassGaussianProcessClassification` wraps it one-vs-rest and softmax-normalises the per-class probabilities. Both expose `predict` and `predict_proba`.
