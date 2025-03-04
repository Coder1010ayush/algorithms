# Gaussian Process Implementation

## Overview
This codebase implements Gaussian Processes (GP) for both regression and classification. It includes multiple kernel functions and provides hyperparameter optimization. The GP framework is useful for making predictions with uncertainty estimation.

## Dependencies
The following Python libraries are required:
- `numpy` for numerical operations
- `math` for mathematical functions
- `scipy.spatial.distance` for computing Euclidean distances
- `scipy.special` for Bessel functions and Gamma function
- `scipy.linalg` for Cholesky decomposition and solving linear systems
- `scipy.optimize` for hyperparameter optimization
- `scipy.special.expit` for logistic sigmoid function
- `scipy.special.softmax` for multi-class classification probabilities

## Kernel Functions
Gaussian Processes rely on kernel functions to model similarities between data points. The following kernels are implemented:

### 1. Matern Kernel
```python
def matern_kernel(x: np.ndarray, x_prime: np.ndarray, length_scale=1.0, nu=1.5)
```
- `nu=0.5`: Exponential kernel (absolute exponential)
- `nu=1.5`: Matern kernel with smoothness 1.5
- `nu=2.5`: Matern kernel with smoothness 2.5
- General case uses Bessel function (`bessel_kv`) and Gamma function

### 2. Radial Basis Function (RBF) Kernel
```python
def rbf_kernel(x: np.ndarray, x_prime, sigma=1.0, length_scale=1.0)
```
- Also known as the squared exponential kernel
- Provides smooth function approximation

### 3. Periodic Kernel
```python
def periodic_kernel(x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, period=1.0)
```
- Captures periodic patterns in data

### 4. Linear Kernel
```python
def linear_kernel(x: np.ndarray, x_prime: np.ndarray, sigma_b=1.0, sigma_v=1.0)
```
- Suitable for modeling linear relationships

### 5. Rational Quadratic Kernel
```python
def rational_quadratic_kernel(x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, alpha=1.0)
```
- A mix between RBF and polynomial kernels

### 6. White Noise Kernel
```python
def white_noise_kernel(x: np.ndarray, x_prime: np.ndarray, sigma=1.0)
```
- Adds independent noise to the model

## Gaussian Process (GP) Classes

### 1. GaussianProcess (Base GP Model)
#### Constructor
```python
class GaussianProcess:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None)
```
- Supports RBF, Matern, Periodic, and Linear kernels
- `noise` parameter controls the observation noise
- `kernel_params` allows custom kernel settings

#### Methods
- **`fit(X, y)`**: Trains the GP model
- **`predict(X_test, return_std=False)`**: Returns predictions and optionally uncertainty estimates
- **`log_marginal_likelihood()`**: Computes model likelihood
- **`optimize_hyperparameters()`**: Optimizes kernel hyperparameters via Maximum Likelihood Estimation (MLE)

### 2. GaussianProcessRegression (GPR)
This is a specific implementation of GP for regression.
```python
class GaussianProcessRegression:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None)
```
- Trains on continuous target values
- Uses Cholesky decomposition for numerical stability

#### Methods
- **`fit(X, y)`**: Learns from training data
- **`predict(X_test, return_std=False)`**: Predicts regression values

### 3. GaussianProcessClassification (GPC)
Binary classification using Gaussian Processes.
```python
class GaussianProcessClassification:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None, max_iter=10)
```
- Uses the Laplace approximation to handle the non-Gaussian posterior

#### Methods
- **`fit(X, y)`**: Trains on binary classification labels (`0,1` converted to `-1,1`)
- **`predict(X_test)`**: Outputs class probabilities using the sigmoid function

### 4. MultiClassGaussianProcessClassification (Multi-Class GPC)
Extends `GaussianProcessClassification` to multi-class problems using one-vs-rest (OvR) approach.
```python
class MultiClassGaussianProcessClassification:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None, max_iter=10)
```

#### Methods
- **`fit(X, y)`**: Fits multiple binary classifiers for each class
- **`predict(X_test)`**: Uses softmax over individual class probabilities

## Example Usage

### Regression Example
```python
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([1, 2, 1.5, 3.5, 2.8])

gp = GaussianProcessRegression(kernel="rbf", kernel_params={"length_scale": 1.0})
gp.fit(X_train, y_train)
X_test = np.array([[2.5], [3.5]])
predictions, std_dev = gp.predict(X_test, return_std=True)
```

### Binary Classification Example
```python
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([0, 1, 0, 1, 0])

gpc = GaussianProcessClassification(kernel="rbf", kernel_params={"length_scale": 1.0})
gpc.fit(X_train, y_train)
X_test = np.array([[2.5], [3.5]])
pred_probs = gpc.predict(X_test)
```

### Multi-Class Classification Example
```python
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_train = np.array([0, 1, 2, 1, 0])

gpc_multi = MultiClassGaussianProcessClassification(kernel="rbf")
gpc_multi.fit(X_train, y_train)
X_test = np.array([[2.5, 3.5]])
predictions, probabilities = gpc_multi.predict(X_test)
```