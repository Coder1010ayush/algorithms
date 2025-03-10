# Optimizers Documentation

## Overview
This module implements various optimization algorithms used for training deep learning models. Each optimizer adjusts model parameters based on computed gradients to minimize loss functions effectively.

## Base Class
### `BaseOptimiser`
#### Description
The base class for all optimizers, defining a common interface.
#### Methods
- `__init__(lr: float = 1e-4)`: Initializes the optimizer with a learning rate.
- `zero_grad(params)`: Clears the gradients of all parameters (must be implemented by subclasses).
- `step(params)`: Updates parameters based on gradients (must be implemented by subclasses).

---
## Optimizers
### `GradientOptimiser`
#### Description
Implements basic gradient descent optimization.
#### Formula
  ```
  θ = θ - η ∇J(θ)
  ```
#### Methods
- `zero_grad(params)`: Resets gradients to zero.
- `step(params)`: Updates parameters using gradient descent.

---
### `AdamOptimizer`
#### Description
Implements the Adam optimization algorithm, which adapts learning rates for each parameter.
#### Formula
  ```
  m_t = β₁m_(t-1) + (1 - β₁)g_t
  v_t = β₂v_(t-1) + (1 - β₂)g_t²
  m̂_t = m_t / (1 - β₁ᵗ), v̂_t = v_t / (1 - β₂ᵗ)
  θ = θ - (η / √(v̂_t + ε)) m̂_t
  ```
#### Parameters
- `lr`: Learning rate (default: 0.001)
- `beta1`: Decay rate for the first moment estimate (default: 0.9)
- `beta2`: Decay rate for the second moment estimate (default: 0.999)
- `epsilon`: Small constant to prevent division by zero (default: 1e-8)
#### Methods
- `zero_grad(params)`: Resets gradients to zero.
- `step(params)`: Updates parameters using bias-corrected first and second moment estimates.

---
### `SGDOptimizer`
#### Description
Implements Stochastic Gradient Descent (SGD) with momentum.
#### Formula
  ```
  v_t = γv_(t-1) + η∇J(θ)
  θ = θ - v_t
  ```
#### Parameters
- `lr`: Learning rate (default: 0.01)
- `momentum`: Momentum factor (default: 0.0)
#### Methods
- `zero_grad(params)`: Resets gradients to zero.
- `step(params)`: Updates parameters using SGD with momentum.

---
### `RMSPropOptimizer`
#### Description
Implements RMSProp, which scales learning rates based on recent gradient magnitudes.
#### Formula
  ```
  E[g²]_t = γE[g²]_(t-1) + (1 - γ)g_t²
  θ = θ - (η / √(E[g²]_t + ε)) g_t
  ```
#### Parameters
- `lr`: Learning rate (default: 0.001)
- `beta`: Decay rate (default: 0.9)
- `epsilon`: Small constant to prevent division by zero (default: 1e-8)
#### Methods
- `zero_grad(params)`: Resets gradients to zero.
- `step(params)`: Updates parameters using a moving average of squared gradients.

---
### `AdaGradOptimizer`
#### Description
Implements the AdaGrad optimizer, which adapts learning rates for each parameter.
#### Formula
  ```
  G_t = G_(t-1) + g_t²
  θ = θ - (η / √(G_t + ε)) g_t
  ```
#### Parameters
- `lr`: Learning rate (default: 0.01)
- `epsilon`: Small constant to prevent division by zero (default: 1e-8)
#### Methods
- `zero_grad(params)`: Resets gradients to zero.
- `step(params)`: Updates parameters using accumulated squared gradients.

---
### `AdaDeltaOptimizer`
#### Description
Implements the AdaDelta optimization algorithm, which adapts learning rates dynamically.
#### Formula
  ```
  E[g²]_t = ρE[g²]_(t-1) + (1 - ρ)g_t²
  θ = θ - (η / √(E[g²]_t + ε)) g_t
  ```
#### Parameters
- `lr`: Learning rate (default: 1.0)
- `rho`: Decay rate (default: 0.95)
- `epsilon`: Small constant to prevent division by zero (default: 1e-6)
#### Methods
- `zero_grad(params)`: Resets gradients to zero.
- `step(params)`: Updates parameters using an adaptive learning rate.

---
### `NadamOptimizer`
#### Description
Implements Nadam, a variant of Adam with Nesterov momentum.
#### Formula
  ```
  m_t = β₁m_(t-1) + (1 - β₁)g_t
  v_t = β₂v_(t-1) + (1 - β₂)g_t²
  θ = θ - (η / √(v_t + ε)) (β₁m_t + (1 - β₁)g_t / (1 - β₁ᵗ))
  ```
#### Parameters
- `lr`: Learning rate (default: 0.001)
- `beta1`: Decay rate for first moment estimate (default: 0.9)
- `beta2`: Decay rate for second moment estimate (default: 0.999)
- `epsilon`: Small constant to prevent division by zero (default: 1e-8)
#### Methods
- `zero_grad(params)`: Resets gradients to zero.
- `step(params)`: Updates parameters using Nadam optimization.

---
### `NAGOptimizer`
#### Description
Implements Nesterov Accelerated Gradient (NAG), an improved version of momentum-based SGD.
#### Formula
  ```
  v_t = γv_(t-1) + η∇J(θ - γv_(t-1))
  θ = θ - v_t
  ```
#### Parameters
- `lr`: Learning rate (default: 0.01)
- `momentum`: Momentum factor (default: 0.9)
#### Methods
- `zero_grad(params)`: Resets gradients to zero.
- `step(params)`: Updates parameters using Nesterov lookahead gradients.

## Usage Example
```python
params = [Param(np.array([1.0, 2.0]), np.array([0.1, 0.2]))] 
optimizer = AdamOptimizer(lr=0.001)
optimizer.zero_grad(params)
optimizer.step(params)
```
## Tips for Choosing Optimizers
- **SGD:** Simple and effective for large datasets.  
- **Momentum:** Helps overcome local minima efficiently.  
- **NAG:** Better than Momentum for oscillatory terrains.  
- **Adagrad:** Good for sparse data but suffers from aggressive decay.  
- **Adadelta & RMSProp:** Improve on Adagrad’s decay issue.  
- **Adam:** Most popular choice due to adaptive learning rates.  
- **Nadam:** Ideal for tasks needing both momentum and lookahead capabilities.


