
# Optimizers in Machine Learning

This repository provides **detailed explanations** of various optimizers used in training neural networks, including their update formulas and guidelines on when to use them.

---

## Table of Contents
1. [Gradient Descent (SGD)](#1-gradient-descent-sgd)
2. [Momentum Optimizer](#2-momentum-optimizer)
3. [Nesterov Accelerated Gradient (NAG)](#3-nesterov-accelerated-gradient-nag)
4. [Adagrad Optimizer](#4-adagrad-optimizer)
5. [Adadelta Optimizer](#5-adadelta-optimizer)
6. [RMSProp Optimizer](#6-rmsprop-optimizer)
7. [Adam Optimizer](#7-adam-optimizer)
8. [Nadam Optimizer](#8-nadam-optimizer)
9. [Tips for Choosing Optimizers](#-tips-for-choosing-optimizers)

---

## 1. Gradient Descent (SGD)
- **Formula:**  
  ```
  θ = θ - η ∇J(θ)
  ```
- **Description:**  
  Vanilla SGD updates weights using the gradient of the loss function with respect to the parameters.

---

## 2. Momentum Optimizer
- **Formula:**  
  ```
  v_t = γv_(t-1) + η∇J(θ)
  θ = θ - v_t
  ```
- **Description:**  
  Adds momentum to the updates to accelerate convergence.

---

## 3. Nesterov Accelerated Gradient (NAG)
- **Formula:**  
  ```
  v_t = γv_(t-1) + η∇J(θ - γv_(t-1))
  θ = θ - v_t
  ```
- **Description:**  
  Looks ahead before applying the gradient, reducing oscillations.

---

## 4. Adagrad Optimizer
- **Formula:**  
  ```
  G_t = G_(t-1) + g_t²
  θ = θ - (η / √(G_t + ε)) g_t
  ```
- **Description:**  
  Adapts learning rate based on historical gradients.

---

## 5. Adadelta Optimizer
- **Formula:**  
  ```
  E[g²]_t = ρE[g²]_(t-1) + (1 - ρ)g_t²
  θ = θ - (η / √(E[g²]_t + ε)) g_t
  ```
- **Description:**  
  An extension of Adagrad that reduces aggressive decay.

---

## 6. RMSProp Optimizer
- **Formula:**  
  ```
  E[g²]_t = γE[g²]_(t-1) + (1 - γ)g_t²
  θ = θ - (η / √(E[g²]_t + ε)) g_t
  ```
- **Description:**  
  Fixes Adagrad's decaying learning rate issue.

---

## 7. Adam Optimizer
- **Formula:**  
  ```
  m_t = β₁m_(t-1) + (1 - β₁)g_t
  v_t = β₂v_(t-1) + (1 - β₂)g_t²
  m̂_t = m_t / (1 - β₁ᵗ), v̂_t = v_t / (1 - β₂ᵗ)
  θ = θ - (η / √(v̂_t + ε)) m̂_t
  ```
- **Description:**  
  Combines momentum and RMSProp.

---

## 8. Nadam Optimizer 🚀
- **Formula:**  
  ```
  m_t = β₁m_(t-1) + (1 - β₁)g_t
  v_t = β₂v_(t-1) + (1 - β₂)g_t²
  θ = θ - (η / √(v_t + ε)) (β₁m_t + (1 - β₁)g_t / (1 - β₁ᵗ))
  ```
- **Description:**  
  Adam with Nesterov momentum.

---

## Tips for Choosing Optimizers
- **SGD:** Simple and effective for large datasets.  
- **Momentum:** Helps overcome local minima efficiently.  
- **NAG:** Better than Momentum for oscillatory terrains.  
- **Adagrad:** Good for sparse data but suffers from aggressive decay.  
- **Adadelta & RMSProp:** Improve on Adagrad’s decay issue.  
- **Adam:** Most popular choice due to adaptive learning rates.  
- **Nadam:** Ideal for tasks needing both momentum and lookahead capabilities.

---