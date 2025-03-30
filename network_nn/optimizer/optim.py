# ------------------------------ *utf-8 encoding* ----------------------------
import os
import json
import numpy as np
import math
import random


class BaseOptimiser:
    def __init__(self, lr: float = 1e-4):
        self.lr = lr
    
    def __set_lr(self , lr):
        self.lr = lr

    def zero_grad(self, params):
        raise NotImplementedError("zero_grad is not implemented for this optimiser.")

    def step(self, params):
        raise NotImplementedError("step method is not implemented for this optimiser.")


class GradientOptimiser(BaseOptimiser):
    def __init__(self, lr=0.0001):
        super().__init__(lr)

    def zero_grad(self, params):
        for param in params:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)

    def step(self, params):
        for param in params:
            if param.grad is not None:
                param.data -= self.lr * param.grad


class AdamOptimizer(BaseOptimiser):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def zero_grad(self, params):
        for param in params:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)

    def step(self, params):
        self.t += 1
        for idx, param in enumerate(params):
            if param.grad is None:
                continue

            if idx not in self.m:
                self.m[idx] = np.zeros_like(param.grad)
                self.v[idx] = np.zeros_like(param.grad)

            # Update biased first moment estimate
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * param.grad
            # Update biased second raw moment estimate
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (param.grad**2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[idx] / (1 - self.beta1**self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[idx] / (1 - self.beta2**self.t)

            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class SGDOptimizer(BaseOptimiser):
    def __init__(self, lr=0.01, momentum=0.0):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity = {}

    def zero_grad(self, params):
        for param in params:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)

    def step(self, params):
        for idx, param in enumerate(params):
            if param.grad is None:
                continue

            if idx not in self.velocity:
                self.velocity[idx] = np.zeros_like(param.grad)

            # Update velocity and parameters
            self.velocity[idx] = (
                self.momentum * self.velocity[idx] + self.lr * param.grad
            )
            param.data -= self.velocity[idx]


class RMSPropOptimizer(BaseOptimiser):
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.epsilon = epsilon
        self.s = {}

    def zero_grad(self, params):
        for param in params:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)

    def step(self, params):
        for idx, param in enumerate(params):
            if param.grad is None:
                continue

            if idx not in self.s:
                self.s[idx] = np.zeros_like(param.grad)

            self.s[idx] = self.beta * self.s[idx] + (1 - self.beta) * (param.grad**2)
            param.data -= self.lr * param.grad / (np.sqrt(self.s[idx]) + self.epsilon)


class AdaGradOptimizer(BaseOptimiser):
    def __init__(self, lr=0.01, epsilon=1e-8):
        super().__init__(lr)
        self.epsilon = epsilon
        self.s = {}

    def zero_grad(self, params):
        for param in params:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)

    def step(self, params):
        for idx, param in enumerate(params):
            if param.grad is None:
                continue

            if idx not in self.s:
                self.s[idx] = np.zeros_like(param.grad)

            self.s[idx] += param.grad**2
            param.data -= self.lr * param.grad / (np.sqrt(self.s[idx]) + self.epsilon)


class AdaDeltaOptimizer(BaseOptimiser):
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6):
        super().__init__(lr)
        self.rho = rho
        self.epsilon = epsilon
        self.Eg = {}
        self.Edx = {}

    def zero_grad(self, params):
        for param in params:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)

    def step(self, params):
        for idx, param in enumerate(params):
            if param.grad is None:
                continue

            if idx not in self.Eg:
                self.Eg[idx] = np.zeros_like(param.grad)
                self.Edx[idx] = np.zeros_like(param.grad)

            self.Eg[idx] = self.rho * self.Eg[idx] + (1 - self.rho) * (param.grad**2)

            dx = (
                -np.sqrt(self.Edx[idx] + self.epsilon)
                / np.sqrt(self.Eg[idx] + self.epsilon)
                * param.grad
            )

            self.Edx[idx] = self.rho * self.Edx[idx] + (1 - self.rho) * (dx**2)
            param.data += dx


class NadamOptimizer(BaseOptimiser):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def zero_grad(self, params):
        for param in params:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)

    def step(self, params):
        self.t += 1
        for idx, param in enumerate(params):
            if param.grad is None:
                continue

            if idx not in self.m:
                self.m[idx] = np.zeros_like(param.grad)
                self.v[idx] = np.zeros_like(param.grad)

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * param.grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (param.grad**2)

            # Bias-corrected first moment estimate
            m_hat = self.m[idx] / (1 - self.beta1**self.t)
            # Bias-corrected second raw moment estimate
            v_hat = self.v[idx] / (1 - self.beta2**self.t)

            # Nesterov update
            nesterov_m = self.beta1 * m_hat + (1 - self.beta1) * param.grad
            param.data -= self.lr * nesterov_m / (np.sqrt(v_hat) + self.epsilon)


class NAGOptimizer(BaseOptimiser):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity = {}

    def zero_grad(self, params):
        for param in params:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)

    def step(self, params):
        for idx, param in enumerate(params):
            if param.grad is None:
                continue

            if idx not in self.velocity:
                self.velocity[idx] = np.zeros_like(param.grad)

            lookahead_param = param.data - self.momentum * self.velocity[idx]
            original_param = param.data
            param.data = lookahead_param

            lookahead_grad = param.grad
            param.data = original_param
            self.velocity[idx] = (
                self.momentum * self.velocity[idx] + self.lr * lookahead_grad
            )
            param.data -= self.velocity[idx]
