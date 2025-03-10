# ------------------------------ *utf-8 encoding* ----------------------------
import os
import json
import numpy as np
import math
import random


class BaseOptimiser:
    def __init__(self, lr: float = 1e-4):
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
