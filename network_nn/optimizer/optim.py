# ------------------------------ *utf-8 encoding* ----------------------------
import os
import json
import numpy as np
import math
import random


class BaseOptimiser:
    def __init__(self, param, lr: float = 1e-4):
        self.param = param
        self.lr = lr

    def zero_grad(self):
        raise NotImplementedError("zero grad is not implemented for this optimiser.")

    def step(self):
        raise NotImplementedError("Step method is not implemented for optimiser.")


class AdamOptimizer(BaseOptimiser):

    def __init__(self, parameters, lr=0.0001):
        super().__init__(parameters, lr)

    def zero_grad(self):
        return super().zero_grad()

    def step(self):
        return super().step()


class GradientOptimiser(BaseOptimiser):

    def __init__(self, param, lr=0.0001):
        super().__init__(param, lr)

    def zero_grad(self):

        for param in self.param:
            if param["grad"] is not None:
                param["grad"] = np.ones(param["grad"])

    def step(self):
        for param in self.param:
            if param["grad"] is not None:
                param["grad"] -= self.lr * param["grad"]
