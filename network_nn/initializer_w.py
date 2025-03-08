# ---------------------------- *utf-8 encoding* ---------------------------------
# ------------------ utf-8 encoding ------------------------
import numpy as np
import os
import sys
import math
import random
from tensor import Tensor


class Initializer:
    def __init__(self):
        pass

    def initialize(self, shape: tuple, init_type: str, retain_grad: bool, meta: dict):
        self.shape = shape
        self.init_type = init_type
        self.retain_grad = retain_grad
        self.meta = meta

    def forward(self, shape: tuple, init_type: str, retain_grad: bool, meta: dict):
        self.initialize(shape, init_type, retain_grad, meta)
        if self.init_type == "uniform":
            return self.uniform(
                shape=self.shape,
                retain_grad=self.retain_grad,
                low=self.meta["low"],
                high=self.meta["high"],
            )
        elif self.init_type == "normal":
            return self.normal(
                shape=self.shape,
                retain_grad=self.retain_grad,
                loc=self.meta["loc"],
                scale=self.meta["scale"],
            )
        elif self.init_type == "random":
            return self.randn(shape=self.shape, retain_grad=self.retain_grad)
        elif self.init_type == "ones":
            return self.ones(
                shape=self.shape, retain_grad=self.retain_grad, dtype=float
            )
        elif self.init_type == "zeros":
            return self.zeros(
                shape=self.shape, retain_grad=self.retain_grad, dtype=float
            )
        elif self.init_type == "constants":
            return self.constants(
                shape=self.shape,
                dtype=float,
                retain_grad=self.retain_grad,
                val=self.meta["val"],
            )
        elif self.init_type == "xavier_uniform":
            return self.xavier_uniform(
                shape=self.shape,
                n_in=self.meta["n_in"],
                n_out=self.meta["n_out"],
                retain_grad=self.retain_grad,
            )
        elif self.init_type == "xavier_normal":
            return self.xavier_normal(
                shape=self.shape,
                n_in=self.meta["n_in"],
                n_out=self.meta["n_out"],
                retain_grad=self.retain_grad,
            )
        elif self.init_type == "lecun_uniform":
            return self.lecun_uniform(
                shape=self.shape, n_in=self.meta["n_in"], retain_grad=self.retain_grad
            )
        elif self.init_type == "lecun_normal":
            return self.lecun_normal(
                shape=self.shape, n_in=self.meta["n_in"], retain_grad=self.retain_grad
            )
        else:
            raise ValueError(f"Unsupported init type {self.init_type} is given!")

    def randn(self, shape: tuple, dtype=float, retain_grad=False):
        data = np.random.randn(*shape)
        return Tensor(data=data, dtype=dtype, retain_grad=retain_grad)

    def identity(self, n, dtype=float, retain_grad=False):
        data = np.identity(n=n, dtype=dtype)
        return Tensor(data=data, retain_grad=retain_grad, dtype=dtype)

    def ones(self, shape, dtype=float, retain_grad=False):
        data = np.ones(shape=shape, dtype=dtype)
        return Tensor(data=data, retain_grad=retain_grad, dtype=dtype)

    def zeros(self, shape, dtype=float, retain_grad=False):
        data = np.zeros(shape=shape, dtype=dtype)
        return Tensor(data=data, retain_grad=retain_grad, dtype=dtype)

    def constants(self, shape, val, dtype=float, retain_grad=False):
        data = np.full(shape=shape, fill_value=val, dtype=dtype)
        return Tensor(data=data, retain_grad=retain_grad, dtype=dtype)

    def arange(self, n1=0, n2=100, n3=1, dtype=np.float32, retain_grad=False):
        data = np.arange(start=n1, step=n3, stop=n2, dtype=dtype)
        return Tensor(data=data, retain_grad=retain_grad, dtype=dtype)

    def uniform(self, shape, retain_grad=False, low=0, high=1):
        data = np.random.uniform(low=low, high=high, size=shape)
        return Tensor(data=data, retain_grad=retain_grad, dtype=float)

    def normal(self, shape, retain_grad=False, loc=0, scale=0.5):
        data = np.random.normal(loc=loc, scale=scale, size=shape)
        return Tensor(data=data, retain_grad=retain_grad, dtype=float)

    def xavier_uniform(self, shape, n_in, n_out, retain_grad=False):
        limit = np.sqrt(6 / (n_in + n_out))
        data = np.random.uniform(low=-limit, high=limit, size=shape)
        return Tensor(data=data, dtype=data.dtype, retain_grad=retain_grad)

    def xavier_normal(self, shape, n_in, n_out, retain_grad=False):
        std = np.sqrt(2 / (n_in + n_out))
        data = np.random.normal(loc=0, scale=std, size=shape)
        return Tensor(data=data, retain_grad=retain_grad, dtype=data.dtype)

    def lecun_uniform(self, shape, n_in, retain_grad=False):
        limit = np.sqrt(1 / n_in)
        data = np.random.uniform(low=-limit, high=limit, size=shape)
        return Tensor(data=data, retain_grad=retain_grad, dtype=data.dtype)

    def lecun_normal(self, shape, n_in, retain_grad=False):
        std = np.sqrt(1 / n_in)
        data = np.random.normal(loc=0, scale=std, size=shape)
        return Tensor(data=data, retain_grad=retain_grad, dtype=data.dtype)
