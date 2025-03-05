# -------------------------------------- utf-8 encoding -------------------------------
import os
import numpy as np
import sys
from typing import Dict, List, Tuple, Union
from contextlib import contextmanager
from typing import Generator

from zen import diff


# for switcging grad computation mode


class NoGradientContext:
    def __init__(self):
        self.prev_state = None

    def __enter__(self):
        self.prev_state = is_grad_enabled()
        set_grad_enabled(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_grad_enabled(self.prev_state)


# by defalt gradient will calculated
_grad_enabled = True


def is_grad_enabled() -> bool:
    return _grad_enabled


def set_grad_enabled(mode: bool) -> None:
    global _grad_enabled
    _grad_enabled = mode


# testing mode


@contextmanager
def no_grad() -> Generator:
    prev_state = is_grad_enabled()
    try:
        set_grad_enabled(False)
        yield
    finally:
        set_grad_enabled(prev_state)


# training mode


@contextmanager
def use_grad() -> Generator:
    prev_state = is_grad_enabled()
    try:
        set_grad_enabled(True)
        yield
    finally:
        set_grad_enabled(prev_state)


class Tensor:

    def __init__(
        self,
        data,
        dtype: Union[int, float, np.dtype] = None,
        retain_grad=True,
        operation: str = None,
        creator: list = [],
    ):
        if data is not None:
            from zen.utils import check_tensor_type

            if not check_tensor_type(data):
                return TypeError(
                    f"{data} can only be int , float , np.array or cupy.ndarray!"
                )
        self.data = data
        self.retain_grad = retain_grad
        set_grad_enabled(mode=self.retain_grad)
        self.grad = None
        self.operation = operation
        self.creator = creator

    def __repr__(self):
        return f"netwok_nn.tensor({self.data} , grad = {self.grad})"

    @property
    def shape(self):
        return f"Size({self.data.shape})"

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self):
        return diff.transpose(self)

    def transpose(self):
        return diff.transpose(self)

    def __getitem__(self, key):
        # index based slicing on tensor object
        if not isinstance(key, tuple):
            key = (key,)

        if Ellipsis in key:
            ellipsis_idx = key.index(Ellipsis)
            n_missing = self.ndim - (len(key) - 1)
            key = (
                key[:ellipsis_idx]
                + (slice(None),) * n_missing
                + key[ellipsis_idx + 1 :]
            )

        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))

        sliced_data = self.data[key]
        return Tensor(sliced_data, dtype=self.data.dtype, retain_grad=self.retain_grad)

    def __setitem__(self, key, value):
        # index based operation (set or unset values)
        if isinstance(value, Tensor):
            value = value.data

        elif not isinstance(value, np.ndarray):
            value = np.array(value, dtype=self.data.dtype)

        if not isinstance(key, tuple):
            key = (key,)

        if Ellipsis in key:
            ellipsis_idx = key.index(Ellipsis)
            n_missing = self.ndim - (len(key) - 1)
            key = (
                key[:ellipsis_idx]
                + (slice(None),) * n_missing
                + key[ellipsis_idx + 1 :]
            )

        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))
        self.data[key] = value

    def clip_grad(self, min_val=-1e10, max_val=1e10):
        if self.grad is not None:
            np.clip(self.grad, min_val, max_val, out=self.grad)

    # backpropogation function
    def backpropogate(self):
        if not self.requires_grad:
            raise ValueError("Gradient tracking is not enabled for this tensor.")
        self.grad = np.ones(shape=self.data.shape, dtype=self.data.dtype)
        nodes_to_process = [self]

        while nodes_to_process:
            current_node = nodes_to_process.pop()
            if current_node.inputs_node:
                if current_node.operation:
                    operation_class = getattr(
                        diff, current_node.operation.split("<")[1].strip(">")
                    )
                    operation_instance = operation_class()

                    operation_instance.backward(current_node)
                    current_node.clip_grad()

                for input_node in current_node.inputs_node:
                    if input_node.requires_grad:
                        nodes_to_process.append(input_node)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< arithmetic operations >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # addition , subtraction , multiplication , division , power , mean , std , exponential , trigo functions , log , custom_function template

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(data=other, retain_grad=self.retain_grad)
        return diff.add(self, other)

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(data=other, retain_grad=self.retain_grad)
        return diff.sub(self, other)

    def __rsub__(self, other):
        return other - self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(data=other, retain_grad=self.retain_grad)
        return diff.mul(self, other)

    def __rmul__(self, other):
        return other * self

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(data=other, retain_grad=self.retain_grad)
        return diff.div(self, other)

    def __rtruediv__(self, other):
        return other / self

    def matmul(self, other):
        return diff.matmul(self, other)

    def mean(self, axis: int = -1):
        return diff.mean(self, axis)

    def std(self, axis: int = -1):
        return diff.std(self, axis)

    def pow(self, other: int = 1):
        return diff.pow(self, other)

    def log(self, base: int = 10, is_natural=False):
        if is_natural:
            return diff.nlog(self)
        return diff.log(self, base)

    def exp(self, base: int = -1, is_default=True):
        if is_default:
            return diff.exp(self, base)
        else:
            return diff.exp(self, base)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< some advance mathematical functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # sigmoid , relu  , gelu , softmax etc

    def sigmoid(self):
        return diff.sigmoid(self)

    def relu(self):
        return diff.relu(self)

    def gelu(self):
        return diff.gelu(self)

    def softmax(self, dim: int = -1):
        return diff.softmax(self, dim)

    def sin(self):
        return diff.sin(self)

    def cos(self):
        return diff.cos(self)

    def tan(self):
        return diff.tan(self)

    def cosec(self):
        return diff.cosec(self)

    def sec(self):
        return diff.sec(self)

    def sqrt(self):
        return diff.sqrt(self)

    def flatten(self):
        return diff.flatten(self)
