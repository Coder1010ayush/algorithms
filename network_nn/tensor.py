# -------------------------------------- utf-8 encoding -------------------------------
import os
import numpy as np
import sys
from typing import Dict, List, Tuple, Union
from contextlib import contextmanager
from typing import Generator
import autograd.autodiff as diff
from utils import check_tensor_type


def device(array, device_id=0):
    try:
        import cupy as cp

        try:
            n_devices = cp.cuda.runtime.getDeviceCount()
            if device_id >= n_devices:
                raise RuntimeError(
                    f"Device {device_id} not found. Available devices: {n_devices}"
                )
            with cp.cuda.Device(device_id):
                return cp.asarray(array)
        except cp.cuda.runtime.CUDARuntimeError:
            return array
    except ImportError:
        return array


def as_numpy_array(x):
    from tensor import Tensor

    if isinstance(x, Tensor):
        x = x.data
    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x


def as_cupy_array(x):
    from tensor import Tensor

    if isinstance(x, Tensor):
        x = x.data
    from torch import is_grad_enabled

    if not is_grad_enabled():
        raise Exception("CuPy cannot be loaded. Install CuPy!")
    else:
        try:
            import cupy as cp

            return cp.asarray(x)
        except ImportError:
            return x


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
        meta: dict = None,
    ):
        if data is not None:
            if not check_tensor_type(data):
                raise TypeError(
                    f"{data} can only be int , float , np.array or cupy.ndarray!"
                )
        self.data = np.asarray(data, dtype=dtype)
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

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, key):
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

        return diff.slice_tensor(self, key)

    def __setitem__(self, key, value):
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
        diff.set_item(self, key, value)

    def clip_grad(self, min_val=-1e10, max_val=1e10):
        if self.grad is not None:
            np.clip(self.grad, min_val, max_val, out=self.grad)

    # backpropogation function
    def backpropogate(self):
        if not self.retain_grad:
            raise ValueError("Gradient tracking is not enabled for this tensor.")
        self.grad = np.ones(shape=self.data.shape, dtype=self.data.dtype)
        nodes_to_process = [self]

        while nodes_to_process:
            current_node = nodes_to_process.pop()
            if current_node.creator:
                if current_node.operation:
                    operation_class = getattr(
                        diff, current_node.operation.split("<")[1].strip(">")
                    )
                    operation_instance = operation_class()

                    operation_instance.backward(current_node)
                    current_node.clip_grad()

                for input_node in current_node.creator:
                    if input_node.retain_grad:
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

    def sum(self, axis: int = None):
        return diff.sum(self, axis)

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

    def transpose(self):
        return diff.transpose(self)

    def permute(self, axis: tuple = None):
        return diff.permute(self, axis)

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
