from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, Union

import numpy as np

_grad_enabled = True


def is_grad_enabled() -> bool:
    return _grad_enabled


def set_grad_enabled(mode: bool) -> None:
    global _grad_enabled
    _grad_enabled = bool(mode)


@contextmanager
def no_grad() -> Iterator[None]:
    prev = _grad_enabled
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(prev)


@contextmanager
def enable_grad() -> Iterator[None]:
    prev = _grad_enabled
    set_grad_enabled(True)
    try:
        yield
    finally:
        set_grad_enabled(prev)


Scalar = Union[int, float, np.number]
ArrayLike = Union[Scalar, Sequence, np.ndarray, "Tensor"]


class Tensor:
    """N-dimensional array with reverse-mode automatic differentiation."""

    __array_priority__ = 1000

    def __init__(self, data: ArrayLike, requires_grad: bool = False, dtype=None):
        if isinstance(data, Tensor):
            data = data.data
        self.data: np.ndarray = np.asarray(data, dtype=dtype)
        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional[np.ndarray] = None
        self._ctx = None

    # ----------------------------------------------------------------- basics
    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    @property
    def is_leaf(self) -> bool:
        return self._ctx is None

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        grad_flag = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({np.array2string(self.data, precision=4, suppress_small=True)}{grad_flag})"

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return self.data.item()

    def detach(self) -> "Tensor":
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self) -> None:
        self.grad = None

    def copy(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    # --------------------------------------------------------------- autograd
    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require grad.")
        if grad is None:
            grad = np.ones_like(self.data)
        grad = np.asarray(grad.data if isinstance(grad, Tensor) else grad, dtype=self.data.dtype)
        if grad.shape != self.shape:
            grad = np.broadcast_to(grad, self.shape).copy()

        order = self._topological_order()
        self.grad = grad if self.grad is None else self.grad + grad
        for node in reversed(order):
            ctx = node._ctx
            grads = ctx.backward(node.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for parent, g in zip(ctx.inputs, grads):
                if g is None or not parent.requires_grad:
                    continue
                g = np.asarray(g)
                if g.shape != parent.shape:
                    g = np.reshape(g, parent.shape)
                parent.grad = g if parent.grad is None else parent.grad + g

    def _topological_order(self) -> list:
        order, visited, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited or node._ctx is None:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._ctx.inputs:
                stack.append((parent, False))
        return order

    # ------------------------------------------------------------- arithmetic
    @staticmethod
    def _wrap(other: ArrayLike) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        return _ops.add(self, self._wrap(other))

    def __radd__(self, other):
        return _ops.add(self._wrap(other), self)

    def __sub__(self, other):
        return _ops.sub(self, self._wrap(other))

    def __rsub__(self, other):
        return _ops.sub(self._wrap(other), self)

    def __mul__(self, other):
        return _ops.mul(self, self._wrap(other))

    def __rmul__(self, other):
        return _ops.mul(self._wrap(other), self)

    def __truediv__(self, other):
        return _ops.div(self, self._wrap(other))

    def __rtruediv__(self, other):
        return _ops.div(self._wrap(other), self)

    def __pow__(self, exponent):
        return _ops.pow(self, exponent)

    def __neg__(self):
        return _ops.neg(self)

    def __matmul__(self, other):
        return _ops.matmul(self, self._wrap(other))

    def __rmatmul__(self, other):
        return _ops.matmul(self._wrap(other), self)

    # comparisons produce plain boolean arrays; they are not differentiable
    def __eq__(self, other):
        return self.data == self._wrap(other).data

    def __ne__(self, other):
        return self.data != self._wrap(other).data

    def __lt__(self, other):
        return self.data < self._wrap(other).data

    def __le__(self, other):
        return self.data <= self._wrap(other).data

    def __gt__(self, other):
        return self.data > self._wrap(other).data

    def __ge__(self, other):
        return self.data >= self._wrap(other).data

    __hash__ = object.__hash__

    # ---------------------------------------------------------------- indexing
    def __getitem__(self, key):
        return _ops.getitem(self, key)

    def __setitem__(self, key, value) -> None:
        if self._ctx is not None:
            raise RuntimeError("In-place assignment on a non-leaf tensor is not supported.")
        self.data[key] = value.data if isinstance(value, Tensor) else value

    # ------------------------------------------------------------------ math
    def matmul(self, other):
        return _ops.matmul(self, self._wrap(other))

    def pow(self, exponent):
        return _ops.pow(self, exponent)

    def exp(self):
        return _ops.exp(self)

    def log(self):
        return _ops.log(self)

    def sqrt(self):
        return _ops.sqrt(self)

    def abs(self):
        return _ops.abs(self)

    def sin(self):
        return _ops.sin(self)

    def cos(self):
        return _ops.cos(self)

    def tan(self):
        return _ops.tan(self)

    def clip(self, min_val, max_val):
        return _ops.clip(self, min_val, max_val)

    def masked_fill(self, mask, value):
        return _ops.masked_fill(self, mask, value)

    # ------------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims=False):
        return _ops.sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return _ops.mean(self, axis, keepdims)

    def var(self, axis=None, keepdims=False, ddof=0):
        return _ops.var(self, axis, keepdims, ddof)

    def std(self, axis=None, keepdims=False, ddof=0):
        return _ops.std(self, axis, keepdims, ddof)

    def max(self, axis=None, keepdims=False):
        return _ops.max(self, axis, keepdims)

    def min(self, axis=None, keepdims=False):
        return _ops.min(self, axis, keepdims)

    # ------------------------------------------------------------------ shape
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _ops.reshape(self, shape)

    view = reshape

    def flatten(self, start_dim: int = 0):
        lead = self.shape[:start_dim]
        return _ops.reshape(self, lead + (-1,))

    def transpose(self, *axes):
        """No args reverses all axes; two ints swap those axes; a full tuple permutes."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if len(axes) == 0:
            axes = tuple(reversed(range(self.ndim)))
        elif len(axes) == 2:
            a, b = axes
            perm = list(range(self.ndim))
            perm[a], perm[b] = perm[b], perm[a]
            axes = tuple(perm)
        return _ops.permute(self, axes)

    def permute(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return _ops.permute(self, axes)

    def squeeze(self, axis=None):
        return _ops.squeeze(self, axis)

    def unsqueeze(self, axis: int):
        return _ops.unsqueeze(self, axis)

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _ops.broadcast_to(self, shape)

    # ------------------------------------------------------------ activations
    def relu(self):
        return _ops.relu(self)

    def sigmoid(self):
        return _ops.sigmoid(self)

    def tanh(self):
        return _ops.tanh(self)

    def gelu(self):
        return _ops.gelu(self)

    def softmax(self, axis: int = -1):
        return _ops.softmax(self, axis)

    def log_softmax(self, axis: int = -1):
        return _ops.log_softmax(self, axis)


class Parameter(Tensor):
    """A tensor that is registered as a trainable parameter of a Module."""

    def __init__(self, data: ArrayLike, requires_grad: bool = True, dtype=None):
        super().__init__(data, requires_grad=requires_grad, dtype=dtype)

    def __repr__(self) -> str:
        return "Parameter containing:\n" + super().__repr__()


from network_nn.autograd import functional as _ops  # noqa: E402  (circular import by design)
