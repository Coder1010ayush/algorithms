from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from network_nn.tensor import Tensor, is_grad_enabled


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` over the axes that NumPy broadcasting introduced so it matches `shape`."""
    grad = np.asarray(grad)
    if grad.shape == shape:
        return grad
    lead = grad.ndim - len(shape)
    if lead > 0:
        grad = grad.sum(axis=tuple(range(lead)))
    axes = tuple(i for i, (s, g) in enumerate(zip(shape, grad.shape)) if s == 1 and g != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


def normalize_axes(axis: Union[None, int, Sequence[int]], ndim: int) -> Optional[Tuple[int, ...]]:
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(a % ndim for a in axis)


def expand_reduced(grad: np.ndarray, shape: tuple, axes: Optional[Tuple[int, ...]], keepdims: bool) -> np.ndarray:
    """Broadcast a reduced gradient back to the shape of the input it was reduced from."""
    if axes is None:
        return np.broadcast_to(grad, shape)
    if not keepdims:
        grad = np.expand_dims(grad, axes)
    return np.broadcast_to(grad, shape)


class Function:
    """Base class for differentiable operations.

    Subclasses implement `forward(*arrays, **kwargs) -> ndarray` and
    `backward(grad) -> ndarray | tuple[ndarray | None, ...]` with one gradient per
    Tensor argument, in order. Call an op through `Op.apply(...)`.
    """

    def __init__(self):
        self.inputs: Tuple[Tensor, ...] = ()

    def forward(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Tensor:
        fn = cls()
        tensors = tuple(a for a in args if isinstance(a, Tensor))
        raw = [a.data if isinstance(a, Tensor) else a for a in args]
        out = Tensor(fn.forward(*raw, **kwargs))
        if is_grad_enabled() and any(t.requires_grad for t in tensors):
            out.requires_grad = True
            fn.inputs = tensors
            out._ctx = fn
        return out

    def __repr__(self) -> str:
        return f"<{type(self).__name__}Backward>"
