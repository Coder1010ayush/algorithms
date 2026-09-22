from itertools import product
from typing import Sequence, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from network_nn.autograd.function import Function

IntOrTuple = Union[int, Sequence[int]]


def _ntuple(value: IntOrTuple, n: int) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * n
    value = tuple(int(v) for v in value)
    if len(value) != n:
        raise ValueError(f"Expected {n} values, got {value}")
    return value


def resolve_padding(padding, kernel: Tuple[int, ...]) -> Tuple[int, ...]:
    if padding == "valid":
        return (0,) * len(kernel)
    if padding == "same":
        return tuple((k - 1) // 2 for k in kernel)
    return _ntuple(padding, len(kernel))


def _windows(x: np.ndarray, kernel: Tuple[int, ...], stride: Tuple[int, ...]) -> np.ndarray:
    """(N, C, *S) -> (N, C, *O, *K) strided view of every receptive field."""
    nd = len(kernel)
    spatial = tuple(range(x.ndim - nd, x.ndim))
    win = sliding_window_view(x, kernel, axis=spatial)
    index = (slice(None),) * (x.ndim - nd) + tuple(slice(None, None, s) for s in stride)
    return win[index]


class ConvNd(Function):
    """N-d cross-correlation of (N, C_in, *S) with weights (C_out, C_in, *K)."""

    def forward(self, x, w, stride=1, padding=0):
        nd = w.ndim - 2
        self.stride = _ntuple(stride, nd)
        self.pad = resolve_padding(padding, w.shape[2:])
        self.w = w
        self.xp = np.pad(x, [(0, 0), (0, 0)] + [(p, p) for p in self.pad])
        self.win = _windows(self.xp, w.shape[2:], self.stride)  # (N, C, *O, *K)
        spatial_axes = list(range(2 + nd, 2 + 2 * nd))
        out = np.tensordot(self.win, w, axes=([1] + spatial_axes, [1] + list(range(2, 2 + nd))))
        return np.moveaxis(out, -1, 1)  # (N, F, *O)

    def backward(self, grad):
        nd = self.w.ndim - 2
        out_shape = grad.shape[2:]
        out_axes = list(range(2, 2 + nd))
        grad_w = np.tensordot(grad, self.win, axes=([0] + out_axes, [0] + out_axes))

        grad_xp = np.zeros_like(self.xp)
        for offset in product(*[range(k) for k in self.w.shape[2:]]):
            w_tap = self.w[(slice(None), slice(None)) + offset]  # (F, C)
            contrib = np.tensordot(grad, w_tap, axes=([1], [0]))  # (N, *O, C)
            contrib = np.moveaxis(contrib, -1, 1)
            region = tuple(slice(o, o + n * s, s) for o, n, s in zip(offset, out_shape, self.stride))
            grad_xp[(slice(None), slice(None)) + region] += contrib

        unpad = tuple(slice(p, grad_xp.shape[i + 2] - p) for i, p in enumerate(self.pad))
        return grad_xp[(slice(None), slice(None)) + unpad], grad_w


def conv_output_shape(spatial: Sequence[int], kernel: Sequence[int], stride: Sequence[int], pad: Sequence[int]):
    return tuple((s + 2 * p - k) // st + 1 for s, k, st, p in zip(spatial, kernel, stride, pad))
