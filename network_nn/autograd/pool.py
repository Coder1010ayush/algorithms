from itertools import product

import numpy as np

from network_nn.autograd.conv import _ntuple, _windows
from network_nn.autograd.function import Function


class MaxPoolNd(Function):
    def forward(self, x, kernel_size, stride=None):
        nd = x.ndim - 2
        self.kernel = _ntuple(kernel_size, nd)
        self.stride = self.kernel if stride is None else _ntuple(stride, nd)
        self.shape = x.shape
        win = _windows(x, self.kernel, self.stride)
        lead = win.shape[: 2 + nd]
        flat = win.reshape(lead + (-1,))
        self.argmax = np.argmax(flat, axis=-1)
        return np.take_along_axis(flat, self.argmax[..., None], axis=-1)[..., 0]

    def backward(self, grad):
        nd = len(self.kernel)
        out_shape = grad.shape[2:]
        offsets = np.unravel_index(self.argmax, self.kernel)
        grids = np.meshgrid(*[np.arange(n) for n in grad.shape], indexing="ij")
        coords = [grids[0], grids[1]] + [
            grids[2 + d] * self.stride[d] + offsets[d] for d in range(nd)
        ]
        out = np.zeros(self.shape, dtype=grad.dtype)
        np.add.at(out, tuple(coords), grad)
        return out


class AvgPoolNd(Function):
    def forward(self, x, kernel_size, stride=None):
        nd = x.ndim - 2
        self.kernel = _ntuple(kernel_size, nd)
        self.stride = self.kernel if stride is None else _ntuple(stride, nd)
        self.shape = x.shape
        win = _windows(x, self.kernel, self.stride)
        return win.mean(axis=tuple(range(2 + nd, 2 + 2 * nd)))

    def backward(self, grad):
        out = np.zeros(self.shape, dtype=grad.dtype)
        share = grad / np.prod(self.kernel)
        out_shape = grad.shape[2:]
        for offset in product(*[range(k) for k in self.kernel]):
            region = tuple(slice(o, o + n * s, s) for o, n, s in zip(offset, out_shape, self.stride))
            out[(slice(None), slice(None)) + region] += share
        return out
