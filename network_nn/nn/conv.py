import numpy as np

from network_nn.autograd import functional as F
from network_nn.autograd.conv import _ntuple
from network_nn.nn import init
from network_nn.nn.module import Module
from network_nn.tensor import Parameter, Tensor


class _ConvNd(Module):
    ndim: int = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = _ntuple(kernel_size, self.ndim)
        self.stride = _ntuple(stride, self.ndim)
        self.padding = padding
        self.weight = Parameter(np.empty((out_channels, in_channels) + self.kernel_size))
        self.bias = Parameter(np.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * int(np.prod(self.kernel_size))
            bound = 1.0 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        out = F.conv1d(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            out = out + self.bias.reshape((1, -1) + (1,) * self.ndim)
        return out

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding!r}, bias={self.bias is not None}"
        )


class Conv1d(_ConvNd):
    ndim = 1


class Conv2d(_ConvNd):
    ndim = 2


class Conv3d(_ConvNd):
    ndim = 3


class _PoolNd(Module):
    ndim: int = 1
    op = None

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = _ntuple(kernel_size, self.ndim)
        self.stride = self.kernel_size if stride is None else _ntuple(stride, self.ndim)

    def forward(self, x: Tensor) -> Tensor:
        return type(self).op(x, self.kernel_size, self.stride)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}"


class MaxPool1d(_PoolNd):
    ndim, op = 1, staticmethod(F.max_pool1d)


class MaxPool2d(_PoolNd):
    ndim, op = 2, staticmethod(F.max_pool2d)


class MaxPool3d(_PoolNd):
    ndim, op = 3, staticmethod(F.max_pool3d)


class AvgPool1d(_PoolNd):
    ndim, op = 1, staticmethod(F.avg_pool1d)


class AvgPool2d(_PoolNd):
    ndim, op = 2, staticmethod(F.avg_pool2d)


class AvgPool3d(_PoolNd):
    ndim, op = 3, staticmethod(F.avg_pool3d)


class GlobalAvgPool(Module):
    """(N, C, *spatial) -> (N, C)"""

    def forward(self, x: Tensor) -> Tensor:
        return F.global_avg_pool(x)
