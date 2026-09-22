import numpy as np

from network_nn.autograd import functional as F
from network_nn.nn.module import Module
from network_nn.tensor import Parameter, Tensor, no_grad


class LayerNorm(Module):
    """Normalises over the last dimension, then applies a learnable affine transform."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape, self.eps = normalized_shape, eps
        self.weight = Parameter(np.ones(normalized_shape)) if elementwise_affine else None
        self.bias = Parameter(np.zeros(normalized_shape)) if elementwise_affine else None

    def forward(self, x: Tensor) -> Tensor:
        out = F.layer_norm(x, self.eps)
        if self.weight is not None:
            out = out * self.weight + self.bias
        return out

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


class _BatchNorm(Module):
    """Normalises each channel (axis 1) over batch and spatial axes; tracks running stats for eval."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.weight = Parameter(np.ones(num_features)) if affine else None
        self.bias = Parameter(np.zeros(num_features)) if affine else None
        self.register_buffer("running_mean", Tensor(np.zeros(num_features)))
        self.register_buffer("running_var", Tensor(np.ones(num_features)))

    def _channel_shape(self, ndim: int):
        return (1, self.num_features) + (1,) * (ndim - 2)

    def forward(self, x: Tensor) -> Tensor:
        axes = tuple(i for i in range(x.ndim) if i != 1)
        if self.training:
            out = F.batch_norm(x, self.eps)
            with no_grad():
                n = x.size / self.num_features
                mean = x.data.mean(axis=axes)
                var = x.data.var(axis=axes) * n / max(n - 1, 1)
                self.running_mean.data[...] = (1 - self.momentum) * self.running_mean.data + self.momentum * mean
                self.running_var.data[...] = (1 - self.momentum) * self.running_var.data + self.momentum * var
        else:
            shape = self._channel_shape(x.ndim)
            mean = Tensor(self.running_mean.data.reshape(shape))
            std = Tensor(np.sqrt(self.running_var.data.reshape(shape) + self.eps))
            out = (x - mean) / std
        if self.weight is not None:
            shape = self._channel_shape(x.ndim)
            out = out * self.weight.reshape(shape) + self.bias.reshape(shape)
        return out

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"


class BatchNorm1d(_BatchNorm):
    """Input (N, C) or (N, C, L)."""


class BatchNorm2d(_BatchNorm):
    """Input (N, C, H, W)."""


class BatchNorm3d(_BatchNorm):
    """Input (N, C, D, H, W)."""
