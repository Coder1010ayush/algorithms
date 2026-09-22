import numpy as np

from network_nn.autograd.function import Function, normalize_axes


class Normalize(Function):
    """(x - mean) / sqrt(var + eps) over `axes`. LayerNorm and BatchNorm are both built on this."""

    def forward(self, a, axes, eps=1e-5):
        self.axes = normalize_axes(axes, a.ndim)
        mean = np.mean(a, axis=self.axes, keepdims=True)
        var = np.var(a, axis=self.axes, keepdims=True)
        self.inv_std = 1.0 / np.sqrt(var + eps)
        self.xhat = (a - mean) * self.inv_std
        self.mean, self.var = mean, var
        return self.xhat

    def backward(self, grad):
        g_mean = np.mean(grad, axis=self.axes, keepdims=True)
        gx_mean = np.mean(grad * self.xhat, axis=self.axes, keepdims=True)
        return self.inv_std * (grad - g_mean - self.xhat * gx_mean)
