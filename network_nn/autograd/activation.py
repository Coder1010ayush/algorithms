import numpy as np

from network_nn.autograd.function import Function

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


class ReLU(Function):
    def forward(self, a):
        self.mask = a > 0
        return np.maximum(a, 0)

    def backward(self, grad):
        return grad * self.mask


class LeakyReLU(Function):
    def forward(self, a, negative_slope=0.01):
        self.slope = np.where(a > 0, 1.0, negative_slope)
        return a * self.slope

    def backward(self, grad):
        return grad * self.slope


class Sigmoid(Function):
    def forward(self, a):
        self.out = 1.0 / (1.0 + np.exp(-a))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1.0 - self.out)


class Tanh(Function):
    def forward(self, a):
        self.out = np.tanh(a)
        return self.out

    def backward(self, grad):
        return grad * (1.0 - self.out * self.out)


class Swish(Function):
    def forward(self, a):
        self.a = a
        self.sig = 1.0 / (1.0 + np.exp(-a))
        return a * self.sig

    def backward(self, grad):
        return grad * (self.sig + self.a * self.sig * (1.0 - self.sig))


class GELU(Function):
    """Tanh approximation of GELU, as used in GPT-style transformers."""

    def forward(self, a):
        self.a = a
        self.inner = _SQRT_2_OVER_PI * (a + 0.044715 * a**3)
        self.t = np.tanh(self.inner)
        return 0.5 * a * (1.0 + self.t)

    def backward(self, grad):
        d_inner = _SQRT_2_OVER_PI * (1.0 + 3 * 0.044715 * self.a**2)
        return grad * (0.5 * (1.0 + self.t) + 0.5 * self.a * (1.0 - self.t**2) * d_inner)


class Softmax(Function):
    def forward(self, a, axis=-1):
        self.axis = axis
        shifted = a - np.max(a, axis=axis, keepdims=True)
        e = np.exp(shifted)
        self.out = e / np.sum(e, axis=axis, keepdims=True)
        return self.out

    def backward(self, grad):
        return self.out * (grad - np.sum(grad * self.out, axis=self.axis, keepdims=True))


class LogSoftmax(Function):
    def forward(self, a, axis=-1):
        self.axis = axis
        shifted = a - np.max(a, axis=axis, keepdims=True)
        self.out = shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
        return self.out

    def backward(self, grad):
        return grad - np.exp(self.out) * np.sum(grad, axis=self.axis, keepdims=True)
