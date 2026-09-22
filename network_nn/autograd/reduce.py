import numpy as np

from network_nn.autograd.function import Function, expand_reduced, normalize_axes


class Sum(Function):
    def forward(self, a, axis=None, keepdims=False):
        self.shape, self.keepdims = a.shape, keepdims
        self.axes = normalize_axes(axis, a.ndim)
        return np.sum(a, axis=self.axes, keepdims=keepdims)

    def backward(self, grad):
        return expand_reduced(grad, self.shape, self.axes, self.keepdims)


class Mean(Function):
    def forward(self, a, axis=None, keepdims=False):
        self.shape, self.keepdims = a.shape, keepdims
        self.axes = normalize_axes(axis, a.ndim)
        self.count = a.size if self.axes is None else int(np.prod([a.shape[i] for i in self.axes]))
        return np.mean(a, axis=self.axes, keepdims=keepdims)

    def backward(self, grad):
        return expand_reduced(grad, self.shape, self.axes, self.keepdims) / self.count


class Var(Function):
    def forward(self, a, axis=None, keepdims=False, ddof=0):
        self.keepdims = keepdims
        self.axes = normalize_axes(axis, a.ndim)
        count = a.size if self.axes is None else int(np.prod([a.shape[i] for i in self.axes]))
        self.scale = 2.0 / (count - ddof)
        self.centered = a - np.mean(a, axis=self.axes, keepdims=True)
        return np.var(a, axis=self.axes, keepdims=keepdims, ddof=ddof)

    def backward(self, grad):
        grad = expand_reduced(grad, self.centered.shape, self.axes, self.keepdims)
        return grad * self.scale * self.centered


class Std(Function):
    def forward(self, a, axis=None, keepdims=False, ddof=0):
        self.keepdims = keepdims
        self.axes = normalize_axes(axis, a.ndim)
        count = a.size if self.axes is None else int(np.prod([a.shape[i] for i in self.axes]))
        self.centered = a - np.mean(a, axis=self.axes, keepdims=True)
        self.std = np.std(a, axis=self.axes, keepdims=True, ddof=ddof)
        self.denom = count - ddof
        return np.std(a, axis=self.axes, keepdims=keepdims, ddof=ddof)

    def backward(self, grad):
        grad = expand_reduced(grad, self.centered.shape, self.axes, self.keepdims)
        return grad * self.centered / (self.denom * self.std)


class _MinMax(Function):
    reducer = None

    def forward(self, a, axis=None, keepdims=False):
        self.shape, self.keepdims = a.shape, keepdims
        self.axes = normalize_axes(axis, a.ndim)
        extreme = self.reducer(a, axis=self.axes, keepdims=True)
        hits = a == extreme
        self.mask = hits / np.sum(hits, axis=self.axes, keepdims=True)
        return self.reducer(a, axis=self.axes, keepdims=keepdims)

    def backward(self, grad):
        return expand_reduced(grad, self.shape, self.axes, self.keepdims) * self.mask


class Max(_MinMax):
    reducer = staticmethod(np.max)


class Min(_MinMax):
    reducer = staticmethod(np.min)
