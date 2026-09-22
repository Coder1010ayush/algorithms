import numpy as np

from network_nn.autograd.function import Function, unbroadcast


class Reshape(Function):
    def forward(self, a, shape):
        self.shape = a.shape
        return np.reshape(a, shape)

    def backward(self, grad):
        return np.reshape(grad, self.shape)


class Permute(Function):
    def forward(self, a, axes):
        self.axes = tuple(axes)
        return np.transpose(a, self.axes)

    def backward(self, grad):
        return np.transpose(grad, np.argsort(self.axes))


class Squeeze(Function):
    def forward(self, a, axis=None):
        self.shape = a.shape
        return np.squeeze(a, axis=axis)

    def backward(self, grad):
        return np.reshape(grad, self.shape)


class Unsqueeze(Function):
    def forward(self, a, axis):
        self.axis = axis
        return np.expand_dims(a, axis)

    def backward(self, grad):
        return np.squeeze(grad, axis=self.axis)


class BroadcastTo(Function):
    def forward(self, a, shape):
        self.shape = a.shape
        return np.broadcast_to(a, shape).copy()

    def backward(self, grad):
        return unbroadcast(grad, self.shape)


class GetItem(Function):
    def forward(self, a, key):
        self.shape, self.key = a.shape, key
        return a[key]

    def backward(self, grad):
        out = np.zeros(self.shape, dtype=grad.dtype)
        np.add.at(out, self.key, grad)
        return out


class Stack(Function):
    def forward(self, *arrays, axis=0):
        self.axis, self.n = axis, len(arrays)
        return np.stack(arrays, axis=axis)

    def backward(self, grad):
        return tuple(np.take(grad, i, axis=self.axis) for i in range(self.n))


class Concat(Function):
    def forward(self, *arrays, axis=0):
        self.axis = axis
        self.splits = np.cumsum([a.shape[axis] for a in arrays])[:-1]
        return np.concatenate(arrays, axis=axis)

    def backward(self, grad):
        return tuple(np.split(grad, self.splits, axis=self.axis))


class Pad(Function):
    def forward(self, a, pad_width, mode="constant", constant_values=0):
        self.pad_width = np.broadcast_to(np.asarray(pad_width, dtype=int), (a.ndim, 2))
        return np.pad(a, self.pad_width, mode=mode, constant_values=constant_values)

    def backward(self, grad):
        slices = tuple(slice(int(lo), grad.shape[i] - int(hi)) for i, (lo, hi) in enumerate(self.pad_width))
        return grad[slices]
