import numpy as np

from network_nn.autograd.function import Function, unbroadcast


class MatMul(Function):
    def forward(self, a, b):
        self.a, self.b = a, b
        return np.matmul(a, b)

    def backward(self, grad):
        a, b = self.a, self.b
        if a.ndim == 1 and b.ndim == 1:
            return grad * b, grad * a
        if b.ndim == 1:
            ga = np.expand_dims(grad, -1) * b
            gb = np.tensordot(grad, a, axes=(range(grad.ndim), range(a.ndim - 1)))
            return unbroadcast(ga, a.shape), gb
        if a.ndim == 1:
            ga = np.matmul(grad[..., None, :], np.swapaxes(b, -1, -2))[..., 0, :]
            ga = ga.reshape(-1, a.shape[0]).sum(axis=0)
            gb = np.matmul(a[:, None], grad[..., None, :])
            return ga, unbroadcast(gb, b.shape)
        ga = np.matmul(grad, np.swapaxes(b, -1, -2))
        gb = np.matmul(np.swapaxes(a, -1, -2), grad)
        return unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)
