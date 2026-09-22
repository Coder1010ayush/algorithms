import numpy as np

from network_nn.autograd.function import Function


def _reduce(value, reduction):
    if reduction == "mean":
        return np.mean(value)
    if reduction == "sum":
        return np.sum(value)
    if reduction == "none":
        return value
    raise ValueError(f"Unknown reduction '{reduction}'")


def _reduce_grad(grad, shape, reduction):
    if reduction == "mean":
        return np.broadcast_to(grad, shape) / np.prod(shape)
    if reduction == "sum":
        return np.broadcast_to(grad, shape)
    return grad


class MSELoss(Function):
    def forward(self, pred, target, reduction="mean"):
        self.diff, self.reduction = pred - target, reduction
        return _reduce(self.diff**2, reduction)

    def backward(self, grad):
        g = _reduce_grad(grad, self.diff.shape, self.reduction) * 2.0 * self.diff
        return g, -g


class BCELoss(Function):
    """Binary cross entropy on probabilities in (0, 1)."""

    def forward(self, prob, target, reduction="mean", eps=1e-12):
        self.p = np.clip(prob, eps, 1.0 - eps)
        self.t, self.reduction = target, reduction
        loss = -(target * np.log(self.p) + (1.0 - target) * np.log(1.0 - self.p))
        return _reduce(loss, reduction)

    def backward(self, grad):
        g = _reduce_grad(grad, self.p.shape, self.reduction)
        return g * (self.p - self.t) / (self.p * (1.0 - self.p)), None


class BCEWithLogitsLoss(Function):
    def forward(self, logits, target, reduction="mean"):
        self.sig = 1.0 / (1.0 + np.exp(-logits))
        self.t, self.reduction = target, reduction
        loss = np.maximum(logits, 0) - logits * target + np.log1p(np.exp(-np.abs(logits)))
        return _reduce(loss, reduction)

    def backward(self, grad):
        g = _reduce_grad(grad, self.sig.shape, self.reduction)
        return g * (self.sig - self.t), None


class CrossEntropyLoss(Function):
    """Softmax cross entropy from logits of shape (N, C) or (N, C, ...) with integer class targets."""

    def forward(self, logits, target, reduction="mean"):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        self.probs = np.exp(log_probs)
        self.target = np.asarray(target, dtype=np.int64)
        self.reduction = reduction
        picked = np.take_along_axis(log_probs, self.target[:, None, ...], axis=1)[:, 0, ...]
        return _reduce(-picked, reduction)

    def backward(self, grad):
        onehot = np.zeros_like(self.probs)
        np.put_along_axis(onehot, self.target[:, None, ...], 1.0, axis=1)
        g = _reduce_grad(grad, self.target.shape, self.reduction)
        return np.expand_dims(g, 1) * (self.probs - onehot), None


class HuberLoss(Function):
    def forward(self, pred, target, delta=1.0, reduction="mean"):
        self.diff, self.delta, self.reduction = pred - target, delta, reduction
        a = np.abs(self.diff)
        loss = np.where(a <= delta, 0.5 * self.diff**2, delta * (a - 0.5 * delta))
        return _reduce(loss, reduction)

    def backward(self, grad):
        g = _reduce_grad(grad, self.diff.shape, self.reduction)
        local = np.where(np.abs(self.diff) <= self.delta, self.diff, self.delta * np.sign(self.diff))
        return g * local, -g * local
