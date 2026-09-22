from network_nn.autograd import functional as F
from network_nn.nn.module import Module
from network_nn.tensor import Tensor


class _Loss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class MSELoss(_Loss):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(pred, target, self.reduction)


class BCELoss(_Loss):
    def forward(self, prob: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(prob, target, self.reduction)


class BCEWithLogitsLoss(_Loss):
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(logits, target, self.reduction)


class CrossEntropyLoss(_Loss):
    def forward(self, logits: Tensor, target) -> Tensor:
        return F.cross_entropy(logits, target, self.reduction)


class HuberLoss(_Loss):
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__(reduction)
        self.delta = delta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(pred, target, self.delta, self.reduction)
