# -------------------------------------- utf-8 encoding -----------------------------------------
# this file contains kinds of layers test cases that are implemented here
import numpy as np
from tensor import Tensor
from initializer_w import Initializer
from layers.models import Linear
from optimizer.optim import GradientOptimiser, AdamOptimizer
from layers.module import Module
from autograd.autodiff import mse


class LinearModel(Module):

    def __init__(
        self,
        in_features: int = 5,
        out_features: int = 1,
        init_type="uniform",
        meta: dict = {"low": 0.0, "high": 1.0},
    ):
        super(LinearModel, self).__init__()
        self.in_feature = in_features
        self.out_feature = out_features
        self.init_type = init_type
        self.meta = meta
        self.linear = Linear(
            in_feature=self.in_feature,
            out_feature=self.out_feature,
            init_type=self.init_type,
            meta=self.meta,
        )
        self.add_module("linear", self.linear)

    def forward(self, x: Tensor):

        out = self.linear(x)
        return out


if __name__ == "__main__":
    # defining input data for this
    init_w = Initializer()
    x = np.linspace(-10, 10, 50).reshape(-1, 1)
    # noise = np.random.normal(0, 2, size=x.shape)
    y = (3 * x + 2).reshape(-1, 1)
    x = Tensor(data=x, retain_grad=True)
    y = Tensor(data=y, retain_grad=True)

    model = LinearModel(in_features=1)
    optimiser = AdamOptimizer(lr=0.001)
    for epoch in range(2000):
        out = model(x)
        loss = mse(predictions=out, targets=y)
        loss.backprop()
        print(f"Loss at Epoch {epoch} is : {loss}")
        optimiser.step(params=model.parameters())
