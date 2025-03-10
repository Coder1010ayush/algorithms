# -------------------------------------- utf-8 encoding -----------------------------------------
# this file contains kinds of layers test cases that are implemented here
from tensor import Tensor
from initializer_w import Initializer
from layers.models import Linear
from optimizer.optim import GradientOptimiser
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
    x = init_w.forward(shape=(50, 5), init_type="random", retain_grad=True, meta=None)
    y = init_w.forward(shape=(50, 1), init_type="random", retain_grad=True, meta=None)

    model = LinearModel()
    optimiser = GradientOptimiser(param=model.parameters())
    for epoch in range(20):
        out = model(x)
        loss = mse(predictions=out, targets=y)
        loss.backprop()
        print(f"Loss at Epoch {epoch} is : {loss}")
        optimiser.step()

    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}")
    #     # print("Value:\n", param["value"])
    #     # print("Gradient:\n", param["grad"])
    #     # print("-" * 50)
    #     print(param)
