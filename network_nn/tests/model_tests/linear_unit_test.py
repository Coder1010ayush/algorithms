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
    def forward(self, x: Tensor):

        out = self.linear(x)
        return out


