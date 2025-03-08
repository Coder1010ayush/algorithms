# ------------------------- *utf-8 encoding* ----------------------------
from initializer_w import Initializer
from typing import Literal
from layers.module import Module
from tensor import Tensor


class Linear(Module):

    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        init_type: Literal[
            "uniform",
            "normal",
            "constent",
            "ones",
            "zeros",
            "xaviour_uniform",
            "xaviour_normal",
            "kaining_uniform",
            "kaining_normal",
            "trunc_normal",
            "orthogonal",
        ] = "uniform",
    ):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.init_type = init_type
        # initializing weight of the model

    def forward(self, *inputs):
        return super().forward(*inputs)
