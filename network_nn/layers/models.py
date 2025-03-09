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
        meta: dict = {"low": 0.0, "high": 1.0},
    ):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.meta = meta

        self.init_type = init_type
        # initializing weight of the model
        self.initializer = Initializer()
        self.shape = (in_feature, out_feature)
        self.__init_w(shape=self.shape, meta=self.meta)

    def __init_w(self, shape, meta):
        self.weight = self.initializer.forward(
            shape=shape,
            init_type=self.init_type,
            retain_grad=True,
            meta=meta,
        )
        self.bias = self.initializer.ones(
            shape=(shape[1],), dtype=float, retain_grad=True
        )

    def forward(self, x: Tensor):
        o1 = x.matmul(self.weight) + self.bias
        return o1
