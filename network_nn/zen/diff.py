# <<<<<<<<<<<<<<<<<<<<<<<<< utf-8 encoding >>>>>>>>>>>>>>>>>>>>>>>>>>
import os
import sys
import zen
from zen.utils import asTensor

# <<<<<<<<<<<<<<<<<<<<<<<< base class or template for defining custom fnction for  backward gradient computation support >>>>>>>>>>>>>>>>>>>>>>>>>


class BaseOperationHandler:
    def __call__(self, *args):
        self.inputs = [asTensor(x) for x in args]
        return self.forward(inputs=self.inputs)

    def forward(self, inputs: list):
        raise NotImplementedError("not defined")

    def backward(self, grad):
        raise NotImplementedError("not defined")


class AddElementWise(BaseOperationHandler):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        left, right = inputs
        out = zen.Tensor(
            data=left + right,
            retain_grad=True,
            operation="Backward<AddElementWise>",
            creator=[left, right],
        )
        return out

    def backward(self):
        return super().backward()


def add(f, g):
    return AddElementWise(f, g)


def sub(f, g):
    pass


def mul(f, g):
    pass


def matmul(f, g):
    pass


def div(f, g):
    pass
