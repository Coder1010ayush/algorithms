# <<<<<<<<<<<<<<<<<<<<<<<<< utf-8 encoding >>>>>>>>>>>>>>>>>>>>>>>>>>
import os
import sys
import numpy as np
from utils import asTensor

# <<<<<<<<<<<<<<<<<<<<<<<< base class or template for defining custom fnction for  backward gradient computation support >>>>>>>>>>>>>>>>>>>>>>>>>


def sum_axis(input_shape, grad_shape):
    axis = tuple(range(len(grad_shape) - len(input_shape)))
    offset = len(grad_shape) - len(input_shape)
    axis += tuple(
        i + offset
        for i, (s_in, s_g) in enumerate(zip(input_shape, grad_shape[offset:]))
        if s_in == 1
    )
    return axis


def handle_broadcasting_and_reshape(input, grad):
    if input.data.shape != grad.shape:
        axis = sum_axis(input.data.shape, grad.shape)
        grad = np.sum(grad, axis=axis, keepdims=True)
        grad = np.reshape(grad, input.data.shape)
    return grad


class BaseOperationHandler:
    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError("Forward pass not implemented for this operation.")

    def backward(self, out_grad):
        raise NotImplementedError("Backward pass not implemented for this operation.")


class AddElementWise(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        left, right = inputs
        data = left.data + right.data
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<AddElementWise>",
            creator=[left, right],
        )

    def backward(self, out_node):
        input_f, input_s = out_node.creator
        grad = out_node.grad
        grad_f = handle_broadcasting_and_reshape(input=input_f, grad=grad)
        grad_s = handle_broadcasting_and_reshape(input=input_s, grad=grad)
        input_f.grad = grad_f if input_f.grad is None else input_f.grad + grad_f
        input_s.grad = grad_s if input_s.grad is None else input_s.grad + grad_s


def add(f, g):
    return AddElementWise()([f, g])


class SubtractionElementWise(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        left, right = inputs
        data = left.data - right.data
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<SubtractionElementWise>",
            creator=[left, right],
        )

    def backward(self, out_node):
        input_f, input_s = out_node.creator
        grad = out_node.grad
        grad_f = handle_broadcasting_and_reshape(input=input_f, grad=grad)
        grad_s = handle_broadcasting_and_reshape(input=input_s, grad=grad)

        input_f.grad = grad_f if input_f.grad is None else input_f.grad + grad_f
        input_s.grad = -grad_s if input_s.grad is None else input_s.grad - grad_s


def sub(f, g):
    return SubtractionElementWise()([f, g])


class ElementWiseMultiply(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        left, right = inputs
        data = left.data * right.data
        return Tensor(
            data=data,
            retain_grad=left.retain_grad,
            operation="Backward<ElementWiseMultiply>",
            creator=[left, right],
        )

    def backward(self, out_grad):
        input_f, input_s = out_grad.creator
        grad = out_grad.grad
        grad_f = grad * input_s.data
        grad_s = grad * input_f.data

        grad_f = handle_broadcasting_and_reshape(input=input_f, grad=grad_f)
        grad_s = handle_broadcasting_and_reshape(input=input_s, grad=grad_s)

        input_f.grad = grad_f if input_f.grad is None else input_f.grad + grad_f
        input_s.grad = grad_s if input_s.grad is None else input_s.grad + grad_s


def mul(f, g):
    return ElementWiseMultiply()([f, g])


class ElementWiseDivision(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        left, right = inputs
        if np.any(right.data == 0):
            raise ValueError(
                "Division by zero detected in ElementWiseDivision forward pass."
            )
        data = left.data / right.data
        return Tensor(
            data=data,
            retain_grad=left.retain_grad,
            operation="Backward<ElementWiseDivision>",
            creator=[left, right],
        )

    def backward(self, out_grad):
        input_f, input_s = out_grad.creator
        grad = out_grad.grad
        grad_f = grad * (1 / input_s.data)
        grad_s = -grad * (input_f.data / (input_s.data**2))

        grad_f = handle_broadcasting_and_reshape(input=input_f, grad=grad_f)
        grad_s = handle_broadcasting_and_reshape(input=input_s, grad=grad_s)

        input_f.grad = grad_f if input_f.grad is None else input_f.grad + grad_f
        input_s.grad = grad_s if input_s.grad is None else input_s.grad + grad_s


def div(f, g):
    return ElementWiseDivision()([f, g])


def matmul(f, g):
    pass


class TransposeMatrix(BaseOperationHandler):

    def forward(self, inputs):
        input_f = inputs[0]
        data = input_f.data.T
        from tensor import Tensor

        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<TransposeMatrix>",
            creator=[input_f],
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad
        input_f.grad = grad.T


def transpose(f):
    return TransposeMatrix()([f])


class Permute(BaseOperationHandler):
    def forward(self, inputs, axes):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.transpose(input_f.data, axes)
        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<Permute>",
            creator=[input_f],
            meta={"axes": axes},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad
        axes = out_grad.meta["axes"]

        reverse_axes = np.argsort(axes)
        grad = np.transpose(grad, reverse_axes)
        input_f.grad = grad if input_f.grad is None else input_f.grad + grad


def permute(f, axes):
    return Permute()([f], axes)


class Reshape(BaseOperationHandler):
    def forward(self, inputs, new_shape):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.reshape(input_f.data, new_shape)
        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<Reshape>",
            creator=[input_f],
            meta={"original_shape": input_f.data.shape},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad
        original_shape = out_grad.meta["original_shape"]
        grad = np.reshape(grad, original_shape)
        input_f.grad = grad if input_f.grad is None else input_f.grad + grad


def reshape(f, new_shape):
    return Reshape()([f], new_shape)


class Sum(BaseOperationHandler):
    def forward(self, inputs, axis=None, keepdims=False):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.sum(input_f.data, axis=axis, keepdims=keepdims)
        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<Sum>",
            creator=[input_f],
            meta={"axis": axis, "keepdims": keepdims},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad
        axis = out_grad.meta["axis"]
        keepdims = out_grad.meta["keepdims"]

        if not keepdims and axis is not None:
            shape = list(input_f.data.shape)
            for ax in axis if isinstance(axis, tuple) else (axis,):
                shape[ax] = 1
            grad = np.reshape(grad, shape)

        input_f.grad = grad if input_f.grad is None else input_f.grad + grad


def sum(f, axis=None, keepdims=False):
    return Sum()([f], axis, keepdims)


class Mean(BaseOperationHandler):
    def forward(self, inputs, axis=None, keepdims=False):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.mean(input_f.data, axis=axis, keepdims=keepdims)
        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<Mean>",
            creator=[input_f],
            meta={"axis": axis, "keepdims": keepdims},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad
        axis = out_grad.meta["axis"]
        keepdims = out_grad.meta["keepdims"]

        if not keepdims and axis is not None:
            shape = list(input_f.data.shape)
            for ax in axis if isinstance(axis, tuple) else (axis,):
                shape[ax] = 1
            grad = np.reshape(grad, shape)

        scale = np.prod(
            [
                input_f.data.shape[ax]
                for ax in (axis if isinstance(axis, tuple) else [axis])
            ]
        )
        grad = grad / scale
        input_f.grad = grad if input_f.grad is None else input_f.grad + grad


def mean(f, axis=None, keepdims=False):
    return Mean()([f], axis, keepdims)


class Std(BaseOperationHandler):
    def forward(self, inputs, axis=None, keepdims=False):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.std(input_f.data, axis=axis, keepdims=keepdims)
        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<Std>",
            creator=[input_f],
            meta={"axis": axis, "keepdims": keepdims},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad
        axis = out_grad.meta["axis"]
        keepdims = out_grad.meta["keepdims"]
        mean_data = np.mean(input_f.data, axis=axis, keepdims=True)

        if not keepdims and axis is not None:
            shape = list(input_f.data.shape)
            for ax in axis if isinstance(axis, tuple) else (axis,):
                shape[ax] = 1
            grad = np.reshape(grad, shape)

        std_data = np.std(input_f.data, axis=axis, keepdims=True)
        grad = grad * (input_f.data - mean_data) / (std_data * np.prod(std_data.shape))
        input_f.grad = grad if input_f.grad is None else input_f.grad + grad


def std(f, axis=None, keepdims=False):
    return Std()([f], axis, keepdims)
