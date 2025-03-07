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

        if input_f.data.shape != grad.shape:
            axis_f = sum_axis(input_f.data.shape, grad.shape)
            grad_f = np.sum(grad, axis=axis_f, keepdims=True)
            grad_f = np.reshape(grad_f, input_f.data.shape)
        else:
            grad_f = grad

        if input_s.data.shape != grad.shape:
            axis_s = sum_axis(input_s.data.shape, grad.shape)
            grad_s = np.sum(grad, axis=axis_s, keepdims=True)
            grad_s = np.reshape(grad_s, input_s.data.shape)
        else:
            grad_s = grad

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
        if input_f.data.shape != grad.shape:
            axis_f = sum_axis(input_f.data.shape, grad.shape)
            grad_f = np.sum(grad, axis=axis_f, keepdims=True)
            grad_f = np.reshape(grad_f, input_f.data.shape)
        else:
            grad_f = grad

        if input_s.data.shape != grad.shape:
            axis_s = sum_axis(input_s.data.shape, grad.shape)
            grad_s = np.sum(grad, axis=axis_s, keepdims=True)
            grad_s = np.reshape(grad_s, input_s.data.shape)
        else:
            grad_s = grad

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
        input_f, input_s = out_grad
        grad = out_grad.grad
        grad_f = grad * input_s.data
        grad_s = grad * input_f.data

        if input_f.data.shape != grad_f.shape:
            axis_f = sum_axis(input_f.data.shape, grad_f.shape)
            grad_f = np.sum(grad_f, axis=axis_f, keepdims=True)
            grad_f = np.reshape(grad_f, input_f.data.shape)

        if input_s.data.shape != grad_s.shape:
            axis_s = sum_axis(input_s.data.shape, grad_s.shape)
            grad_s = np.sum(grad_s, axis=axis_s, keepdims=True)
            grad_s = np.reshape(grad_s, input_s.data.shape)

        input_f.grad = grad_f if input_f.grad is None else input_f.grad + grad_f
        input_s.grad = grad_s if input_s.grad is None else input_s.grad + grad_s


def mul(f, g):
    return ElementWiseMultiply()([f, g])


class ElementWiseDivision(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        left, right = inputs
        data = left.data / right.data
        return Tensor(
            data=data,
            retain_grad=left.retain_grad,
            operation="Backward<ElementWiseDivision>",
            creator=[left, right],
        )

    def backward(self, out_grad):
        input_f, input_s = out_grad
        grad = out_grad.grad
        grad_f = grad * 1 / input_s.data
        grad_s = -grad * (input_f.data / (input_s.data**2))

        if input_f.data.shape != grad_f.shape:
            axis_f = sum_axis(input_f.data.shape, grad_f.shape)
            grad_f = np.sum(grad_f, axis=axis_f, keepdims=True)
            grad_f = np.reshape(grad_f, input_f.data.shape)

        if input_s.data.shape != grad_s.shape:
            axis_s = sum_axis(input_s.data.shape, grad_s.shape)
            grad_s = np.sum(grad_s, axis=axis_s, keepdims=True)
            grad_s = np.reshape(grad_s, input_s.data.shape)

        input_f.grad = grad_f if input_f.grad is None else input_f.grad + grad_f
        input_s.grad = grad_s if input_s.grad is None else input_s.grad + grad_s


def div(f, g):
    return ElementWiseDivision()([f, g])


def matmul(f, g):
    pass
