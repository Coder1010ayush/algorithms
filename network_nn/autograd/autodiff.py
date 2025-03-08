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


class MatMul(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        left, right = inputs
        data = np.matmul(left.data, right.data)
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<MatMul>",
            creator=[left, right],
        )

    def backward(self, out_grad):
        input_f, input_s = out_grad.creator
        grad = out_grad.grad
        grad_f = np.matmul(grad, input_s.data.T)
        grad_s = np.matmul(input_f.data.T, grad)

        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad_f)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad_f)
        )
        input_s.grad = (
            handle_broadcasting_and_reshape(input_s, grad_s)
            if input_s.grad is None
            else input_s.grad + handle_broadcasting_and_reshape(input_s, grad_s)
        )


def matmul(f, g):
    return MatMul()([f, g])


class Power(BaseOperationHandler):
    def forward(self, inputs, power):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.power(input_f.data, power)
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<Power>",
            creator=[input_f],
            meta={"power": power},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        power = out_grad.meta["power"]
        grad = out_grad.grad * power * np.power(input_f.data, power - 1)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def power(f, p):
    return Power()([f], power=p)


class Log(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.log(input_f.data)
        return Tensor(
            data=data, retain_grad=True, operation="Backward<Log>", creator=[input_f]
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad * (1 / input_f.data)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def log(f):
    return Log()([f])


class Exp(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.exp(input_f.data)
        return Tensor(
            data=data, retain_grad=True, operation="Backward<Exp>", creator=[input_f]
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad * np.exp(input_f.data)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def exp(f):
    return Exp()([f])


class Sqrt(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.sqrt(input_f.data)
        return Tensor(
            data=data, retain_grad=True, operation="Backward<Sqrt>", creator=[input_f]
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad * (0.5 * np.power(input_f.data, -0.5))
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def sqrt(f):
    return Sqrt()([f])


class ReLU(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.maximum(0, input_f.data)
        return Tensor(
            data=data, retain_grad=True, operation="Backward<ReLU>", creator=[input_f]
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad * (input_f.data > 0)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def relu(f):
    return ReLU()([f])


class Sigmoid(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        data = 1 / (1 + np.exp(-input_f.data))
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<Sigmoid>",
            creator=[input_f],
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        sigmoid_out = 1 / (1 + np.exp(-input_f.data))
        grad = out_grad.grad * sigmoid_out * (1 - sigmoid_out)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def sigmoid(f):
    return Sigmoid()([f])


class Tanh(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.tanh(input_f.data)
        return Tensor(
            data=data, retain_grad=True, operation="Backward<Tanh>", creator=[input_f]
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad * (1 - np.tanh(input_f.data) ** 2)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def tanh(f):
    return Tanh()([f])


class Softmax(BaseOperationHandler):
    def forward(self, inputs, axis=-1):
        from tensor import Tensor

        input_f = inputs[0]
        exps = np.exp(input_f.data - np.max(input_f.data, axis=axis, keepdims=True))
        data = exps / np.sum(exps, axis=axis, keepdims=True)
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<Softmax>",
            creator=[input_f],
            meta={"axis": axis},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        axis = out_grad.meta["axis"]
        grad = out_grad.grad
        softmax_out = out_grad.data
        grad = softmax_out * (
            grad - np.sum(grad * softmax_out, axis=axis, keepdims=True)
        )
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def softmax(f, axis=-1):
    return Softmax()([f], axis=axis)


class Clip(BaseOperationHandler):
    def forward(self, inputs, min_val, max_val):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.clip(input_f.data, min_val, max_val)
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<Clip>",
            creator=[input_f],
            meta={"min_val": min_val, "max_val": max_val},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        min_val, max_val = out_grad.meta["min_val"], out_grad.meta["max_val"]
        grad = out_grad.grad * ((input_f.data >= min_val) & (input_f.data <= max_val))
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def clip(f, min_val, max_val):
    return Clip()([f], min_val=min_val, max_val=max_val)


class Abs(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.abs(input_f.data)
        return Tensor(
            data=data, retain_grad=True, operation="Backward<Abs>", creator=[input_f]
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad * np.sign(input_f.data)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def abs(f):
    return Abs()([f])


class Negative(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        data = -input_f.data
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<Negative>",
            creator=[input_f],
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = -out_grad.grad
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def neg(f):
    return Negative()([f])


class ReduceSum(BaseOperationHandler):
    def forward(self, inputs, axis=None, keepdims=False):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.sum(input_f.data, axis=axis, keepdims=keepdims)
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<ReduceSum>",
            creator=[input_f],
            meta={"axis": axis, "keepdims": keepdims},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        axis = out_grad.meta["axis"]
        expanded_grad = (
            np.expand_dims(out_grad.grad, axis=axis)
            if axis is not None
            else out_grad.grad
        )
        grad = np.broadcast_to(expanded_grad, input_f.data.shape)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def reduce_sum(f, axis=None, keepdims=False):
    return ReduceSum()([f], axis=axis, keepdims=keepdims)


class Broadcast(BaseOperationHandler):
    def forward(self, inputs, shape):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.broadcast_to(input_f.data, shape)
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<Broadcast>",
            creator=[input_f],
            meta={"shape": shape},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = handle_broadcasting_and_reshape(input_f, out_grad.grad)
        input_f.grad = grad if input_f.grad is None else input_f.grad + grad


def broadcast(f, shape):
    return Broadcast()([f], shape=shape)


class Concat(BaseOperationHandler):
    def forward(self, inputs, axis=0):
        from tensor import Tensor

        data = np.concatenate([i.data for i in inputs], axis=axis)
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<Concat>",
            creator=inputs,
            meta={"axis": axis},
        )

    def backward(self, out_grad):
        axis = out_grad.meta["axis"]
        splits = np.split(
            out_grad.grad,
            [i.data.shape[axis] for i in out_grad.creator[:-1]],
            axis=axis,
        )
        for i, inp in enumerate(out_grad.creator):
            inp.grad = splits[i] if inp.grad is None else inp.grad + splits[i]


def concat(*args, axis=0):
    return Concat()(list(args), axis=axis)


class Pad(BaseOperationHandler):
    def forward(self, inputs, pad_width, mode="constant", constant_values=0):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.pad(
            input_f.data, pad_width, mode=mode, constant_values=constant_values
        )
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<Pad>",
            creator=[input_f],
            meta={"pad_width": pad_width},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        slices = tuple(
            slice(p[0], -p[1] if p[1] > 0 else None) for p in out_grad.meta["pad_width"]
        )
        grad = out_grad.grad[slices]
        input_f.grad = grad if input_f.grad is None else input_f.grad + grad


def pad(f, pad_width, mode="constant", constant_values=0):
    return Pad()([f], pad_width=pad_width, mode=mode, constant_values=constant_values)
