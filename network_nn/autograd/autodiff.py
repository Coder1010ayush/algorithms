# <<<<<<<<<<<<<<<<<<<<<<<<< utf-8 encoding >>>>>>>>>>>>>>>>>>>>>>>>>>
import os
import sys
import numpy as np
from utils import asTensor

# <<<<<<<<<<<<<<<<<<<<<<<< base class or template for defining custom fnction for  backward gradient computation support >>>>>>>>>>>>>>>>>>>>>>>>>
# in depth implementation for forward and backprop for all the major operations which i use in pytorch more often.
# this is highly flexible and easy to add any new operation ( just have to add a class with forward and backward function that's it. )

# def sum_axis(input_shape, grad_shape): # initial version has always some limitation ( ahh )
#     input_ndim = len(input_shape)
#     grad_ndim = len(grad_shape)
#     sum_axes = []
#     for i in range(1, max(input_ndim, grad_ndim) + 1):
#         input_dim = input_shape[-i] if i <= input_ndim else 1
#         grad_dim = grad_shape[-i] if i <= grad_ndim else 1
#         if input_dim != grad_dim and grad_dim == 1:
#             sum_axes.append(grad_ndim - i)
#     for i in range(grad_ndim - input_ndim):
#         sum_axes.append(i)
#     sum_axes = list(sorted(set(sum_axes)))
#     return tuple(sum_axes)

def sum_axis(input_shape, grad_shape):
    axis = tuple(range(len(grad_shape) - len(input_shape)))
    offset = len(grad_shape) - len(input_shape)
    axis += tuple(
        i + offset
        for i, (s_in, s_g) in enumerate(zip(input_shape, grad_shape[offset:]))
        if s_in == 1
    )
    return axis


# Broadcasting Rules:
# 1. Align Shapes from Right:
#    - Broadcasting compares shapes element-wise starting from the rightmost dimension.
#    - Example:
#         Shape A:      (3, 4)
#         Shape B:          (4)
#      Valid broadcast because dimensions align from the right.

# 2. Dimension Compatibility:
#    - Two dimensions are compatible if:
#      a. They are equal, or
#      b. One of them is 1.
#    - Example:
#         Shape A: (5, 3, 4)
#         Shape B:      (1, 4)
#      Result:     (5, 3, 4) - Valid broadcast due to rule (b).

# 3. Expand Missing Dimensions:
#    - If a tensor has fewer dimensions, prepend 1s to match the other tensor's shape.
#    - Example:
#         Shape A: (4, 3)
#         Shape B:    (3)
#      Interpreted as:
#         Shape A: (4, 3)
#         Shape B: (1, 3) - Expanded shape

# 4. Output Shape Determination:
#    - The output shape is the maximum size along each dimension.
#    - Example:
#         Shape A: (5, 1, 4)
#         Shape B:    (3, 1)
#      Result:     (5, 3, 4)

# 5. Broadcasting Fails When:
#    - A dimension is neither 1 nor equal between tensors.
#    - Example:
#         Shape A: (3, 4)
#         Shape B: (2, 4)
#      Error: Incompatible shapes.

# Summary:
#    - Right-align shapes, use 1 as a flexible dimension.
#    - Expand or replicate dimensions with 1 to match the larger shape.
#    - If any dimension is incompatible, broadcasting raises an error.


def handle_broadcasting_and_reshape(input, grad):
    if np.isscalar(grad) or grad.shape == ():
        grad = np.full(input.data.shape, grad)

    if input.data.shape != grad.shape:
        axis = sum_axis(input.data.shape, grad.shape)
        grad = np.sum(grad, axis=axis, keepdims=True)

        try:
            grad = np.reshape(grad, input.data.shape)
        except ValueError:
            grad = np.squeeze(
                grad, axis=axis
            )  # quite forward (does not need to explaination)
    return grad


def ensure_minimum_dims(array, min_dims=1):
    while array.ndim < min_dims:
        array = np.expand_dims(array, axis=0)
    return array


class BaseOperationHandler:
    def __call__(self, *inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError("Forward pass not implemented for this operation.")

    def backward(self, out_grad):
        raise NotImplementedError("Backward pass not implemented for this operation.")


class Stack(BaseOperationHandler):
    def forward(self, inputs, axis=0):
        from tensor import Tensor

        tensors = inputs[0]
        axis = inputs[1]

        data = np.stack([t.data for t in tensors], axis=axis)
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<Stack>",
            creator=tensors,
            meta={"axis": axis},
        )

    def backward(self, out_grad):
        axis = out_grad.meta["axis"]
        input_tensors = out_grad.creator

        # Split the gradient along the stacking axis
        grads = np.split(out_grad.grad, len(input_tensors), axis=axis)
        # magic happens here ( it is necessary )
        # Squeeze the axis used for splitting to revert the stacking
        grads = [np.squeeze(g, axis=axis) for g in grads]

        for i, tensor in enumerate(input_tensors):
            grad = handle_broadcasting_and_reshape(tensor, grads[i])
            if tensor.grad is not None:
                tensor.grad += grad
            else:
                tensor.grad = grad


def stack(tensors, axis=0):
    return Stack()(tensors, axis)

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
    return AddElementWise()(f, g)


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
    return SubtractionElementWise()(f, g)


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
    return ElementWiseMultiply()(f, g)


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
    return ElementWiseDivision()(f, g)


class TransposeMatrix(BaseOperationHandler):

    def forward(self, inputs):
        input_f = inputs[0]
        axis = inputs[1]
        data = np.transpose(input_f.data , axis)
        from tensor import Tensor

        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<TransposeMatrix>",
            creator=[input_f],
            meta= {"axis" : axis}
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad
        input_f.grad = np.transpose(grad , out_grad.meta["axis"])


def transpose(f  , axis = -1):
    return TransposeMatrix()(f ,axis)


class Permute(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        axes = inputs[1]
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
    return Permute()(f, axes)


class Reshape(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        new_shape = inputs[1]
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
    return Reshape()(f, new_shape)


class Sum(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        axis = inputs[1] if len(inputs) > 1 else None
        keepdims = inputs[2] if len(inputs) > 2 else False

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

        if input_f.grad is None:
            input_f.grad = grad
        else:
            input_f.grad += grad


def sum(f, axis=None, keepdims=False):
    return Sum()(f, axis, keepdims)


class Mean(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        axis = inputs[1]
        keepdims = inputs[2]
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
    return Mean()(f, axis, keepdims)


class Std(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        axis = inputs[1]
        keepdims = inputs[2]
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
    return Std()(f, axis, keepdims)


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

        if np.isscalar(grad) or grad.shape == ():
            grad = np.ones_like(out_grad.data)

        shape_f = input_f.data.shape
        shape_s = input_s.data.shape

        if len(shape_s) == 1:
            grad_f = np.matmul(grad[..., np.newaxis], input_s.data[np.newaxis, :])
            grad_s = np.matmul(input_f.data.T, grad)
        elif len(shape_f) == 1:
            grad_f = np.matmul(grad[:, np.newaxis], input_s.data[np.newaxis, :])
            grad_s = np.matmul(input_f.data, grad)
        else:
            grad_f = np.matmul(
                grad, np.expand_dims(input_s.data, axis=0).swapaxes(-1, -2)
            )
            grad_s = np.matmul(
                np.expand_dims(input_f.data, axis=0).swapaxes(-1, -2), grad
            )

        while grad_f.ndim > len(shape_f):
            grad_f = grad_f.sum(axis=0)
        for i, dim in enumerate(shape_f):
            if dim == 1:
                grad_f = grad_f.sum(axis=i, keepdims=True)

        while grad_s.ndim > len(shape_s):
            grad_s = grad_s.sum(axis=0)
        for i, dim in enumerate(shape_s):
            if dim == 1:
                grad_s = grad_s.sum(axis=i, keepdims=True)

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
    return MatMul()(f, g)


class Power(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        power = inputs[1]
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
    return Power()(f, p)


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
    return Log()(f)


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
    return Exp()(f)


class Sin(BaseOperationHandler):

    def forward(self, inputs):
        input_f = inputs[0]
        from tensor import Tensor

        data = np.sin(input_f.data)
        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<Sin>",
            creator=[input_f],
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad * np.cos(input_f.data)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def sin(f):
    return Sin()(f)


class Cos(BaseOperationHandler):

    def forward(self, inputs):
        input_f = inputs[0]
        from tensor import Tensor

        data = np.cos(input_f.data)
        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<Cos>",
            creator=[input_f],
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = -out_grad.grad * np.sin(input_f.data)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def cos(f):
    return Cos()(f)


class Tan(BaseOperationHandler):

    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        data = np.tan(input_f.data)
        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<Tan>",
            creator=[input_f],
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = out_grad.grad * (1 / np.cos(input_f.data))
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def tan(f):
    return Tan()(f)


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
    return Sqrt()(f)


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
    return ReLU()(f)


class LeakyRELU(BaseOperationHandler):

    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        alpha = inputs[1]

        data = np.maximum(alpha * input_f.data, input_f.data)
        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<LeakyRELU>",
            creator=[input_f],
            meta={"alpha": alpha},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        alpha = input_f.meta["alpha"]
        grad = out_grad.grad

        local_grad = np.where(input_f.data > 0, 1, alpha)
        grad = grad * local_grad

        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def leaky_relu(f, alpha=0.1):
    return LeakyRELU()(f, alpha)


class MSE(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        predictions, targets = inputs
        data = np.mean((predictions.data - targets.data) ** 2)
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<MSE>",
            creator=[predictions, targets],
        )

    def backward(self, out_grad):
        predictions, targets = out_grad.creator
        grad = (2 * (predictions.data - targets.data)) / targets.data.size
        predictions.grad = (
            handle_broadcasting_and_reshape(predictions, grad * out_grad.grad)
            if predictions.grad is None
            else predictions.grad
            + handle_broadcasting_and_reshape(predictions, grad * out_grad.grad)
        )
        if targets.retain_grad:
            targets.grad = (
                handle_broadcasting_and_reshape(targets, -grad * out_grad.grad)
                if targets.grad is None
                else targets.grad
                + handle_broadcasting_and_reshape(targets, -grad * out_grad.grad)
            )


def mse(predictions, targets):
    return MSE()(predictions, targets)


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
    return Sigmoid()(f)


class Swish(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        data = input_f.data * (1 / (1 + np.exp(-input_f.data)))
        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<Swish>",
            creator=[input_f],
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        sigmoid_out = 1 / (1 + np.exp(-input_f.data))
        grad = (
            sigmoid_out + input_f.data * sigmoid_out * (1 - sigmoid_out)
        ) * out_grad.grad

        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def swish(f):
    return Swish()(f)


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
    return Tanh()(f)


class LayerNormalization(BaseOperationHandler):

    def forward(self, inputs):
        input_f = inputs[0]
        mu = np.mean(input_f.data)
        variaence = np.var(input_f.data)
        ep = 1e-8
        data = (input_f.data - mu) / (np.sqrt(variaence**2 + ep))
        from tensor import Tensor

        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation="Backward<LayerNormalization>",
            creator=[input_f],
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        N = input_f.data.size
        std_inv = 1.0 / np.sqrt(self.variance + self.ep)

        dnorm = out_grad.grad
        # magic happens here
        dvar = np.sum(dnorm * (input_f.data - self.mu) * -0.5 * std_inv**3, axis=0)
        dmu = np.sum(dnorm * -std_inv, axis=0) + dvar * np.mean(
            -2.0 * (input_f.data - self.mu), axis=0
        )
        grad = (dnorm * std_inv) + (dvar * 2 * (input_f.data - self.mu) / N) + (dmu / N)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )


def layer_norm(f):
    return LayerNormalization()(f)


class Softmax(BaseOperationHandler):
    def forward(self, inputs, axis=-1):
        from tensor import Tensor

        input_f = inputs[0]
        axis = inputs[1]
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
    return Softmax()(f, axis)


class Clip(BaseOperationHandler):
    def forward(self, inputs, min_val, max_val):
        from tensor import Tensor

        input_f = inputs[0]
        min_val = inputs[1]
        max_val = inputs[2]
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
    return Clip()(f, min_val, max_val)


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
    return Abs()(f)

class Unsqueeze(BaseOperationHandler):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        input_f = inputs[0]
        from tensor import Tensor

        data = np.expand_dims(input_f.data, axis=self.axis)
        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation=f"Backward<Unsqueeze(axis={self.axis})>",
            creator=[input_f],
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = np.squeeze(out_grad.grad, axis=self.axis)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )

def unsqueeze(f, axis):
    return Unsqueeze(axis)(f)

class Squeeze(BaseOperationHandler):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def forward(self, inputs):
        input_f = inputs[0]
        from tensor import Tensor

        data = np.squeeze(input_f.data, axis=self.axis)
        return Tensor(
            data=data,
            retain_grad=input_f.retain_grad,
            operation=f"Backward<Squeeze(axis={self.axis})>",
            creator=[input_f],
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        grad = np.expand_dims(out_grad.grad, axis=self.axis)
        input_f.grad = (
            handle_broadcasting_and_reshape(input_f, grad)
            if input_f.grad is None
            else input_f.grad + handle_broadcasting_and_reshape(input_f, grad)
        )

def squeeze(f, axis=None):
    return Squeeze(axis)(f)

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
    return Negative()(f)

# class Stack(BaseOperationHandler): # stack operation
#     def forward(self, inputs, axis=0):
#         from tensor import Tensor

#         tensors = inputs[0] 
#         axis = inputs[1] 

#         data = np.stack([t.data for t in tensors], axis=axis)
#         return Tensor(
#             data=data,
#             retain_grad=True,
#             operation="Backward<Stack>",
#             creator=tensors,
#             meta={"axis": axis},
#         )

#     def backward(self, out_grad):
#         axis = out_grad.meta["axis"]
#         input_tensors = out_grad.creator

#         grads = np.split(out_grad.grad, len(input_tensors), axis=axis)

#         for i, tensor in enumerate(input_tensors):
#             grad = handle_broadcasting_and_reshape(tensor, grads[i])
#             tensor.grad = grad if tensor.grad is None else tensor.grad + grad


# def stack(tensors, axis=0):
#     return Stack()(tensors, axis)

class ReduceSum(BaseOperationHandler):
    def forward(self, inputs, axis=None, keepdims=False):
        from tensor import Tensor

        input_f = inputs[0]
        axis = inputs[1]
        keepdims = inputs[2]
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
    return ReduceSum()(f, axis, keepdims)


class Broadcast(BaseOperationHandler):
    def forward(self, inputs, shape):
        from tensor import Tensor

        input_f = inputs[0]
        shape = inputs[1]
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
    return Broadcast()(f, shape)


class Concat(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor
        inputs_data = inputs[0]
        axis = inputs[1]
        data = np.concatenate([i.data for i in inputs_data], axis=axis)
        return Tensor(
            data=data,
            retain_grad=True,
            operation="Backward<Concat>",
            creator=inputs_data,
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


def concat(*args):
    return Concat()(list(args[0]), args[1])


class Pad(BaseOperationHandler):
    def forward(self, inputs, pad_width, mode="constant", constant_values=0):
        from tensor import Tensor

        input_f = inputs[0]
        pad_width = inputs[1]
        mode = inputs[2]
        constant_values = inputs[3]
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
    return Pad()(f, pad_width, mode, constant_values)


class Slice(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        key = inputs[1]
        # input_f , key = inputs
        sliced_data = input_f.data[key]
        return Tensor(
            data=sliced_data,
            retain_grad=True,
            operation="Backward<Slice>",
            creator=[input_f],
            meta={"key": key},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        key = out_grad.meta["key"]

        grad = np.zeros_like(input_f.data)
        grad[key] = out_grad.grad
        input_f.grad = grad if input_f.grad is None else input_f.grad + grad


def slice_tensor(f, key):
    return Slice()(f, key)


class SetItem(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        input_f = inputs[0]
        key = inputs[1]
        value = inputs[2]
        new_data = input_f.data.copy()

        if isinstance(value, Tensor):
            value = value.data
        elif not isinstance(value, np.ndarray):
            value = np.array(value, dtype=input_f.data.dtype)

        new_data[key] = value
        return Tensor(
            data=new_data,
            retain_grad=True,
            operation="Backward<SetItem>",
            creator=[input_f],
            meta={"key": key},
        )

    def backward(self, out_grad):
        input_f = out_grad.creator[0]
        key = out_grad.meta["key"]
        grad = np.zeros_like(input_f.data)
        grad[key] = out_grad.grad[key]
        input_f.grad = grad if input_f.grad is None else input_f.grad + grad


def set_item(f, key, value):
    return SetItem()(f, key, value)


# thanks to my semester Prof. Prerna Mukherji for helping to implement conv backward propogation method
class Conv1D(BaseOperationHandler):
    def forward(self, inputs):
        input_f = inputs[0]
        filters = inputs[1]
        stride = inputs[2]
        padding = inputs[3]
        input_data = input_f.data
        filter_data = filters.data

        batch_size, channels, input_len = input_data.shape
        num_filters, _, filter_len = filter_data.shape

        if padding == "same":
            pad = (filter_len - 1) // 2
            input_data = np.pad(
                input_data, ((0, 0), (0, 0), (pad, pad)), mode="constant"
            )
        elif padding != "valid":
            raise ValueError("Padding must be 'same' or 'valid'.")

        output_len = (input_data.shape[2] - filter_len) // stride + 1
        output = np.zeros((batch_size, num_filters, output_len))

        for b in range(batch_size):  # Batch dimension
            for f in range(num_filters):  # Filter dimension
                for i in range(output_len):
                    for c in range(channels):  # Channel dimension
                        segment = input_data[b, c, i * stride : i * stride + filter_len]
                        output[b, f, i] += np.sum(segment * filter_data[f, c])

        from tensor import Tensor

        return Tensor(
            data=output,
            retain_grad=True,
            operation="Backward<Conv1D>",
            creator=[input_f, filters],
            meta={
                "input_shape": input_f.data.shape,
                "filter_shape": filter_data.shape,
                "stride": stride,
                "padding": padding,
            },
        )

    def backward(self, out_grad):
        input_f, filters = out_grad.creator
        input_shape = out_grad.meta["input_shape"]
        filter_shape = out_grad.meta["filter_shape"]
        stride = out_grad.meta["stride"]
        padding = out_grad.meta["padding"]

        batch_size, channels, input_len = input_shape
        num_filters, _, filter_len = filter_shape

        grad_input = np.zeros_like(input_f.data)
        grad_filters = np.zeros_like(filters.data)

        if padding == "same":
            pad = (filter_len - 1) // 2
            input_padded = np.pad(
                input_f.data, ((0, 0), (0, 0), (pad, pad)), mode="constant"
            )
            grad_input_padded = np.pad(
                grad_input, ((0, 0), (0, 0), (pad, pad)), mode="constant"
            )
        else:
            input_padded = input_f.data
            grad_input_padded = grad_input

        for b in range(batch_size):
            for f in range(num_filters):
                for i in range(out_grad.data.shape[2]):
                    for c in range(channels):
                        start = i * stride
                        end = start + filter_len
                        # magic happens for me
                        # print(filters.data.shape)
                        grad_input_padded[b, c, start:end] += (
                            # minor fix
                            # filters.data[f, c] * out_grad.grad[b, f, i]
                            filters.data[f, c]
                            * (
                                out_grad.grad
                                if (
                                    np.isscalar(out_grad.grad)
                                    or out_grad.grad.shape == ()
                                )
                                else out_grad.grad[b, f, i]
                            )
                        )
                        grad_filters[f, c] += (
                            # minor fix
                            # input_padded[b, c, start:end] * out_grad.grad[b, f, i]
                            input_padded[b, c, start:end]
                            * (
                                out_grad.grad
                                if (
                                    np.isscalar(out_grad.grad)
                                    or out_grad.grad.shape == ()
                                )
                                else out_grad.grad[b, f, i]
                            )
                        )

        if padding == "same":
            grad_input = (
                grad_input_padded[:, :, pad:-pad] if pad > 0 else grad_input_padded
            )
        else:
            grad_input = grad_input_padded

        input_f.grad = grad_input if input_f.grad is None else input_f.grad + grad_input
        filters.grad = (
            grad_filters if filters.grad is None else filters.grad + grad_filters
        )


def conv1d(f, filters, stride=1, padding="valid"):
    return Conv1D()(f, filters, stride, padding)


class Conv2D(BaseOperationHandler):
    def forward(self, inputs):
        input_f = inputs[0]
        filters = inputs[1]
        stride = inputs[2]
        padding = inputs[3]
        input_data = input_f.data
        filter_data = filters.data

        batch_size, channels, input_h, input_w = input_data.shape
        num_filters, _, filter_h, filter_w = filter_data.shape

        if padding == "same":
            pad_h = (filter_h - 1) // 2
            pad_w = (filter_w - 1) // 2
            input_data = np.pad(
                input_data,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
            )
        elif padding != "valid":
            raise ValueError("Padding must be 'same' or 'valid'.")

        output_h = (input_data.shape[2] - filter_h) // stride + 1
        output_w = (input_data.shape[3] - filter_w) // stride + 1
        output = np.zeros((batch_size, num_filters, output_h, output_w))

        for b in range(batch_size):
            for f in range(num_filters):
                for i in range(output_h):
                    for j in range(output_w):
                        for c in range(channels):
                            h_start = i * stride
                            w_start = j * stride
                            segment = input_data[
                                b,
                                c,
                                h_start : h_start + filter_h,
                                w_start : w_start + filter_w,
                            ]
                            output[b, f, i, j] += np.sum(segment * filter_data[f, c])
        from tensor import Tensor

        return Tensor(
            data=output,
            retain_grad=True,
            operation="Backward<Conv2D>",
            creator=[input_f, filters],
            meta={
                "input_shape": input_f.data.shape,
                "filter_shape": filter_data.shape,
                "stride": stride,
                "padding": padding,
            },
        )

    def backward(self, out_grad):
        input_f, filters = out_grad.creator
        input_data = input_f.data
        filter_data = filters.data
        stride = out_grad.meta["stride"]
        padding = out_grad.meta["padding"]

        batch_size, channels, input_h, input_w = input_data.shape
        num_filters, _, filter_h, filter_w = filter_data.shape

        if padding == "same":
            pad_h = (filter_h - 1) // 2
            pad_w = (filter_w - 1) // 2
            input_data = np.pad(
                input_data,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
            )

        grad_input = np.zeros_like(input_data)
        grad_filters = np.zeros_like(filter_data)

        for b in range(out_grad.data.shape[0]):
            for f in range(num_filters):
                for i in range(out_grad.data.shape[2]):
                    for j in range(out_grad.data.shape[3]):
                        for c in range(channels):
                            h_start = i * stride
                            w_start = j * stride
                            segment = input_data[
                                b,
                                c,
                                h_start : h_start + filter_h,
                                w_start : w_start + filter_w,
                            ]
                            grad_filters[f, c] += segment * out_grad.data[b, f, i, j]
                            grad_input[
                                b,
                                c,
                                h_start : h_start + filter_h,
                                w_start : w_start + filter_w,
                            ] += (
                                filter_data[f, c] * out_grad.data[b, f, i, j]
                            )

        if padding == "same":
            grad_input = grad_input[:, :, pad_h:-pad_h, pad_w:-pad_w]

        input_f.grad = grad_input if input_f.grad is None else input_f.grad + grad_input
        filters.grad = (
            grad_filters if filters.grad is None else filters.grad + grad_filters
        )


def conv2d(f, filters, stride=1, padding="valid"):
    return Conv2D()(f, filters, stride, padding)


class Conv3D(BaseOperationHandler):
    def forward(self, inputs):
        input_f = inputs[0]
        filters = inputs[1]
        stride = inputs[2]
        padding = inputs[3]
        input_data = input_f.data
        filter_data = filters.data

        batch_size, channels, input_d, input_h, input_w = input_data.shape
        num_filters, _, filter_d, filter_h, filter_w = filter_data.shape

        if padding == "same":
            pad_d = (filter_d - 1) // 2
            pad_h = (filter_h - 1) // 2
            pad_w = (filter_w - 1) // 2
            input_data = np.pad(
                input_data,
                ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
            )
        elif padding != "valid":
            raise ValueError("Padding must be 'same' or 'valid'.")

        output_d = (input_data.shape[2] - filter_d) // stride + 1
        output_h = (input_data.shape[3] - filter_h) // stride + 1
        output_w = (input_data.shape[4] - filter_w) // stride + 1
        output = np.zeros((batch_size, num_filters, output_d, output_h, output_w))

        for b in range(batch_size):
            for f in range(num_filters):
                for d in range(output_d):
                    for i in range(output_h):
                        for j in range(output_w):
                            for c in range(channels):
                                d_start = d * stride
                                h_start = i * stride
                                w_start = j * stride
                                segment = input_data[
                                    b,
                                    c,
                                    d_start : d_start + filter_d,
                                    h_start : h_start + filter_h,
                                    w_start : w_start + filter_w,
                                ]
                                output[b, f, d, i, j] += np.sum(
                                    segment * filter_data[f, c]
                                )
        from tensor import Tensor
        return Tensor(
            data=output,
            retain_grad=True,
            operation="Backward<Conv3D>",
            creator=[input_f, filters],
            meta={
                "input_shape": input_f.data.shape,
                "filter_shape": filter_data.shape,
                "stride": stride,
                "padding": padding,
            },
        )

    def backward(self, out_grad):
        input_f, filters = out_grad.creator
        input_data = input_f.data
        filter_data = filters.data
        stride = out_grad.meta["stride"]
        padding = out_grad.meta["padding"]

        batch_size, channels, input_d, input_h, input_w = input_data.shape
        num_filters, _, filter_d, filter_h, filter_w = filter_data.shape

        if padding == "same":
            pad_d = (filter_d - 1) // 2
            pad_h = (filter_h - 1) // 2
            pad_w = (filter_w - 1) // 2
            input_data = np.pad(
                input_data,
                ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
            )

        grad_input = np.zeros_like(input_data)
        grad_filters = np.zeros_like(filter_data)

        for b in range(batch_size):
            for f in range(num_filters):
                for d in range(out_grad.data.shape[2]):
                    for i in range(out_grad.data.shape[3]):
                        for j in range(out_grad.data.shape[4]):
                            for c in range(channels):
                                d_start = d * stride
                                h_start = i * stride
                                w_start = j * stride

                                segment = input_data[
                                    b,
                                    c,
                                    d_start : d_start + filter_d,
                                    h_start : h_start + filter_h,
                                    w_start : w_start + filter_w,
                                ]
                                grad_filters[f, c] += (
                                    segment * out_grad.data[b, f, d, i, j]
                                )
                                grad_input[
                                    b,
                                    c,
                                    d_start : d_start + filter_d,
                                    h_start : h_start + filter_h,
                                    w_start : w_start + filter_w,
                                ] += (
                                    filter_data[f, c] * out_grad.data[b, f, d, i, j]
                                )

        if padding == "same":
            grad_input = grad_input[:, :, pad_d:-pad_d, pad_h:-pad_h, pad_w:-pad_w]

        input_f.grad = grad_input if input_f.grad is None else input_f.grad + grad_input
        filters.grad = (
            grad_filters if filters.grad is None else filters.grad + grad_filters
        )

def conv3d(f, filters, stride=1, padding="valid"):
    return Conv3D()(f, filters, stride, padding)


class Conv3DOptimized(BaseOperationHandler):
    def im3col(self, input_data, filter_shape, stride):
        batch_size, channels, input_d, input_h, input_w = input_data.shape
        filter_d, filter_h, filter_w = filter_shape

        output_d = (input_d - filter_d) // stride + 1
        output_h = (input_h - filter_h) // stride + 1
        output_w = (input_w - filter_w) // stride + 1

        col = np.zeros((batch_size, channels, filter_d, filter_h, filter_w, output_d, output_h, output_w))

        for d in range(filter_d):
            for i in range(filter_h):
                for j in range(filter_w):
                    col[:, :, d, i, j, :, :, :] = input_data[
                        :, :, d : d + output_d * stride : stride,
                        i : i + output_h * stride : stride,
                        j : j + output_w * stride : stride
                    ]

        return col.reshape(batch_size, -1, output_d * output_h * output_w)

    def forward(self, inputs):
        input_f = inputs[0]
        filters = inputs[1]
        stride = inputs[2]
        padding = inputs[3]
        input_data = input_f.data
        filter_data = filters.data

        batch_size, channels, input_d, input_h, input_w = input_data.shape
        num_filters, _, filter_d, filter_h, filter_w = filter_data.shape

        if padding == "same":
            pad_d = (filter_d - 1) // 2
            pad_h = (filter_h - 1) // 2
            pad_w = (filter_w - 1) // 2
            input_data = np.pad(
                input_data,
                ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
            )
        elif padding != "valid":
            raise ValueError("Padding must be 'same' or 'valid'.")

        col_input = self.im3col(input_data, (filter_d, filter_h, filter_w), stride)
        col_filters = filter_data.reshape(num_filters, -1)

        output = np.tensordot(col_filters, col_input, axes=([1], [1]))
        output = output.transpose(1, 0, 2).reshape(batch_size, num_filters, 
                    (input_d - filter_d) // stride + 2,
                    (input_h - filter_h) // stride + 2,
                    (input_w - filter_w) // stride + 2)
        from tensor import Tensor
        return Tensor(
            data=output,
            retain_grad=True,
            operation="Backward<Conv3D>",
            creator=[input_f, filters],
            meta={
                "input_shape": input_f.data.shape,
                "filter_shape": filter_data.shape,
                "stride": stride,
                "padding": padding,
            },
        )

    def backward(self, out_grad):
        input_f, filters = out_grad.creator
        input_data = input_f.data
        filter_data = filters.data
        stride = out_grad.meta["stride"]
        padding = out_grad.meta["padding"]

        batch_size, channels, input_d, input_h, input_w = input_data.shape
        num_filters, _, filter_d, filter_h, filter_w = filter_data.shape

        if padding == "same":
            pad_d = (filter_d - 1) // 2
            pad_h = (filter_h - 1) // 2
            pad_w = (filter_w - 1) // 2
            input_data = np.pad(
                input_data,
                ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
            )

        col_input = self.im3col(input_data, (filter_d, filter_h, filter_w), stride)
        col_out_grad = out_grad.data.reshape(batch_size, num_filters, -1)

        grad_filters = np.tensordot(col_out_grad, col_input, axes=([0, 2], [0, 2]))
        grad_filters = grad_filters.reshape(num_filters, channels, filter_d, filter_h, filter_w)

        col_filters = filter_data.reshape(num_filters, -1)
        grad_input_col = np.tensordot(col_filters.T, col_out_grad, axes=([1], [1]))
        grad_input_col = grad_input_col.transpose(1, 0, 2).reshape(input_data.shape)

        if padding == "same":
            grad_input_col = grad_input_col[:, :, pad_d:-pad_d, pad_h:-pad_h, pad_w:-pad_w]

        input_f.grad = grad_input_col if input_f.grad is None else input_f.grad + grad_input_col
        filters.grad = grad_filters if filters.grad is None else filters.grad + grad_filters


def conv3d_optimised(f, filters, stride=1, padding="same"):
    return Conv3DOptimized()(f, filters, stride, padding)
"""

========================================
          POOLING LAYERS NOTES (ACCORDING TO SEMESTER CLASS)
========================================
Convention Notation:
  - Input shapes:
      * 1D: (N, C, L)
      * 2D: (N, C, H, W)
      * 3D: (N, C, D, H, W)
  - Kernel size: (k), (kH, kW), (kD, kH, kW)
  - Stride: (s), (sH, sW), (sD, sH, sW)

Output Shape Formula:
  out_dim = floor((input_dim - kernel_dim) / stride_dim) + 1

----------------------------
MaxPool (1D / 2D / 3D)
----------------------------
Forward:
  y = max(x[window])
  - Picks max value in each kernel region.
  - Stores argmax index for backward routing.

Backward:
  - Gradient flows only to the max index.
  - All other locations receive 0.

----------------------------
AvgPool (1D / 2D / 3D)
----------------------------
Forward:
  y = mean(x[window])
  - Averages all values in the kernel region.

Backward:
  - Evenly distributes grad over the kernel:
    dx = grad / (kernel_volume)

----------------------------
GlobalAvgPool2D
----------------------------
Forward:
  y[n, c] = mean(x[n, c, :, :])
  - Output shape: (N, C)

Backward:
  dx = grad / (H * W)
  - Gradient is broadcast evenly across each HxW feature map.

----------------------------
Shape Summary:
----------------------------
Layer            Input Shape         → Output Shape
-----------------------------------------------------
MaxPool1D        (N, C, L)           → (N, C, (L - k) // s + 1)
MaxPool2D        (N, C, H, W)        → (N, C, H_out, W_out)
MaxPool3D        (N, C, D, H, W)     → (N, C, D_out, H_out, W_out)
AvgPool1D        (N, C, L)           → same   [ i have never used it do not know where it is used ]
AvgPool2D        (N, C, H, W)        → same
AvgPool3D        (N, C, D, H, W)     → same
GlobalAvgPool2D  (N, C, H, W)        → (N, C) [ never used in real world project do not know its benefit or when to use. ]

where:
  D_out = (D - kD) // sD + 1
  H_out = (H - kH) // sH + 1
  W_out = (W - kW) // sW + 1

"""

class MaxPool2D(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        x, kernel_size, stride = inputs
        x_data = x.data
        N, C, H, W = x_data.shape
        kH, kW = kernel_size
        sH, sW = stride

        out_H = (H - kH) // sH + 1
        out_W = (W - kW) // sW + 1

        out = np.zeros((N, C, out_H, out_W), dtype=x_data.dtype)
        max_indices = {}

        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * sH
                        h_end = h_start + kH
                        w_start = j * sW
                        w_end = w_start + kW

                        window = x_data[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        max_idx = np.unravel_index(np.argmax(window), window.shape)

                        out[n, c, i, j] = max_val
                        max_indices[(n, c, i, j)] = (h_start + max_idx[0], w_start + max_idx[1])

        return Tensor(
            data=out,
            retain_grad=True,
            operation="Backward<MaxPool2D>",
            creator=[x],
            meta={"kernel_size": kernel_size, "stride": stride, "max_indices": max_indices, "input_shape": x_data.shape},
        )

    def backward(self, out_grad):
        x = out_grad.creator[0]
        kernel_size = out_grad.meta["kernel_size"]
        stride = out_grad.meta["stride"]
        max_indices = out_grad.meta["max_indices"]
        N, C, H, W = out_grad.meta["input_shape"]

        dx = np.zeros((N, C, H, W), dtype=out_grad.grad.dtype)
        grad_out = out_grad.grad
        out_H, out_W = grad_out.shape[2], grad_out.shape[3]

        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_idx, w_idx = max_indices[(n, c, i, j)]
                        dx[n, c, h_idx, w_idx] += grad_out[n, c, i, j]

        grad = handle_broadcasting_and_reshape(x, dx)
        if x.grad is not None:
            x.grad += grad
        else:
            x.grad = grad

def maxpool2d(x ,kernel_size:tuple , stride : tuple ):
    return MaxPool2D()(x , kernel_size , stride)

class AvgPool2D(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        x, kernel_size, stride = inputs
        x_data = x.data
        N, C, H, W = x_data.shape
        kH, kW = kernel_size
        sH, sW = stride

        out_H = (H - kH) // sH + 1
        out_W = (W - kW) // sW + 1

        out = np.zeros((N, C, out_H, out_W), dtype=x_data.dtype)

        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * sH
                        h_end = h_start + kH
                        w_start = j * sW
                        w_end = w_start + kW

                        window = x_data[n, c, h_start:h_end, w_start:w_end]
                        out[n, c, i, j] = np.mean(window)

        return Tensor(
            data=out,
            retain_grad=True,
            operation="Backward<AvgPool2D>",
            creator=[x],
            meta={"kernel_size": kernel_size, "stride": stride, "input_shape": x_data.shape},
        )

    def backward(self, out_grad):
        x = out_grad.creator[0]
        kernel_size = out_grad.meta["kernel_size"]
        stride = out_grad.meta["stride"]
        N, C, H, W = out_grad.meta["input_shape"]
        kH, kW = kernel_size
        sH, sW = stride

        dx = np.zeros((N, C, H, W), dtype=out_grad.grad.dtype)
        grad_out = out_grad.grad
        out_H, out_W = grad_out.shape[2], grad_out.shape[3]
        pool_area = kH * kW

        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * sH
                        h_end = h_start + kH
                        w_start = j * sW
                        w_end = w_start + kW

                        # Distribute grad_out[n, c, i, j] evenly over the window
                        dx[n, c, h_start:h_end, w_start:w_end] += grad_out[n, c, i, j] / pool_area

        grad = handle_broadcasting_and_reshape(x, dx)
        if x.grad is not None:
            x.grad += grad
        else:
            x.grad = grad

def avgpool2d(x ,kernel_size:tuple , stride : tuple ):
    return AvgPool2D()(x , kernel_size , stride)

class MaxPool1D(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        x, kernel_size, stride = inputs
        x_data = x.data
        N, C, L = x_data.shape
        k, s = kernel_size, stride

        out_L = (L - k) // s + 1
        out = np.zeros((N, C, out_L), dtype=x_data.dtype)
        max_indices = {}

        for n in range(N):
            for c in range(C):
                for i in range(out_L):
                    start = i * s
                    end = start + k
                    window = x_data[n, c, start:end]
                    max_val = np.max(window)
                    max_idx = np.argmax(window)
                    out[n, c, i] = max_val
                    max_indices[(n, c, i)] = start + max_idx

        return Tensor(
            data=out,
            retain_grad=True,
            operation="Backward<MaxPool1D>",
            creator=[x],
            meta={"kernel_size": kernel_size, "stride": stride, "max_indices": max_indices, "input_shape": x_data.shape},
        )

    def backward(self, out_grad):
        x = out_grad.creator[0]
        k, s = out_grad.meta["kernel_size"], out_grad.meta["stride"]
        max_indices = out_grad.meta["max_indices"]
        N, C, L = out_grad.meta["input_shape"]

        dx = np.zeros((N, C, L), dtype=out_grad.grad.dtype)
        grad_out = out_grad.grad

        for n in range(N):
            for c in range(C):
                for i in range(grad_out.shape[2]):
                    idx = max_indices[(n, c, i)]
                    dx[n, c, idx] += grad_out[n, c, i]

        grad = handle_broadcasting_and_reshape(x, dx)
        if x.grad is not None:
            x.grad += grad
        else:
            x.grad = grad

def maxpool1d(x ,kernel_size:int , stride : int ):
    return MaxPool1D()(x , kernel_size , stride)

class AvgPool1D(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        x, kernel_size, stride = inputs
        x_data = x.data
        N, C, L = x_data.shape
        k, s = kernel_size, stride

        out_L = (L - k) // s + 1
        out = np.zeros((N, C, out_L), dtype=x_data.dtype)

        for n in range(N):
            for c in range(C):
                for i in range(out_L):
                    start = i * s
                    end = start + k
                    out[n, c, i] = np.mean(x_data[n, c, start:end])

        return Tensor(
            data=out,
            retain_grad=True,
            operation="Backward<AvgPool1D>",
            creator=[x],
            meta={"kernel_size": kernel_size, "stride": stride, "input_shape": x_data.shape},
        )

    def backward(self, out_grad):
        x = out_grad.creator[0]
        k, s = out_grad.meta["kernel_size"], out_grad.meta["stride"]
        N, C, L = out_grad.meta["input_shape"]

        dx = np.zeros((N, C, L), dtype=out_grad.grad.dtype)
        grad_out = out_grad.grad
        pool_area = k

        for n in range(N):
            for c in range(C):
                for i in range(grad_out.shape[2]):
                    start = i * s
                    end = start + k
                    dx[n, c, start:end] += grad_out[n, c, i] / pool_area

        grad = handle_broadcasting_and_reshape(x, dx)
        if x.grad is not None:
            x.grad += grad
        else:
            x.grad = grad

def avgpool1d(x ,kernel_size:int , stride : int ):
    return AvgPool1D()(x , kernel_size , stride)

class GlobalAvgPool2D(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        x = inputs[0]
        x_data = x.data
        N, C, H, W = x_data.shape

        out = np.mean(x_data, axis=(2, 3), keepdims=False)  # shape: (N, C)

        return Tensor(
            data=out,
            retain_grad=True,
            operation="Backward<GlobalAvgPool2D>",
            creator=[x],
            meta={"input_shape": x_data.shape},
        )

    def backward(self, out_grad):
        x = out_grad.creator[0]
        N, C, H, W = out_grad.meta["input_shape"]
        grad_out = out_grad.grad

        # Expand grad shape (N, C) -> (N, C, H, W) with equal distribution => similar to pytorch
        grad = grad_out[:, :, None, None] / (H * W)
        grad = np.broadcast_to(grad, (N, C, H, W))

        grad = handle_broadcasting_and_reshape(x, grad)
        if x.grad is not None:
            x.grad += grad
        else:
            x.grad = grad

def globalpool2d(x):
    return GlobalAvgPool2D()(x)

class MaxPool3D(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        x, kernel_size, stride = inputs
        x_data = x.data
        N, C, D, H, W = x_data.shape
        kD, kH, kW = kernel_size
        sD, sH, sW = stride

        out_D = (D - kD) // sD + 1
        out_H = (H - kH) // sH + 1
        out_W = (W - kW) // sW + 1

        out = np.zeros((N, C, out_D, out_H, out_W), dtype=x_data.dtype)
        max_indices = {}

        for n in range(N):
            for c in range(C):
                for d in range(out_D):
                    for i in range(out_H):
                        for j in range(out_W):
                            d_start = d * sD
                            h_start = i * sH
                            w_start = j * sW
                            d_end = d_start + kD
                            h_end = h_start + kH
                            w_end = w_start + kW

                            window = x_data[n, c, d_start:d_end, h_start:h_end, w_start:w_end]
                            max_val = np.max(window)
                            max_idx = np.unravel_index(np.argmax(window), window.shape)

                            out[n, c, d, i, j] = max_val
                            max_indices[(n, c, d, i, j)] = (
                                d_start + max_idx[0],
                                h_start + max_idx[1],
                                w_start + max_idx[2],
                            )

        return Tensor(
            data=out,
            retain_grad=True,
            operation="Backward<MaxPool3D>",
            creator=[x],
            meta={
                "kernel_size": kernel_size,
                "stride": stride,
                "max_indices": max_indices,
                "input_shape": x_data.shape,
            },
        )

    def backward(self, out_grad):
        x = out_grad.creator[0]
        max_indices = out_grad.meta["max_indices"]
        N, C, D, H, W = out_grad.meta["input_shape"]
        grad_out = out_grad.grad
        out_D, out_H, out_W = grad_out.shape[2:]

        dx = np.zeros((N, C, D, H, W), dtype=grad_out.dtype)

        for n in range(N):
            for c in range(C):
                for d in range(out_D):
                    for i in range(out_H):
                        for j in range(out_W):
                            d_idx, h_idx, w_idx = max_indices[(n, c, d, i, j)]
                            dx[n, c, d_idx, h_idx, w_idx] += grad_out[n, c, d, i, j]

        grad = handle_broadcasting_and_reshape(x, dx)
        if x.grad is not None:
            x.grad += grad
        else:
            x.grad = grad

def maxpool3d(x , kernel : tuple , stride : tuple):
    return MaxPool3D(x , kernel , stride)


class AvgPool3D(BaseOperationHandler):
    def forward(self, inputs):
        from tensor import Tensor

        x, kernel_size, stride = inputs
        x_data = x.data
        N, C, D, H, W = x_data.shape
        kD, kH, kW = kernel_size
        sD, sH, sW = stride

        out_D = (D - kD) // sD + 1
        out_H = (H - kH) // sH + 1
        out_W = (W - kW) // sW + 1

        out = np.zeros((N, C, out_D, out_H, out_W), dtype=x_data.dtype)

        for n in range(N):
            for c in range(C):
                for d in range(out_D):
                    for i in range(out_H):
                        for j in range(out_W):
                            d_start = d * sD
                            h_start = i * sH
                            w_start = j * sW
                            d_end = d_start + kD
                            h_end = h_start + kH
                            w_end = w_start + kW

                            window = x_data[n, c, d_start:d_end, h_start:h_end, w_start:w_end]
                            out[n, c, d, i, j] = np.mean(window)

        return Tensor(
            data=out,
            retain_grad=True,
            operation="Backward<AvgPool3D>",
            creator=[x],
            meta={
                "kernel_size": kernel_size,
                "stride": stride,
                "input_shape": x_data.shape,
            },
        )

    def backward(self, out_grad):
        x = out_grad.creator[0]
        kD, kH, kW = out_grad.meta["kernel_size"]
        sD, sH, sW = out_grad.meta["stride"]
        N, C, D, H, W = out_grad.meta["input_shape"]
        grad_out = out_grad.grad
        out_D, out_H, out_W = grad_out.shape[2:]
        pool_area = kD * kH * kW

        dx = np.zeros((N, C, D, H, W), dtype=grad_out.dtype)

        for n in range(N):
            for c in range(C):
                for d in range(out_D):
                    for i in range(out_H):
                        for j in range(out_W):
                            d_start = d * sD
                            h_start = i * sH
                            w_start = j * sW
                            d_end = d_start + kD
                            h_end = h_start + kH
                            w_end = w_start + kW

                            dx[n, c, d_start:d_end, h_start:h_end, w_start:w_end] += (
                                grad_out[n, c, d, i, j] / pool_area
                            )

        grad = handle_broadcasting_and_reshape(x, dx)
        if x.grad is not None:
            x.grad += grad
        else:
            x.grad = grad

def avgpool3d(x , kernel : tuple , stride : tuple):
    return AvgPool3D(x , kernel , stride)


class BatchNorm1D(BaseOperationHandler):

    def forward(self, inputs):
        x = inputs[0].data
        eps = 1e-8
        mean = x.mean(axis = 0)
        sigma = x.var(axis = 0)
        # normilizing data 
        x_new = (x - mean) / np.sqrt(sigma + eps)
        from tensor import Tensor
        return Tensor(
            data=x_new , 
            retain_grad= True , 
            operation="Backward<BatchNorm1D>", 
            creator=[x],
            meta={"eps" : eps , "mean" : mean , "var" : sigma}
        )

    def backward(self, out_grad):
        """
        x̂ = (x - μ) / sqrt(σ² + eps)
        """
        dxhat = out_grad.grad                    # ∂L/∂x̂, shape (N, C)
        x = out_grad.creator[0].data             # original input shape is (N, C)
        N, C = x.shape

        eps = out_grad.meta["eps"]
        mean = out_grad.meta["mean"]
        var = out_grad.meta["var"]
        std_inv = 1.0 / np.sqrt(var + eps)
        xmu = x - mean
        
        # ∂L/∂var
        dvar = np.sum(dxhat * xmu * -0.5 * std_inv**3, axis=0)  # shape (C,)
        
        # ∂L/∂mean
        dmean = np.sum(dxhat * -std_inv, axis=0) + dvar * np.mean(-2.0 * xmu, axis=0)
        
        # ∂L/∂x
        dx = (dxhat * std_inv) + (dvar * 2.0 * xmu / N) + (dmean / N)

        # Apply gradient to x
        handle = out_grad.creator[0]
        dxt = handle_broadcasting_and_reshape(handle , dx)
        if handle.grad is not None:
            handle.grad += dxt
        else:
            handle.grad = dxt

def batchnorm1d(x):
    return BatchNorm1D(x)

class BatchNorm2D(BaseOperationHandler):
    def forward(self, inputs):
        x = inputs[0].data
        eps = 1e-8
        mean = x.mean(axis=(0, 2, 3), keepdims=True)  # shape: (x, C, x, x)
        var = x.var(axis=(0, 2, 3), keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        from tensor import Tensor
        return Tensor(
            data=x_hat,
            retain_grad=True,
            operation="Backward<BatchNorm2D>",
            creator=[x],
            meta={"eps": eps, "mean": mean, "var": var}
        )

    def backward(self, out_grad):
        dxhat = out_grad.grad
        x = out_grad.creator[0].data
        N, C, H, W = x.shape

        eps = out_grad.meta["eps"]
        mean = out_grad.meta["mean"]
        var = out_grad.meta["var"]

        std_inv = 1.0 / np.sqrt(var + eps)
        xmu = x - mean

        dvar = np.sum(dxhat * xmu * -0.5 * std_inv**3, axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dxhat * -std_inv, axis=(0, 2, 3), keepdims=True) + \
                dvar * np.mean(-2.0 * xmu, axis=(0, 2, 3), keepdims=True)

        dx = (dxhat * std_inv) + (dvar * 2.0 * xmu / (N * H * W)) + (dmean / (N * H * W))

        handle = out_grad.creator[0]
        dxt = handle_broadcasting_and_reshape(handle, dx)
        if handle.grad is not None:
            handle.grad += dxt
        else:
            handle.grad = dxt


def batchnorm2d(x):
    return BatchNorm2D(x)

class BatchNorm3D(BaseOperationHandler):
    def forward(self, inputs):
        x = inputs[0].data
        eps = 1e-8
        mean = x.mean(axis=(0, 2, 3, 4), keepdims=True) 
        var = x.var(axis=(0, 2, 3, 4), keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)

        from tensor import Tensor
        return Tensor(
            data=x_hat,
            retain_grad=True,
            operation="Backward<BatchNorm3D>",
            creator=[x],
            meta={"eps": eps, "mean": mean, "var": var}
        )

    def backward(self, out_grad):
        dxhat = out_grad.grad
        x = out_grad.creator[0].data
        N, C, D, H, W = x.shape

        eps = out_grad.meta["eps"]
        mean = out_grad.meta["mean"]
        var = out_grad.meta["var"]

        std_inv = 1.0 / np.sqrt(var + eps)
        xmu = x - mean

        dvar = np.sum(dxhat * xmu * -0.5 * std_inv**3, axis=(0, 2, 3, 4), keepdims=True)
        dmean = np.sum(dxhat * -std_inv, axis=(0, 2, 3, 4), keepdims=True) + \
                dvar * np.mean(-2.0 * xmu, axis=(0, 2, 3, 4), keepdims=True)
        dx = (dxhat * std_inv) + (dvar * 2.0 * xmu / (N * D * H * W)) + \
             (dmean / (N * D * H * W))

        handle = out_grad.creator[0]
        dxt = handle_broadcasting_and_reshape(handle, dx)
        if handle.grad is not None:
            handle.grad += dxt
        else:
            handle.grad = dxt

def batchnorm3d(x):
    return BatchNorm3D(x)
