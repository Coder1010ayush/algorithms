# ---------------------------------------- *utf-8 encoding* ---------------------------------------
# this file contains test cases for the conv function
import os
from tensor import Tensor
from initializer_w import Initializer
from autograd.autodiff import conv1d, conv2d, conv3d

initializer = Initializer()


def create_tensor(shape, init_type="normal", retain_grad=True, meta=None):
    if meta is None:
        meta = {"loc": 0, "scale": 1}
    return initializer.forward(shape, init_type, retain_grad, meta)


def conv2d_type_a():
    A = create_tensor(shape=(1, 1, 5, 5))
    B = create_tensor(shape=(1, 1, 3, 3))
    return A, B


def conv2d_type_b():
    A = create_tensor(shape=(1, 3, 7, 7))
    B = create_tensor(shape=(2, 3, 5, 5))
    return A, B


def conv1d_type_a():
    A = create_tensor(shape=(1, 1, 10))
    B = create_tensor(shape=(1, 1, 3))
    return A, B


def conv1d_type_b():
    A = create_tensor(shape=(1, 3, 10))
    B = create_tensor(shape=(2, 3, 5))
    return A, B


def conv3d_type_a():
    A = create_tensor(shape=(1, 1, 5, 5, 5))
    B = create_tensor(shape=(1, 1, 3, 3, 3))
    return A, B


def conv3d_type_b():
    A = create_tensor(shape=(1, 3, 7, 7, 7))
    B = create_tensor(shape=(2, 3, 5, 5, 5))
    return A, B


def compute_gradient_conv():
    test_cases = [
        (conv2d_type_a, "conv2d"),
        (conv2d_type_b, "conv2d"),
        (conv1d_type_a, "conv1d"),
        (conv1d_type_b, "conv1d"),
        (conv3d_type_a, "conv3d"),
        (conv3d_type_b, "conv3d"),
    ]

    for i, (tensor_func, conv_func) in enumerate(test_cases, start=1):
        a, b = tensor_func()
        if conv_func == "conv2d":
            z = conv2d(a, b, stride=1)
        elif conv_func == "conv1d":
            z = conv1d(a, b, stride=1)
        elif conv_func == "conv3d":
            z = conv3d(a, b, stride=2)

        out = z.sum()
        out.backprop()

        print(f"---------------------- Test Case {i} ----------------------")
        print(f"Gradient w.r.t a: {a.grad}")
        print(f"Gradient w.r.t b: {b.grad}")
        print(f"Gradient w.r.t z: {z.grad}")
        print(f"Gradient w.r.t out: {out.grad}")
        print("------------------------------------------------------------\n")


if __name__ == "__main__":
    compute_gradient_conv()
