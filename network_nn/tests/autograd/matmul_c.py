# ---------------------------------------------- utf-8 encoding ------------------------------------
# This file contains the test cases for matrix multiplication
import numpy as np
from tensor import Tensor
from initializer_w import Initializer

initializer = Initializer()


def create_tensor(shape, init_type="normal", retain_grad=True, meta=None):
    if meta is None:
        meta = {"loc": 0, "scale": 1}
    return initializer.forward(shape, init_type, retain_grad, meta)


def type_a():
    A = create_tensor((2, 3))
    B = create_tensor((3, 4))
    return A, B


def type_b():
    A = create_tensor((2, 1, 3))
    B = create_tensor((3, 4))
    return A, B


def type_c():
    A = create_tensor((2, 3))
    B = create_tensor((3, 4))
    return A, B


def type_d():
    A = create_tensor((2, 3))
    B = create_tensor((3,))
    return A, B


def type_e():
    A = create_tensor((1, 2, 3))
    B = create_tensor((1, 3, 4))
    return A, B


def type_f():
    A = create_tensor((2, 2, 3))
    B = create_tensor((3, 4))
    return A, B


def type_g():
    A = create_tensor((2, 3, 4))
    B = create_tensor((4, 5))
    return A, B


def type_h():
    A = create_tensor((3, 3))
    B = Tensor(np.eye(3), retain_grad=True)
    return A, B


def type_i():
    A = create_tensor((2, 3))
    B = create_tensor((3, 1))
    return A, B


def compute_gradient_matmul():
    for i, tensor_func in enumerate(
        [type_a, type_b, type_c, type_d, type_e, type_f, type_g, type_h, type_i],
        start=1,
    ):
        print(f"---------------------- Test Case {i} ----------------------")
        a, b = tensor_func()
        z = a.matmul(b)
        print(z)
        out = z.sum()
        print(out)
        out.backprop()

        print(f"Gradient w.r.t a: {a.grad}")
        print(f"Gradient w.r.t b: {b.grad}")
        print(f"Gradient w.r.t z: {z.grad}")
        print(f"Gradient w.r.t out: {out.grad}")
        print("------------------------------------------------------------\n")


if __name__ == "__main__":
    compute_gradient_matmul()
