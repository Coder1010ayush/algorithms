# -------------------------------------- utf-8 encoding -------------------------------
# this file contains the test cases for forward and backward propogation for addition element-wise for matrix with broadcastable options also
# ----------------------------- utf-8 encoding --------------------------
# this file contains how pytorch calculate grad for addition for same shape tensor and also broadcastable shape
import numpy as np
from tensor import Tensor
from initializer_w import Initializer

initializer = Initializer()


def create_tensor(shape, init_type="normal", retain_grad=True, meta=None):
    if meta is None:
        meta = {"loc": 0, "scale": 1}
    return initializer.forward(shape, init_type, retain_grad, meta)


def type_a():
    A = create_tensor(shape=(3, 4))
    B = create_tensor(shape=(3, 4))
    return A, B


def type_b():
    A = create_tensor(shape=(3, 1))
    B = create_tensor(shape=(3, 4))
    return A, B


def type_c():
    A = create_tensor(shape=(3, 4))
    B = create_tensor(shape=(4))
    return A, B


def type_d():
    A = create_tensor(shape=(3, 4))
    B = create_tensor(shape=(1))
    return A, B


def type_e():
    A = create_tensor(shape=(1, 3, 4))
    B = create_tensor(shape=(3, 4))
    return A, B


def type_f():
    A = create_tensor(shape=(2, 3, 4))
    B = create_tensor(shape=(3, 4))
    return A, B


def compute_gradient_add():
    for i, tensor_func in enumerate(
        [type_a, type_b, type_c, type_d, type_e, type_f], start=1
    ):
        print(f"---------------------- Test Case {i} ----------------------")
        a, b = tensor_func()
        z = a + b
        out = z.sum()
        out.backprop()

        print(f"Gradient w.r.t a: {a.grad}")
        print(f"Gradient w.r.t b: {b.grad}")
        print(f"Gradient w.r.t z: {z.grad}")
        print(f"Gradient w.r.t out: {out.grad}")
        print("------------------------------------------------------------\n")


if __name__ == "__main__":
    compute_gradient_add()
