# ---------------------------------------------- utf-8 encoding ------------------------------------
# this file contains the test cases for convulational
import torch
import torch.nn.functional as F


def conv2d_type_a():
    A = torch.randn(1, 1, 5, 5, dtype=torch.float32, requires_grad=True)
    B = torch.randn(1, 1, 3, 3, dtype=torch.float32, requires_grad=True)
    return A, B


def conv2d_type_b():
    A = torch.randn(1, 3, 7, 7, dtype=torch.float32, requires_grad=True)
    B = torch.randn(2, 3, 5, 5, dtype=torch.float32, requires_grad=True)
    return A, B


def conv1d_type_a():
    A = torch.randn(1, 1, 10, dtype=torch.float32, requires_grad=True)
    B = torch.randn(1, 1, 3, dtype=torch.float32, requires_grad=True)
    return A, B


def conv1d_type_b():
    A = torch.randn(1, 3, 10, dtype=torch.float32, requires_grad=True)
    B = torch.randn(2, 3, 5, dtype=torch.float32, requires_grad=True)
    return A, B


def conv3d_type_a():
    A = torch.randn(1, 1, 5, 5, 5, dtype=torch.float32, requires_grad=True)
    B = torch.randn(1, 1, 3, 3, 3, dtype=torch.float32, requires_grad=True)
    return A, B


def conv3d_type_b():
    A = torch.randn(1, 3, 7, 7, 7, dtype=torch.float32, requires_grad=True)
    B = torch.randn(2, 3, 5, 5, 5, dtype=torch.float32, requires_grad=True)
    return A, B


def compute_gradient():
    test_cases = [
        (conv2d_type_a, F.conv2d),
        (conv2d_type_b, F.conv2d),
        (conv1d_type_a, F.conv1d),
        (conv1d_type_b, F.conv1d),
        (conv3d_type_a, F.conv3d),
        (conv3d_type_b, F.conv3d),
    ]

    for i, (tensor_func, conv_func) in enumerate(test_cases, start=1):
        a, b = tensor_func()
        z = conv_func(a, b, padding=1)
        z.retain_grad()
        out = z.sum()
        out.retain_grad()
        out.backward()

        print(f"---------------------- Test Case {i} ----------------------")
        print(f"Gradient w.r.t a: {a.grad}")
        print(f"Gradient w.r.t b: {b.grad}")
        print(f"Gradient w.r.t z: {z.grad}")
        print(f"Gradient w.r.t out: {out.grad}")
        print("------------------------------------------------------------\n")


if __name__ == "__main__":
    compute_gradient()
