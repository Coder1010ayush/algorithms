# ----------------------------- utf-8 encoding --------------------------
# this file contains how pytorch calculate grad for addition for same shape tensor and also broadcastable shape
import torch


def type_a():
    A = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    B = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    return A, B


def type_b():
    A = torch.randn(3, 1, dtype=torch.float32, requires_grad=True)
    B = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    return A, B


def type_c():
    A = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    B = torch.randn(4, dtype=torch.float32, requires_grad=True)
    return A, B


def type_d():
    A = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    B = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
    return A, B


def type_e():
    A = torch.randn(1, 3, 4, dtype=torch.float32, requires_grad=True)
    B = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    return A, B


def type_f():
    A = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    B = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    return A, B


def compute_gradient():
    for i, tensor_func in enumerate(
        [type_a, type_b, type_c, type_d, type_e, type_f], start=1
    ):
        a, b = tensor_func()
        z = a + b
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
