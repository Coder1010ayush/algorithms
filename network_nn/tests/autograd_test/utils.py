# ---------------------------------------------- utf-8 encoding ------------------------------------
# this file contains the test cases for view , reshape , permute , transpose
import torch


def view_test():
    A = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    return A.view(6, 4)


def reshape_test():
    A = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    return A.reshape(6, 4)


def permute_test():
    A = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    return A.permute(2, 0, 1)


def transpose_test():
    A = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
    return A.transpose(0, 1)


def edge_case_1():
    A = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    B = A.transpose(0, 1)
    return B.contiguous().view(6, 4)


def edge_case_2():
    A = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    return A.reshape(-1, 4)


def edge_case_3():
    A = torch.randn(2, 3, 4, 5, dtype=torch.float32, requires_grad=True)
    return A.permute(3, 2, 1, 0)


def edge_case_4():
    A = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    B = A.transpose(0, 1)
    return B.reshape(3, -1)


def edge_case_5():
    A = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    return A.view(-1)


def edge_case_6():
    A = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    return A.permute(1, 0, 2)


def edge_case_7():
    A = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    reshaped = A.reshape(2, 12)
    reshaped.retain_grad()
    return reshaped


def edge_case_8():
    A = torch.randn(4, 3, 2, dtype=torch.float32, requires_grad=True)
    permuted = A.permute(2, 1, 0)
    permuted.retain_grad()
    return permuted


def edge_case_9():
    A = torch.randn(3, 3, 3, dtype=torch.float32, requires_grad=True)
    permuted = A.permute(2, 1, 0)
    permuted.retain_grad()
    return permuted


def generalization_case_1():
    A = torch.randn(3, 4, 5, dtype=torch.float32, requires_grad=True)
    return A.view(4, -1)


def generalization_case_2():
    A = torch.randn(3, 4, 5, dtype=torch.float32, requires_grad=True)
    return A.reshape(3, 20)


def generalization_case_3():
    A = torch.randn(3, 4, 5, dtype=torch.float32, requires_grad=True)
    return A.permute(2, 1, 0)


def generalization_case_4():
    A = torch.randn(4, 5, 6, dtype=torch.float32, requires_grad=True)
    return A.reshape(5, -1)


def generalization_case_5():
    A = torch.randn(4, 5, 6, dtype=torch.float32, requires_grad=True)
    return A.view(6, -1)


def compute_gradient_g():
    test_cases = [
        view_test,
        reshape_test,
        permute_test,
        transpose_test,
        edge_case_1,
        edge_case_2,
        edge_case_3,
        edge_case_4,
        edge_case_5,
        edge_case_6,
        edge_case_7,
        edge_case_8,
        edge_case_9,
        generalization_case_1,
        generalization_case_2,
        generalization_case_3,
        generalization_case_4,
        generalization_case_5,
    ]

    for i, tensor_func in enumerate(test_cases, start=1):
        A = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
        Z = tensor_func()
        Z.retain_grad()
        out = Z.sum()
        out.retain_grad()
        out.backward()

        print(f"---------------------- Test Case {i} ----------------------")
        print(f"Gradient w.r.t A: {A.grad}")
        print(f"Gradient w.r.t Z: {Z.grad}")
        print(f"Gradient w.r.t out: {out.grad}")
        print("------------------------------------------------------------\n")


if __name__ == "__main__":
    compute_gradient_g()
