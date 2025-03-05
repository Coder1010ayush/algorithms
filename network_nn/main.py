import numpy as np
import torch


def matmul_forward(A: np.ndarray, B: np.ndarray):
    # Handle vector-vector outer product case
    if A.ndim == 1 and B.ndim == 1:
        A = A[:, np.newaxis]  # Shape: (n, 1)
        B = B[np.newaxis, :]  # Shape: (1, m)
    # Handle single vector case
    elif A.ndim == 1:
        A = A[np.newaxis, :]

    C = np.matmul(A, B)
    cache = (A, B)
    return C, cache


def matmul_backward(grad_output: np.ndarray, cache: tuple):
    A, B = cache
    A_shape_orig = A.shape
    B_shape_orig = B.shape
    grad_shape = grad_output.shape

    # Ensure A is at least 2D
    if len(A_shape_orig) == 1:
        grad_output = grad_output[np.newaxis, :]

    # Handle broadcasting in batch dimensions
    A_batch_dims = A.shape[:-2] if len(A.shape) > 2 else ()
    B_batch_dims = B.shape[:-2] if len(B.shape) > 2 else ()
    grad_batch_dims = grad_output.shape[:-2] if len(grad_output.shape) > 2 else ()

    # Calculate output dimensions for broadcasting
    out_batch_dims = np.broadcast_shapes(A_batch_dims, B_batch_dims, grad_batch_dims)

    # Reshape inputs for broadcasting if needed
    if A.shape[:-2] != out_batch_dims:
        A = np.broadcast_to(A, out_batch_dims + A.shape[-2:])
    if B.shape[:-2] != out_batch_dims:
        B = np.broadcast_to(B, out_batch_dims + B.shape[-2:])
    if grad_output.shape[:-2] != out_batch_dims:
        grad_output = np.broadcast_to(grad_output, out_batch_dims + grad_output.shape[-2:])

    # Calculate gradients with broadcasting support
    grad_A = np.matmul(grad_output, B.swapaxes(-1, -2))
    grad_B = np.matmul(A.swapaxes(-1, -2), grad_output)

    # Sum over broadcast dimensions if necessary
    if A.shape[:-2] != A_batch_dims:
        reduction_axes = tuple(range(len(out_batch_dims) - len(A_batch_dims)))
        grad_A = grad_A.sum(axis=reduction_axes)
    if B.shape[:-2] != B_batch_dims:
        reduction_axes = tuple(range(len(out_batch_dims) - len(B_batch_dims)))
        grad_B = grad_B.sum(axis=reduction_axes)

    # Restore original shapes for vectors
    if len(A_shape_orig) == 1:
        grad_A = grad_A.squeeze(0)

    return grad_A, grad_B


def verify_with_pytorch():

    test_cases = [
        # 1. Basic matrix multiplication (2x2 @ 2x2)
        (np.array([[1., 2.], [3., 4.]]),
         np.array([[5., 6.], [7., 8.]])),

        # 2. Broadcasting with batch dimension (2x1x2 @ 2x2)
        (np.array([[[1., 2.]], [[3., 4.]]]),
         np.array([[5., 6.], [7., 8.]])),

        # 3. Vector-matrix broadcasting (2 @ 2x2)
        (np.array([1., 2.]),
         np.array([[5., 6.], [7., 8.]])),

        # 4. Batch matrix multiplication (2x2x2 @ 2x2x2)
        (np.array([[[1., 2.], [3., 4.]],
                   [[5., 6.], [7., 8.]]]),
         np.array([[[9., 10.], [11., 12.]],
                  [[13., 14.], [15., 16.]]]),
         ),

        # 5. Complex broadcasting (1x2 @ 2x2x2)
        (np.array([[1., 2.]]),
         np.array([[[5., 6.],
                   [7., 8.]],
                   [[9., 10.],
                    [11., 12.]]])),

        # 6. Different input/output dimensions (3x2 @ 2x4)
        (np.array([[1., 2.],
                   [3., 4.],
                   [5., 6.]]),
         np.array([[1., 2., 3., 4.],
                  [5., 6., 7., 8.]])),

        # 7. Single batch with different dimensions (1x3x2 @ 1x2x4)
        (np.array([[[1., 2.],
                    [3., 4.],
                    [5., 6.]]]),
         np.array([[[1., 2., 3., 4.],
                   [5., 6., 7., 8.]]])),

        # 8. Vector-vector outer product (3 @ 4)
        (np.array([1., 2., 3.]),
         np.array([4., 5., 6.])),

        # 9. Multiple batch dimensions (2x3x2x2 @ 2x3x2x2)
        (np.array([[[[1., 2.], [3., 4.]],
                    [[5., 6.], [7., 8.]],
                    [[9., 10.], [11., 12.]]],
                   [[[13., 14.], [15., 16.]],
                    [[17., 18.], [19., 20.]],
                    [[21., 22.], [23., 24.]]]]),
         np.array([[[[1., 2.], [3., 4.]],
                   [[5., 6.], [7., 8.]],
                   [[9., 10.], [11., 12.]]],
                  [[[13., 14.], [15., 16.]],
                   [[17., 18.], [19., 20.]],
                   [[21., 22.], [23., 24.]]]]),
         )
    ]

    for i, (A_np, B_np) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"A shape: {A_np.shape}")
        print(f"B shape: {B_np.shape}")

        # NumPy implementation
        C_np, cache = matmul_forward(A_np, B_np)
        grad_output = np.ones_like(C_np)
        grad_A_np, grad_B_np = matmul_backward(grad_output, cache)

        # PyTorch implementation
        A_torch = torch.tensor(A_np, requires_grad=True)
        B_torch = torch.tensor(B_np, requires_grad=True)
        C_torch = torch.matmul(A_torch, B_torch)
        C_torch.backward(torch.ones_like(C_torch))

        print("\nResults match?")
        print("Forward pass:", np.allclose(C_np, C_torch.detach().numpy()))
        print("grad_A:", np.allclose(grad_A_np, A_torch.grad.numpy()))
        print("grad_B:", np.allclose(grad_B_np, B_torch.grad.numpy()))
        print("\nShapes:")
        print(f"Output - NumPy: {C_np.shape}, PyTorch: {C_torch.shape}")
        print(f"grad_A - NumPy: {grad_A_np.shape}, PyTorch: {A_torch.grad.shape}")
        print(f"grad_B - NumPy: {grad_B_np.shape}, PyTorch: {B_torch.grad.shape}")


if __name__ == "__main__":
    # verify_with_pytorch()
    from tests.simple_tensor import define_tensor
    define_tensor()
