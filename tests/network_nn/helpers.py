"""Finite-difference gradient checking for autograd ops."""

from typing import Callable, Sequence

import numpy as np

from network_nn import Tensor


def numerical_gradient(fn: Callable[[], float], arrays: Sequence[np.ndarray], eps: float = 1e-6):
    grads = []
    for a in arrays:
        g = np.zeros_like(a)
        it = np.nditer(a, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old = a[idx]
            a[idx] = old + eps
            f_plus = fn()
            a[idx] = old - eps
            f_minus = fn()
            a[idx] = old
            g[idx] = (f_plus - f_minus) / (2 * eps)
            it.iternext()
        grads.append(g)
    return grads


def check_gradients(build: Callable[..., Tensor], *arrays: np.ndarray, rtol: float = 1e-5, atol: float = 1e-6, weight=None):
    """`build(*tensors)` returns a Tensor. The scalar objective is sum(out * weight) with a random weight
    so that non-trivial gradients flow through every output element."""
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    rng = np.random.default_rng(123)

    def objective_from(out: Tensor):
        nonlocal weight
        if weight is None:
            weight = rng.standard_normal(out.shape)
        return (out * Tensor(weight)).sum() if out.ndim > 0 else out * float(weight)

    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    obj = objective_from(build(*tensors))
    obj.backward()
    analytic = [t.grad for t in tensors]

    def scalar():
        out = build(*[Tensor(a) for a in arrays])
        return float(objective_from(out).data)

    numeric = numerical_gradient(scalar, arrays)
    for i, (a, n) in enumerate(zip(analytic, numeric)):
        assert a is not None, f"argument {i}: no gradient"
        assert a.shape == n.shape, f"argument {i}: shape {a.shape} != {n.shape}"
        np.testing.assert_allclose(a, n, rtol=rtol, atol=atol, err_msg=f"argument {i}")
