import numpy as np
import pytest

from network_nn import Tensor, functional as F

from tests.network_nn.helpers import check_gradients


@pytest.mark.parametrize(
    "op, shapes",
    [
        (F.add, [(3, 4), (3, 4)]),
        (F.add, [(3, 1), (3, 4)]),
        (F.add, [(4,), (2, 3, 4)]),
        (F.sub, [(3, 4), (4,)]),
        (F.mul, [(2, 3, 4), (3, 4)]),
        (F.mul, [(2, 1, 4), (1, 3, 1)]),
        (F.maximum, [(3, 4), (3, 4)]),
    ],
)
def test_binary_ops_with_broadcasting(op, shapes, rng):
    check_gradients(op, *[rng.standard_normal(s) for s in shapes])


def test_div(rng):
    a, b = rng.standard_normal((3, 4)), rng.uniform(0.5, 2.0, (4,))
    check_gradients(F.div, a, b)


@pytest.mark.parametrize("op", [F.exp, F.sin, F.cos, F.tan, F.abs, F.neg])
def test_unary_ops(op, rng):
    check_gradients(op, rng.uniform(-1.2, 1.2, (3, 4)))


@pytest.mark.parametrize("op", [F.log, F.sqrt])
def test_positive_domain_ops(op, rng):
    check_gradients(op, rng.uniform(0.5, 3.0, (3, 4)))


@pytest.mark.parametrize("p", [2, 3, 0.5, -1])
def test_pow(p, rng):
    check_gradients(lambda a: F.pow(a, p), rng.uniform(0.5, 2.0, (3, 4)))


def test_clip(rng):
    check_gradients(lambda a: F.clip(a, -0.5, 0.5), rng.standard_normal((5, 5)))


def test_masked_fill(rng):
    mask = rng.random((3, 4)) > 0.5
    x = rng.standard_normal((3, 4))
    check_gradients(lambda a: F.masked_fill(a, mask, 3.0), x)
    out = F.masked_fill(Tensor(x), mask, 7.0)
    assert np.all(out.data[mask] == 7.0) and np.all(out.data[~mask] == x[~mask])


def test_where(rng):
    cond = rng.random((3, 4)) > 0.5
    check_gradients(lambda a, b: F.where(cond, a, b), rng.standard_normal((3, 4)), rng.standard_normal((4,)))


def test_python_operators_and_scalars():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = ((2 * a + 1) / 4 - a**2) @ Tensor([1.0, 1.0, 1.0])
    out.backward()
    np.testing.assert_allclose(a.grad, 0.5 - 2 * a.data)


def test_diamond_graph_accumulates_once(rng):
    check_gradients(lambda a: a * a + a, rng.standard_normal((3, 3)))
    check_gradients(lambda a: F.matmul(a, a), rng.standard_normal((3, 3)))


def test_no_grad_context():
    from network_nn import is_grad_enabled, no_grad

    a = Tensor([1.0], requires_grad=True)
    with no_grad():
        assert not is_grad_enabled()
        Tensor([2.0])
        assert not is_grad_enabled()
        out = a * 2
    assert is_grad_enabled()
    assert not out.requires_grad and out.is_leaf
