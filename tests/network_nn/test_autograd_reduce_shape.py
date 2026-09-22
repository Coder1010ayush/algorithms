import numpy as np
import pytest

from network_nn import Tensor, functional as F

from tests.network_nn.helpers import check_gradients


@pytest.mark.parametrize("op", [F.sum, F.mean, F.var, F.std, F.max, F.min])
@pytest.mark.parametrize("axis, keepdims", [(None, False), (0, False), (1, True), ((0, 2), False), (-1, False)])
def test_reductions(op, axis, keepdims, rng):
    x = rng.standard_normal((2, 3, 4))
    check_gradients(lambda a: op(a, axis, keepdims), x)


def test_matmul_shapes(rng):
    cases = [((2, 3), (3, 4)), ((2, 2, 3), (3, 4)), ((2, 3), (3,)), ((3,), (3, 4)), ((3,), (3,)), ((2, 1, 3, 4), (5, 4, 2)), ((2, 1, 3), (3, 4))]
    for sa, sb in cases:
        check_gradients(F.matmul, rng.standard_normal(sa), rng.standard_normal(sb))


def test_reshape_permute_squeeze(rng):
    x = rng.standard_normal((2, 3, 4))
    check_gradients(lambda a: F.reshape(a, (4, 6)), x)
    check_gradients(lambda a: a.reshape(-1, 4), x)
    check_gradients(lambda a: F.permute(a, (2, 0, 1)), x)
    check_gradients(lambda a: a.transpose(0, 2), x)
    check_gradients(lambda a: a.T, x)
    check_gradients(lambda a: F.unsqueeze(a, 1), x)
    check_gradients(lambda a: F.squeeze(a, 1), rng.standard_normal((2, 1, 4)))
    check_gradients(lambda a: a.flatten(1), x)


def test_stack_concat_pad_broadcast(rng):
    a, b = rng.standard_normal((3, 4)), rng.standard_normal((3, 4))
    check_gradients(lambda x, y: F.stack([x, y], 0), a, b)
    check_gradients(lambda x, y: F.stack([x, y], 1), a, b)
    check_gradients(lambda x, y: F.concat([x, y], 1), a, rng.standard_normal((3, 2)))
    check_gradients(lambda x: F.pad(x, ((1, 1), (0, 2))), a)
    check_gradients(lambda x: F.broadcast_to(x, (2, 3, 4)), a)


def test_getitem_variants(rng):
    x = rng.standard_normal((2, 3, 4))
    check_gradients(lambda a: a[:, 1, :], x)
    check_gradients(lambda a: a[0], x)
    check_gradients(lambda a: a[..., 1:3], x)
    check_gradients(lambda a: a[[0, 1, 1], [2, 0, 0]], x)


def test_setitem_on_leaf_only():
    t = Tensor(np.zeros(3))
    t[1] = 5.0
    assert t.data[1] == 5.0
    non_leaf = Tensor(np.zeros(3), requires_grad=True) * 2
    with pytest.raises(RuntimeError):
        non_leaf[0] = 1.0
