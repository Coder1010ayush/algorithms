import numpy as np
import pytest

from network_nn import Tensor, functional as F

from tests.network_nn.helpers import check_gradients


@pytest.mark.parametrize("op", [F.relu, F.sigmoid, F.tanh, F.swish, F.gelu, lambda a: F.leaky_relu(a, 0.1)])
def test_activations(op, rng):
    check_gradients(op, rng.standard_normal((4, 5)) + 0.01)


@pytest.mark.parametrize("axis", [-1, 0])
def test_softmax_and_log_softmax(axis, rng):
    x = rng.standard_normal((3, 5))
    check_gradients(lambda a: F.softmax(a, axis), x)
    check_gradients(lambda a: F.log_softmax(a, axis), x)
    np.testing.assert_allclose(F.softmax(Tensor(x), axis).data.sum(axis=axis), 1.0)


def test_losses(rng):
    pred, target = rng.standard_normal((6, 3)), rng.standard_normal((6, 3))
    for reduction in ("mean", "sum"):
        check_gradients(lambda p, t: F.mse_loss(p, t, reduction), pred, target)
        check_gradients(lambda p, t: F.huber_loss(p, t, 0.5, reduction), pred, target)
    prob, binary = rng.uniform(0.1, 0.9, (6, 1)), rng.integers(0, 2, (6, 1)).astype(float)
    check_gradients(lambda p: F.binary_cross_entropy(p, binary), prob)
    check_gradients(lambda z: F.binary_cross_entropy_with_logits(z, binary), pred[:, :1])
    labels = rng.integers(0, 3, 6)
    check_gradients(lambda z: F.cross_entropy(z, labels), pred)
    expected = -np.mean(np.log(F.softmax(Tensor(pred)).data[np.arange(6), labels]))
    np.testing.assert_allclose(F.cross_entropy(Tensor(pred), labels).data, expected)


def test_normalize_layer_and_batch(rng):
    x = rng.standard_normal((4, 3, 5)) * 3 + 1
    check_gradients(lambda a: F.layer_norm(a), x)
    check_gradients(lambda a: F.batch_norm(a), x)
    out = F.batch_norm(Tensor(x)).data
    np.testing.assert_allclose(out.mean(axis=(0, 2)), 0.0, atol=1e-7)
    np.testing.assert_allclose(out.var(axis=(0, 2)), 1.0, atol=1e-3)


@pytest.mark.parametrize(
    "x_shape, w_shape, stride, padding",
    [
        ((2, 3, 10), (4, 3, 3), 1, "valid"),
        ((2, 3, 10), (4, 3, 3), 2, "same"),
        ((1, 2, 6, 6), (3, 2, 3, 3), 1, "valid"),
        ((1, 2, 7, 7), (3, 2, 3, 3), 2, "same"),
        ((1, 2, 6, 5), (2, 2, 2, 3), (2, 1), 1),
        ((1, 1, 5, 5, 5), (2, 1, 3, 3, 3), 1, "valid"),
        ((1, 2, 6, 6, 6), (2, 2, 3, 3, 3), 2, "same"),
    ],
)
def test_conv_gradients(x_shape, w_shape, stride, padding, rng):
    check_gradients(lambda x, w: F.conv2d(x, w, stride, padding), rng.standard_normal(x_shape), rng.standard_normal(w_shape))


def test_conv2d_matches_direct_loops(rng):
    x, w = rng.standard_normal((1, 2, 5, 5)), rng.standard_normal((3, 2, 3, 3))
    out = F.conv2d(Tensor(x), Tensor(w)).data
    expected = np.zeros((1, 3, 3, 3))
    for f in range(3):
        for i in range(3):
            for j in range(3):
                expected[0, f, i, j] = np.sum(x[0, :, i : i + 3, j : j + 3] * w[f])
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    "op, x_shape, kernel, stride",
    [
        (F.max_pool1d, (2, 3, 8), 2, None),
        (F.avg_pool1d, (2, 3, 9), 3, 2),
        (F.max_pool2d, (1, 2, 6, 6), 2, None),
        (F.max_pool2d, (1, 2, 7, 7), 3, 2),
        (F.avg_pool2d, (1, 2, 6, 6), (2, 3), (2, 1)),
        (F.max_pool3d, (1, 1, 4, 4, 4), 2, None),
        (F.avg_pool3d, (1, 1, 5, 4, 4), 2, 2),
    ],
)
def test_pool_gradients(op, x_shape, kernel, stride, rng):
    check_gradients(lambda x: op(x, kernel, stride), rng.standard_normal(x_shape))


def test_global_avg_pool(rng):
    x = rng.standard_normal((2, 3, 4, 4))
    check_gradients(F.global_avg_pool, x)
    np.testing.assert_allclose(F.global_avg_pool(Tensor(x)).data, x.mean(axis=(2, 3)))
