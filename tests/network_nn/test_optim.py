import numpy as np
import pytest

from network_nn import Parameter, Tensor, optim


def _rosenbrock_step(opt_cls, steps=2000, **kwargs):
    p = Parameter(np.array([-1.0, 1.0]))
    opt = opt_cls([p], **kwargs)
    for _ in range(steps):
        opt.zero_grad()
        x, y = p[0], p[1]
        loss = (1 - x) ** 2 + 100 * (y - x**2) ** 2
        loss.backward()
        opt.step()
    return p.data, float(loss.data)


@pytest.mark.parametrize(
    "opt_cls, kwargs, max_fraction",
    [
        (optim.SGD, dict(lr=1e-3, momentum=0.9), 0.05),
        (optim.SGD, dict(lr=1e-3, momentum=0.9, nesterov=True), 0.05),
        (optim.Adam, dict(lr=0.02), 0.05),
        (optim.AdamW, dict(lr=0.02, weight_decay=1e-4), 0.05),
        (optim.Nadam, dict(lr=0.02), 0.05),
        (optim.RMSprop, dict(lr=0.005), 0.05),
        (optim.Adagrad, dict(lr=0.5), 0.05),
        (optim.Adadelta, dict(lr=1.0), 0.25),  # Adadelta is deliberately conservative on this valley
    ],
)
def test_optimizers_minimise_rosenbrock(opt_cls, kwargs, max_fraction):
    start = (1 - (-1.0)) ** 2 + 100 * (1 - 1) ** 2
    _, final = _rosenbrock_step(opt_cls, **kwargs)
    assert final < start * max_fraction


def test_optimizer_skips_params_without_grad():
    a, b = Parameter(np.ones(2)), Parameter(np.ones(2))
    opt = optim.SGD([a, b], lr=0.1)
    (a * 2).sum().backward()
    opt.step()
    np.testing.assert_allclose(a.data, 0.8)
    np.testing.assert_allclose(b.data, 1.0)


def test_schedulers_follow_expected_curves():
    p = Parameter(np.zeros(1))
    opt = optim.SGD([p], lr=1.0)
    sched = optim.StepLR(opt, step_size=2, gamma=0.5)
    assert [sched.step() for _ in range(4)] == [1.0, 0.5, 0.5, 0.25]

    opt.lr = 1.0
    cos = optim.CosineAnnealingLR(opt, total_steps=4)
    values = [cos.step() for _ in range(4)]
    assert values[-1] == pytest.approx(0.0) and values[1] == pytest.approx(0.5)

    opt.lr = 1.0
    lin = optim.LinearLR(opt, total_steps=4, final_lr=0.0)
    assert [lin.step() for _ in range(4)] == pytest.approx([0.75, 0.5, 0.25, 0.0])

    opt.lr = 1.0
    warm = optim.WarmupLR(opt, warmup_steps=2, after=optim.ExponentialLR(opt, gamma=0.5))
    assert [warm.step() for _ in range(4)] == pytest.approx([0.5, 1.0, 0.5, 0.25])

    opt.lr = 1.0
    plateau = optim.ReduceLROnPlateau(opt, factor=0.1, patience=1)
    for metric in [1.0, 1.0, 1.0]:
        plateau.step(metric)
    assert opt.lr == pytest.approx(0.1)
