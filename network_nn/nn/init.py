"""In-place weight initialisers. Each takes a Tensor and fills `.data`, returning the tensor."""

from typing import Optional, Tuple

import numpy as np

from network_nn.tensor import Tensor

_rng = np.random.default_rng()


def seed(value: Optional[int]) -> None:
    global _rng
    _rng = np.random.default_rng(value)


def fan_in_out(shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(shape) < 2:
        return shape[0], shape[0]
    receptive = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    return shape[1] * receptive, shape[0] * receptive


def _gain(nonlinearity: str, a: float = 0.0) -> float:
    return {
        "linear": 1.0,
        "sigmoid": 1.0,
        "tanh": 5.0 / 3.0,
        "relu": np.sqrt(2.0),
        "leaky_relu": np.sqrt(2.0 / (1.0 + a**2)),
    }[nonlinearity]


def constant_(t: Tensor, value: float) -> Tensor:
    t.data[...] = value
    return t


def zeros_(t: Tensor) -> Tensor:
    return constant_(t, 0.0)


def ones_(t: Tensor) -> Tensor:
    return constant_(t, 1.0)


def uniform_(t: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    t.data[...] = _rng.uniform(a, b, size=t.shape)
    return t


def normal_(t: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    t.data[...] = _rng.normal(mean, std, size=t.shape)
    return t


def xavier_uniform_(t: Tensor, gain: float = 1.0) -> Tensor:
    fi, fo = fan_in_out(t.shape)
    bound = gain * np.sqrt(6.0 / (fi + fo))
    return uniform_(t, -bound, bound)


def xavier_normal_(t: Tensor, gain: float = 1.0) -> Tensor:
    fi, fo = fan_in_out(t.shape)
    return normal_(t, 0.0, gain * np.sqrt(2.0 / (fi + fo)))


def kaiming_uniform_(t: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    fi, fo = fan_in_out(t.shape)
    fan = fi if mode == "fan_in" else fo
    bound = _gain(nonlinearity, a) * np.sqrt(3.0 / fan)
    return uniform_(t, -bound, bound)


def kaiming_normal_(t: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    fi, fo = fan_in_out(t.shape)
    fan = fi if mode == "fan_in" else fo
    return normal_(t, 0.0, _gain(nonlinearity, a) / np.sqrt(fan))


def lecun_uniform_(t: Tensor) -> Tensor:
    fi, _ = fan_in_out(t.shape)
    bound = np.sqrt(3.0 / fi)
    return uniform_(t, -bound, bound)


def lecun_normal_(t: Tensor) -> Tensor:
    fi, _ = fan_in_out(t.shape)
    return normal_(t, 0.0, 1.0 / np.sqrt(fi))


def orthogonal_(t: Tensor, gain: float = 1.0) -> Tensor:
    rows, cols = t.shape[0], int(np.prod(t.shape[1:]))
    flat = _rng.normal(size=(rows, cols))
    q, r = np.linalg.qr(flat if rows >= cols else flat.T)
    q = q * np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    t.data[...] = gain * q.reshape(t.shape)
    return t
