from typing import List, Optional, Tuple

import numpy as np

from network_nn.autograd import functional as F
from network_nn.nn import init
from network_nn.nn.layers import Linear
from network_nn.nn.module import Module, ModuleList
from network_nn.tensor import Tensor


def _zeros(batch: int, hidden: int) -> Tensor:
    return Tensor(np.zeros((batch, hidden)))


class RNNCell(Module):
    """h' = act(x W_ih + b_ih + h W_hh + b_hh)"""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh"):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.act = {"tanh": F.tanh, "relu": F.relu}[nonlinearity]
        self.ih = Linear(input_size, hidden_size, bias)
        self.hh = Linear(hidden_size, hidden_size, bias)
        self._reset(hidden_size)

    def _reset(self, hidden_size: int) -> None:
        bound = 1.0 / np.sqrt(hidden_size)
        for p in self.parameters():
            init.uniform_(p, -bound, bound)

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        h = _zeros(x.shape[0], self.hidden_size) if h is None else h
        return self.act(self.ih(x) + self.hh(h))


class GRUCell(Module):
    """Standard GRU: r, z gates and candidate n with reset applied to the hidden projection."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.ih = Linear(input_size, 3 * hidden_size, bias)
        self.hh = Linear(hidden_size, 3 * hidden_size, bias)
        bound = 1.0 / np.sqrt(hidden_size)
        for p in self.parameters():
            init.uniform_(p, -bound, bound)

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        h = _zeros(x.shape[0], self.hidden_size) if h is None else h
        H = self.hidden_size
        gi, gh = self.ih(x), self.hh(h)
        r = F.sigmoid(gi[:, :H] + gh[:, :H])
        z = F.sigmoid(gi[:, H : 2 * H] + gh[:, H : 2 * H])
        n = F.tanh(gi[:, 2 * H :] + r * gh[:, 2 * H :])
        return (1.0 - z) * n + z * h


class LSTMCell(Module):
    """Standard LSTM with input, forget, cell and output gates. Returns (h, c)."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.ih = Linear(input_size, 4 * hidden_size, bias)
        self.hh = Linear(hidden_size, 4 * hidden_size, bias)
        bound = 1.0 / np.sqrt(hidden_size)
        for p in self.parameters():
            init.uniform_(p, -bound, bound)

    def forward(self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        if state is None:
            state = (_zeros(x.shape[0], self.hidden_size), _zeros(x.shape[0], self.hidden_size))
        h, c = state
        H = self.hidden_size
        gates = self.ih(x) + self.hh(h)
        i = F.sigmoid(gates[:, :H])
        f = F.sigmoid(gates[:, H : 2 * H])
        g = F.tanh(gates[:, 2 * H : 3 * H])
        o = F.sigmoid(gates[:, 3 * H :])
        c_new = f * c + i * g
        h_new = o * F.tanh(c_new)
        return h_new, c_new


class _StackedRecurrent(Module):
    """Runs a cell over a (batch, seq, feature) input for `num_layers` stacked layers."""

    cell_cls = None

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, **cell_kwargs):
        super().__init__()
        self.input_size, self.hidden_size, self.num_layers = input_size, hidden_size, num_layers
        self.cells = ModuleList(
            [self.cell_cls(input_size if i == 0 else hidden_size, hidden_size, bias, **cell_kwargs) for i in range(num_layers)]
        )

    def _initial_state(self, batch: int, state):
        raise NotImplementedError

    def _step(self, cell, x_t, state):
        raise NotImplementedError

    def _stack_state(self, states):
        raise NotImplementedError

    def forward(self, x: Tensor, state=None):
        batch, seq_len, _ = x.shape
        states: List = self._initial_state(batch, state)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer, cell in enumerate(self.cells):
                states[layer] = self._step(cell, x_t, states[layer])
                x_t = states[layer] if not isinstance(states[layer], tuple) else states[layer][0]
            outputs.append(x_t)
        return F.stack(outputs, axis=1), self._stack_state(states)


class RNN(_StackedRecurrent):
    cell_cls = RNNCell

    def _initial_state(self, batch, state):
        return [_zeros(batch, self.hidden_size) for _ in range(self.num_layers)] if state is None else [state[i] for i in range(self.num_layers)]

    def _step(self, cell, x_t, h):
        return cell(x_t, h)

    def _stack_state(self, states):
        return F.stack(states, axis=0)


class GRU(RNN):
    cell_cls = GRUCell


class LSTM(_StackedRecurrent):
    cell_cls = LSTMCell

    def _initial_state(self, batch, state):
        if state is None:
            return [(_zeros(batch, self.hidden_size), _zeros(batch, self.hidden_size)) for _ in range(self.num_layers)]
        h0, c0 = state
        return [(h0[i], c0[i]) for i in range(self.num_layers)]

    def _step(self, cell, x_t, hc):
        return cell(x_t, hc)

    def _stack_state(self, states):
        return F.stack([h for h, _ in states], axis=0), F.stack([c for _, c in states], axis=0)
