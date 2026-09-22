from typing import List, Optional, Tuple

import numpy as np

from network_nn.autograd import functional as F
from network_nn.nn import init
from network_nn.nn.layers import Linear
from network_nn.nn.module import Module, ModuleList
from network_nn.tensor import Tensor


def _zeros(batch: int, hidden: int) -> Tensor:
    return Tensor(np.zeros((batch, hidden)))


def _uniform_reset(module: Module, hidden_size: int) -> None:
    bound = 1.0 / np.sqrt(hidden_size)
    for p in module.parameters():
        init.uniform_(p, -bound, bound)


class RNNCell(Module):
    """h' = act(x W_ih + b_ih + h W_hh + b_hh)"""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh"):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.act = {"tanh": F.tanh, "relu": F.relu}[nonlinearity]
        self.ih = Linear(input_size, hidden_size, bias)
        self.hh = Linear(hidden_size, hidden_size, bias)
        _uniform_reset(self, hidden_size)

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        h = _zeros(x.shape[0], self.hidden_size) if h is None else h
        return self.act(self.ih(x) + self.hh(h))


class GRUCell(Module):
    """Standard GRU: reset and update gates, candidate uses the reset-scaled hidden projection."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.ih = Linear(input_size, 3 * hidden_size, bias)
        self.hh = Linear(hidden_size, 3 * hidden_size, bias)
        _uniform_reset(self, hidden_size)

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
        _uniform_reset(self, hidden_size)

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
        return o * F.tanh(c_new), c_new


class _StackedRecurrent(Module):
    """Runs a cell over (batch, seq, feature) input, layer by layer, optionally in both directions.

    Final state is shaped (num_layers * num_directions, batch, hidden) like PyTorch; the
    output of a bidirectional layer is the concatenation [forward, backward] on the last axis.
    """

    cell_cls = None

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, bidirectional: bool = False, **cell_kwargs):
        super().__init__()
        self.input_size, self.hidden_size, self.num_layers = input_size, hidden_size, num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        cells = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size * self.num_directions
            cells += [self.cell_cls(in_size, hidden_size, bias, **cell_kwargs) for _ in range(self.num_directions)]
        self.cells = ModuleList(cells)

    @property
    def output_size(self) -> int:
        return self.hidden_size * self.num_directions

    def _initial_state(self, batch: int, state) -> List:
        raise NotImplementedError

    def _stack_state(self, states: List):
        raise NotImplementedError

    @staticmethod
    def _hidden(state) -> Tensor:
        return state[0] if isinstance(state, tuple) else state

    def _run(self, cell: Module, x: Tensor, state, reverse: bool):
        seq_len = x.shape[1]
        outputs: List[Optional[Tensor]] = [None] * seq_len
        for t in (range(seq_len - 1, -1, -1) if reverse else range(seq_len)):
            state = cell(x[:, t, :], state)
            outputs[t] = self._hidden(state)
        return F.stack(outputs, axis=1), state

    def forward(self, x: Tensor, state=None):
        states = self._initial_state(x.shape[0], state)
        final = []
        for layer in range(self.num_layers):
            outs = []
            for d in range(self.num_directions):
                idx = layer * self.num_directions + d
                out, s = self._run(self.cells[idx], x, states[idx], reverse=(d == 1))
                outs.append(out)
                final.append(s)
            x = outs[0] if len(outs) == 1 else F.concat(outs, axis=2)
        return x, self._stack_state(final)

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, bidirectional={self.bidirectional}"


class RNN(_StackedRecurrent):
    cell_cls = RNNCell

    def _initial_state(self, batch, state):
        n = self.num_layers * self.num_directions
        return [_zeros(batch, self.hidden_size) for _ in range(n)] if state is None else [state[i] for i in range(n)]

    def _stack_state(self, states):
        return F.stack(states, axis=0)


class GRU(RNN):
    cell_cls = GRUCell


class LSTM(_StackedRecurrent):
    cell_cls = LSTMCell

    def _initial_state(self, batch, state):
        n = self.num_layers * self.num_directions
        if state is None:
            return [(_zeros(batch, self.hidden_size), _zeros(batch, self.hidden_size)) for _ in range(n)]
        h0, c0 = state
        return [(h0[i], c0[i]) for i in range(n)]

    def _stack_state(self, states):
        return F.stack([h for h, _ in states], axis=0), F.stack([c for _, c in states], axis=0)
