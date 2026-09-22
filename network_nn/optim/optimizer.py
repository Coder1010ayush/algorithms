from typing import Dict, Iterable, List

import numpy as np

from network_nn.tensor import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor], lr: float, weight_decay: float = 0.0):
        self.params: List[Tensor] = list(params)
        if not self.params:
            raise ValueError("Optimizer received an empty parameter list.")
        self.lr, self.weight_decay = lr, weight_decay
        self.state: Dict[int, dict] = {}
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
            p.data -= self._update(self.state.setdefault(i, {}), p.data, grad)

    def _update(self, state: dict, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Return the step to subtract from `param`."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lr={self.lr})"


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum, self.nesterov = momentum, nesterov

    def _update(self, state, param, grad):
        if not self.momentum:
            return self.lr * grad
        v = state.get("v", np.zeros_like(param))
        v = self.momentum * v + grad
        state["v"] = v
        return self.lr * (grad + self.momentum * v if self.nesterov else v)


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.b1, self.b2 = betas
        self.eps = eps

    def _moments(self, state, param, grad):
        m = self.b1 * state.get("m", np.zeros_like(param)) + (1 - self.b1) * grad
        v = self.b2 * state.get("v", np.zeros_like(param)) + (1 - self.b2) * grad * grad
        state["m"], state["v"] = m, v
        return m / (1 - self.b1**self.t), v / (1 - self.b2**self.t)

    def _update(self, state, param, grad):
        m_hat, v_hat = self._moments(state, param, grad)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay."""

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            p.data -= self.lr * self.weight_decay * p.data
            p.data -= self._update(self.state.setdefault(i, {}), p.data, p.grad)


class Nadam(Adam):
    def _update(self, state, param, grad):
        m_hat, v_hat = self._moments(state, param, grad)
        nesterov = self.b1 * m_hat + (1 - self.b1) * grad / (1 - self.b1**self.t)
        return self.lr * nesterov / (np.sqrt(v_hat) + self.eps)


class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.alpha, self.eps = alpha, eps

    def _update(self, state, param, grad):
        s = self.alpha * state.get("s", np.zeros_like(param)) + (1 - self.alpha) * grad * grad
        state["s"] = s
        return self.lr * grad / (np.sqrt(s) + self.eps)


class Adagrad(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.eps = eps

    def _update(self, state, param, grad):
        s = state.get("s", np.zeros_like(param)) + grad * grad
        state["s"] = s
        return self.lr * grad / (np.sqrt(s) + self.eps)


class Adadelta(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.rho, self.eps = rho, eps

    def _update(self, state, param, grad):
        eg = self.rho * state.get("eg", np.zeros_like(param)) + (1 - self.rho) * grad * grad
        edx = state.get("edx", np.zeros_like(param))
        delta = np.sqrt(edx + self.eps) / np.sqrt(eg + self.eps) * grad
        state["eg"] = eg
        state["edx"] = self.rho * edx + (1 - self.rho) * delta * delta
        return self.lr * delta
