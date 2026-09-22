from typing import Sequence

import numpy as np


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else out.reshape(())


class HiddenMarkovModel:
    """Discrete-emission HMM trained with Baum-Welch; inference in log space."""

    def __init__(
        self,
        n_states: int,
        n_symbols: int,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.initial_ = np.full(n_states, 1.0 / n_states)
        self.transition_ = self._random_stochastic((n_states, n_states))
        self.emission_ = self._random_stochastic((n_states, n_symbols))
        self.log_likelihood_history: list[float] = []

    def _random_stochastic(self, shape: tuple[int, int]) -> np.ndarray:
        m = self.rng.uniform(0.5, 1.5, size=shape)
        return m / m.sum(axis=1, keepdims=True)

    def set_params(self, initial: np.ndarray, transition: np.ndarray, emission: np.ndarray) -> "HiddenMarkovModel":
        self.initial_, self.transition_, self.emission_ = (np.asarray(a, dtype=float) for a in (initial, transition, emission))
        return self

    def _forward(self, seq: np.ndarray) -> np.ndarray:
        log_a, log_b = np.log(self.transition_), np.log(self.emission_)
        alpha = np.empty((len(seq), self.n_states))
        alpha[0] = np.log(self.initial_) + log_b[:, seq[0]]
        for t in range(1, len(seq)):
            alpha[t] = _logsumexp(alpha[t - 1][:, None] + log_a, axis=0) + log_b[:, seq[t]]
        return alpha

    def _backward(self, seq: np.ndarray) -> np.ndarray:
        log_a, log_b = np.log(self.transition_), np.log(self.emission_)
        beta = np.zeros((len(seq), self.n_states))
        for t in range(len(seq) - 2, -1, -1):
            beta[t] = _logsumexp(log_a + log_b[:, seq[t + 1]] + beta[t + 1], axis=1)
        return beta

    def score(self, sequence: np.ndarray) -> float:
        return float(_logsumexp(self._forward(np.asarray(sequence, dtype=int))[-1]))

    def viterbi(self, sequence: np.ndarray) -> np.ndarray:
        seq = np.asarray(sequence, dtype=int)
        log_a, log_b = np.log(self.transition_), np.log(self.emission_)
        delta = np.log(self.initial_) + log_b[:, seq[0]]
        back = np.zeros((len(seq), self.n_states), dtype=int)
        for t in range(1, len(seq)):
            cand = delta[:, None] + log_a
            back[t] = np.argmax(cand, axis=0)
            delta = cand[back[t], np.arange(self.n_states)] + log_b[:, seq[t]]
        path = np.empty(len(seq), dtype=int)
        path[-1] = int(np.argmax(delta))
        for t in range(len(seq) - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        return self.viterbi(sequence)

    def fit(self, sequences: Sequence[np.ndarray] | np.ndarray) -> "HiddenMarkovModel":
        seqs = [np.asarray(s, dtype=int) for s in ([sequences] if np.ndim(sequences[0]) == 0 else sequences)]
        self.log_likelihood_history = []
        prev = -np.inf
        for _ in range(self.n_iter):
            start = np.zeros(self.n_states)
            trans = np.zeros((self.n_states, self.n_states))
            emit = np.zeros((self.n_states, self.n_symbols))
            total = 0.0
            log_a, log_b = np.log(self.transition_), np.log(self.emission_)
            for seq in seqs:
                alpha, beta = self._forward(seq), self._backward(seq)
                ll = _logsumexp(alpha[-1])
                total += ll
                gamma = np.exp(alpha + beta - ll)
                xi = alpha[:-1, :, None] + log_a[None] + log_b[:, seq[1:]].T[:, None, :] + beta[1:, None, :] - ll
                start += gamma[0]
                trans += np.exp(xi).sum(axis=0)
                np.add.at(emit.T, seq, gamma)
            self.initial_ = start / start.sum()
            self.transition_ = trans / trans.sum(axis=1, keepdims=True)
            self.emission_ = (emit + 1e-12) / (emit + 1e-12).sum(axis=1, keepdims=True)
            self.log_likelihood_history.append(float(total))
            if total - prev < self.tol:
                break
            prev = total
        return self

    def sample(self, length: int, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        states, obs = np.empty(length, dtype=int), np.empty(length, dtype=int)
        states[0] = rng.choice(self.n_states, p=self.initial_)
        for t in range(length):
            if t:
                states[t] = rng.choice(self.n_states, p=self.transition_[states[t - 1]])
            obs[t] = rng.choice(self.n_symbols, p=self.emission_[states[t]])
        return states, obs
