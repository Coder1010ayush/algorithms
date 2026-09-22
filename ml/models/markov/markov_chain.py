from typing import Sequence

import numpy as np


class MarkovChain:
    """First-order Markov chain over integer states 0..n_states-1 estimated by counting."""

    def __init__(self, n_states: int | None = None, smoothing: float = 1.0):
        self.n_states = n_states
        self.smoothing = smoothing
        self.transition_: np.ndarray | None = None
        self.initial_: np.ndarray | None = None

    def fit(self, sequences: Sequence[np.ndarray] | np.ndarray) -> "MarkovChain":
        seqs = [np.asarray(s, dtype=int) for s in ([sequences] if np.ndim(sequences[0]) == 0 else sequences)]
        k = self.n_states or int(max(s.max() for s in seqs)) + 1
        self.n_states = k
        counts = np.full((k, k), self.smoothing)
        start = np.full(k, self.smoothing)
        for s in seqs:
            start[s[0]] += 1
            np.add.at(counts, (s[:-1], s[1:]), 1)
        self.transition_ = counts / counts.sum(axis=1, keepdims=True)
        self.initial_ = start / start.sum()
        return self

    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """Distribution over the next state given the last state of ``sequence``."""
        return self.transition_[int(np.asarray(sequence)[-1])]

    def predict(self, sequence: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(sequence)))

    def score(self, sequence: np.ndarray) -> float:
        s = np.asarray(sequence, dtype=int)
        return float(np.log(self.initial_[s[0]]) + np.log(self.transition_[s[:-1], s[1:]]).sum())

    def sample(self, length: int, random_state: int | None = None, start: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        out = np.empty(length, dtype=int)
        out[0] = rng.choice(self.n_states, p=self.initial_) if start is None else start
        for t in range(1, length):
            out[t] = rng.choice(self.n_states, p=self.transition_[out[t - 1]])
        return out

    def stationary_distribution(self) -> np.ndarray:
        values, vectors = np.linalg.eig(self.transition_.T)
        pi = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
        pi = np.abs(pi)
        return pi / pi.sum()
