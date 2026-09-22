import numpy as np

from ml.base import BaseModel
from ml.utils.distance import pairwise


def _conditional_probabilities(D2: np.ndarray, perplexity: float, tol: float = 1e-5, max_tries: int = 50) -> np.ndarray:
    """Per-row Gaussian kernels with precision found by binary search to hit the target perplexity."""
    n = len(D2)
    target = np.log(perplexity)
    P = np.zeros((n, n))
    for i in range(n):
        d = np.delete(D2[i], i)
        beta, lo, hi = 1.0, 0.0, np.inf
        for _ in range(max_tries):
            p = np.exp(-(d - d.min()) * beta)
            s = p.sum()
            p /= s
            entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)))
            diff = entropy - target
            if abs(diff) < tol:
                break
            if diff > 0:
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo == 0 else (beta + lo) / 2
        P[i, np.arange(n) != i] = p
    return P


class TSNE(BaseModel):
    """t-distributed stochastic neighbour embedding, trained by momentum gradient descent."""

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.rng = np.random.default_rng(random_state)
        self.embedding_: np.ndarray | None = None
        self.kl_divergence_: float | None = None

    def fit(self, X: np.ndarray, y=None) -> "TSNE":
        X = np.asarray(X, dtype=float)
        n = len(X)
        if self.perplexity >= n:
            raise ValueError("perplexity must be smaller than the number of samples")
        P = _conditional_probabilities(pairwise(X, X) ** 2, self.perplexity)
        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)

        Y = self.rng.normal(0.0, 1e-4, (n, self.n_components))
        velocity = np.zeros_like(Y)
        gains = np.ones_like(Y)
        exaggeration_steps = min(250, self.n_iter // 4)
        off_diag = ~np.eye(n, dtype=bool)
        for it in range(self.n_iter):
            P_eff = P * self.early_exaggeration if it < exaggeration_steps else P
            num = 1.0 / (1.0 + pairwise(Y, Y) ** 2)
            num[~off_diag] = 0.0
            Q = np.maximum(num / num.sum(), 1e-12)
            PQ = (P_eff - Q) * num
            grad = 4.0 * ((np.diag(PQ.sum(axis=1)) - PQ) @ Y)
            momentum = 0.5 if it < exaggeration_steps else 0.8
            same_sign = np.sign(grad) == np.sign(velocity)
            gains = np.where(same_sign, gains * 0.8, gains + 0.2).clip(0.01)
            velocity = momentum * velocity - self.learning_rate * gains * grad
            Y += velocity
            Y -= Y.mean(axis=0)
        self.embedding_ = Y
        self.kl_divergence_ = float(np.sum(P[off_diag] * np.log(P[off_diag] / Q[off_diag])))
        return self

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X).embedding_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """t-SNE has no out-of-sample map; returns the training embedding."""
        if self.embedding_ is None:
            raise RuntimeError("call fit first")
        return self.embedding_
