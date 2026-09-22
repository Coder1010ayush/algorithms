from typing import Optional, Sequence, Tuple

import numpy as np

from network_nn.autograd import functional as F
from network_nn.nn.layers import Linear, ReLU, Sigmoid
from network_nn.nn.module import Module, Sequential
from network_nn.tensor import Tensor


def _mlp(sizes: Sequence[int], final_activation: Optional[Module] = None) -> Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(ReLU())
    if final_activation is not None:
        layers.append(final_activation)
    return Sequential(*layers)


class Autoencoder(Module):
    """Fully connected encoder/decoder pair. `hidden` lists the encoder widths ending in the latent size."""

    def __init__(self, input_dim: int, hidden: Sequence[int] = (128, 32), output_activation: Optional[Module] = None):
        super().__init__()
        sizes = [input_dim, *hidden]
        self.encoder = _mlp(sizes)
        self.decoder = _mlp(sizes[::-1], output_activation)
        self.latent_dim = sizes[-1]

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def reconstruction_loss(self, x: Tensor) -> Tensor:
        return F.mse_loss(self(x), x)


class VariationalAutoencoder(Module):
    """Gaussian-prior VAE. `forward` returns (reconstruction, mu, log_var)."""

    def __init__(self, input_dim: int, hidden: Sequence[int] = (128,), latent_dim: int = 8, output_activation: Optional[Module] = Sigmoid(), rng=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.rng = rng or np.random.default_rng()
        self.encoder = _mlp([input_dim, *hidden], ReLU())
        self.mu = Linear(hidden[-1], latent_dim)
        self.log_var = Linear(hidden[-1], latent_dim)
        self.decoder = _mlp([latent_dim, *hidden[::-1], input_dim], output_activation)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        if not self.training:
            return mu
        eps = Tensor(self.rng.standard_normal(mu.shape))
        return mu + F.exp(0.5 * log_var) * eps

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, log_var = self.encode(x)
        return self.decode(self.reparameterize(mu, log_var)), mu, log_var

    def sample(self, n: int) -> Tensor:
        return self.decode(Tensor(self.rng.standard_normal((n, self.latent_dim))))


def vae_loss(recon: Tensor, x: Tensor, mu: Tensor, log_var: Tensor, beta: float = 1.0, reconstruction: str = "mse") -> Tensor:
    """Per-sample reconstruction term plus beta * KL(q(z|x) || N(0, I)), averaged over the batch."""
    if reconstruction == "bce":
        rec = F.binary_cross_entropy(recon, x, "sum")
    else:
        rec = F.mse_loss(recon, x, "sum")
    kl = -0.5 * F.sum(1.0 + log_var - mu * mu - F.exp(log_var))
    return (rec + beta * kl) / x.shape[0]
