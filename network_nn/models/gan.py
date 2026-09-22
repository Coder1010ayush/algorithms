from typing import Dict, Optional, Sequence

import numpy as np

from network_nn.autograd import functional as F
from network_nn.nn.layers import LeakyReLU, Linear, ReLU
from network_nn.nn.module import Module, Sequential
from network_nn.optim.optimizer import Optimizer
from network_nn.tensor import Tensor, no_grad


def _mlp(sizes: Sequence[int], act: Module, final: Optional[Module]) -> Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act)
    if final is not None:
        layers.append(final)
    return Sequential(*layers)


class GAN(Module):
    """Vanilla GAN with the non-saturating generator loss. The discriminator outputs logits."""

    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 16,
        generator_hidden: Sequence[int] = (64, 64),
        discriminator_hidden: Sequence[int] = (64, 64),
        generator: Optional[Module] = None,
        discriminator: Optional[Module] = None,
        rng=None,
    ):
        super().__init__()
        self.latent_dim, self.data_dim = latent_dim, data_dim
        self.rng = rng or np.random.default_rng()
        self.generator = generator or _mlp([latent_dim, *generator_hidden, data_dim], ReLU(), None)
        self.discriminator = discriminator or _mlp([data_dim, *discriminator_hidden, 1], LeakyReLU(0.2), None)

    def noise(self, n: int) -> Tensor:
        return Tensor(self.rng.standard_normal((n, self.latent_dim)))

    def forward(self, z: Tensor) -> Tensor:
        return self.generator(z)

    def sample(self, n: int) -> Tensor:
        with no_grad():
            return self.generator(self.noise(n))

    def discriminator_loss(self, real: Tensor, fake: Tensor) -> Tensor:
        real_logits, fake_logits = self.discriminator(real), self.discriminator(fake)
        ones, zeros = Tensor(np.ones(real_logits.shape)), Tensor(np.zeros(fake_logits.shape))
        return F.binary_cross_entropy_with_logits(real_logits, ones) + F.binary_cross_entropy_with_logits(fake_logits, zeros)

    def generator_loss(self, fake: Tensor) -> Tensor:
        fake_logits = self.discriminator(fake)
        return F.binary_cross_entropy_with_logits(fake_logits, Tensor(np.ones(fake_logits.shape)))

    def train_step(self, real: Tensor, opt_g: Optimizer, opt_d: Optimizer, d_steps: int = 1) -> Dict[str, float]:
        n = real.shape[0]
        for _ in range(d_steps):
            opt_d.zero_grad()
            fake = self.generator(self.noise(n)).detach()
            d_loss = self.discriminator_loss(real, fake)
            d_loss.backward()
            opt_d.step()

        opt_g.zero_grad()
        g_loss = self.generator_loss(self.generator(self.noise(n)))
        g_loss.backward()
        opt_g.step()
        return {"d_loss": float(d_loss.data), "g_loss": float(g_loss.data)}
