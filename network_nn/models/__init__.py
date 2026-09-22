"""Ready-made model architectures built from `network_nn.nn`."""

from network_nn.models.autoencoder import Autoencoder, VariationalAutoencoder, vae_loss
from network_nn.models.gan import GAN

__all__ = ["Autoencoder", "VariationalAutoencoder", "vae_loss", "GAN"]
