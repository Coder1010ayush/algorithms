import numpy as np
import pytest

from network_nn import Tensor, models, nn, optim


def _train(model, loss_fn, x, steps, lr=1e-2):
    opt = optim.Adam(model.parameters(), lr=lr)
    first = None
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(x)
        loss.backward()
        opt.step()
        first = float(loss.data) if first is None else first
    return first, float(loss.data)


def test_autoencoder_compresses_low_rank_data(rng):
    nn.init.seed(0)
    basis = rng.standard_normal((3, 20))
    x = Tensor(rng.standard_normal((128, 3)) @ basis)
    ae = models.Autoencoder(20, hidden=(16, 3))
    first, last = _train(ae, ae.reconstruction_loss, x, steps=300)
    assert ae.encode(x).shape == (128, 3)
    assert last < first * 0.1


def test_vae_trains_and_samples(rng):
    nn.init.seed(0)
    x = Tensor((rng.random((96, 12)) > 0.5).astype(float))
    vae = models.VariationalAutoencoder(12, hidden=(24,), latent_dim=4, rng=rng)

    def loss_fn(batch):
        recon, mu, log_var = vae(batch)
        return models.vae_loss(recon, batch, mu, log_var, reconstruction="bce")

    first, last = _train(vae, loss_fn, x, steps=150)
    assert last < first
    vae.eval()
    recon, mu, _ = vae(x)
    assert recon.shape == x.shape and mu.shape == (96, 4)
    assert np.all((recon.data >= 0) & (recon.data <= 1))
    assert vae.sample(5).shape == (5, 12)


def test_gan_learns_a_shifted_gaussian(rng):
    nn.init.seed(0)
    real_np = rng.normal(loc=4.0, scale=0.5, size=(256, 1))
    gan = models.GAN(data_dim=1, latent_dim=4, generator_hidden=(16,), discriminator_hidden=(16,), rng=rng)
    opt_g = optim.Adam(gan.generator.parameters(), lr=5e-3)
    opt_d = optim.Adam(gan.discriminator.parameters(), lr=5e-3)
    assert abs(gan.sample(256).data.mean() - 4.0) > 1.5
    for _ in range(300):
        stats = gan.train_step(Tensor(real_np), opt_g, opt_d)
    assert np.isfinite(stats["d_loss"]) and np.isfinite(stats["g_loss"])
    samples = gan.sample(512).data
    assert abs(samples.mean() - 4.0) < 1.0
    assert 0.2 < samples.std() < 1.5
    assert abs(stats["d_loss"] - 2 * np.log(2)) < 0.4  # discriminator near the 50/50 equilibrium
    assert gan.sample(7).shape == (7, 1)
