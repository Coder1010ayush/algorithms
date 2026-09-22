import numpy as np
import pytest

from network_nn import Tensor, nn, optim, functional as F

from tests.network_nn.helpers import check_gradients


def _fit(model, x, y, loss_fn, steps=300, lr=0.05):
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return float(loss.data)


def test_linear_fits_line():
    x = np.linspace(-1, 1, 50).reshape(-1, 1)
    model = nn.Linear(1, 1)
    loss = _fit(model, Tensor(x), Tensor(3 * x + 2), F.mse_loss)
    assert loss < 1e-4
    np.testing.assert_allclose(model.weight.data.ravel(), [3.0], atol=1e-2)
    np.testing.assert_allclose(model.bias.data, [2.0], atol=1e-2)


def test_mlp_learns_xor(rng):
    x = rng.standard_normal((128, 2))
    y = ((x[:, 0] * x[:, 1]) > 0).astype(np.int64)
    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
    _fit(model, Tensor(x), y, F.cross_entropy, steps=400)
    pred = model(Tensor(x)).data.argmax(axis=1)
    assert (pred == y).mean() > 0.95


def test_module_registration_and_state_dict():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(2, 3)
            self.blocks = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 1)])
            self.bn = nn.BatchNorm1d(3)

        def forward(self, x):
            x = self.bn(self.a(x))
            for b in self.blocks:
                x = b(x)
            return x

    net = Net()
    names = [n for n, _ in net.named_parameters()]
    assert names == ["a.weight", "a.bias", "blocks.0.weight", "blocks.0.bias", "blocks.1.weight", "blocks.1.bias", "bn.weight", "bn.bias"]
    assert "bn.running_mean" in net.state_dict()
    state = {k: v.copy() for k, v in net.state_dict().items()}
    for p in net.parameters():
        p.data[...] = 0
    net.load_state_dict(state)
    np.testing.assert_array_equal(net.a.weight.data, state["a.weight"].data)
    net.eval()
    assert not net.bn.training
    net.zero_grad()


def test_embedding_and_dropout(rng):
    emb = nn.Embedding(10, 4)
    out = emb(np.array([[1, 2], [3, 1]]))
    assert out.shape == (2, 2, 4)
    out.sum().backward()
    assert emb.weight.grad[1].sum() == pytest.approx(2 * 4)
    drop = nn.Dropout(0.5, rng=rng)
    x = Tensor(np.ones((1000,)))
    kept = (drop(x).data != 0).mean()
    assert 0.4 < kept < 0.6
    drop.eval()
    np.testing.assert_array_equal(drop(x).data, x.data)


def test_layernorm_and_batchnorm_modules(rng):
    ln = nn.LayerNorm(6)
    x = rng.standard_normal((4, 6)) * 5
    check_gradients(lambda a: ln(a), x)
    bn = nn.BatchNorm2d(3)
    x4 = rng.standard_normal((2, 3, 4, 4)) * 2 + 1
    out = bn(Tensor(x4)).data
    np.testing.assert_allclose(out.mean(axis=(0, 2, 3)), 0.0, atol=1e-6)
    bn.eval()
    assert bn(Tensor(x4)).shape == x4.shape
    assert not np.allclose(bn.running_mean.data, 0.0)


@pytest.mark.parametrize("layer, shape", [(nn.Conv1d(3, 4, 3, stride=2, padding="same"), (2, 3, 16)), (nn.Conv2d(3, 2, 3), (1, 3, 8, 8)), (nn.Conv3d(1, 2, 3, stride=2, padding=1), (1, 1, 8, 8, 8))])
def test_conv_layers_forward_and_bias_grad(layer, shape, rng):
    out = layer(Tensor(rng.standard_normal(shape)))
    assert out.shape[:2] == (shape[0], layer.out_channels)
    out.sum().backward()
    assert layer.bias.grad is not None and layer.weight.grad is not None
    np.testing.assert_allclose(layer.bias.grad, np.prod(out.shape) / layer.out_channels * np.ones(layer.out_channels))


def test_cnn_trains_on_simple_task(rng):
    x = rng.standard_normal((32, 1, 8, 8))
    y = (x[:, 0, :4, :].sum(axis=(1, 2)) > x[:, 0, 4:, :].sum(axis=(1, 2))).astype(np.int64)
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding="same"), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(4 * 16, 2))
    _fit(model, Tensor(x), y, F.cross_entropy, steps=150, lr=0.01)
    assert (model(Tensor(x)).data.argmax(1) == y).mean() > 0.9


@pytest.mark.parametrize("cls", [nn.RNN, nn.GRU, nn.LSTM])
def test_recurrent_layers_forward_backward(cls, rng):
    layer = cls(input_size=5, hidden_size=7, num_layers=2)
    x = Tensor(rng.standard_normal((3, 6, 5)), requires_grad=True)
    out, state = layer(x)
    assert out.shape == (3, 6, 7)
    if cls is nn.LSTM:
        assert state[0].shape == (2, 3, 7) and state[1].shape == (2, 3, 7)
    else:
        assert state.shape == (2, 3, 7)
    out.sum().backward()
    assert x.grad.shape == x.shape
    assert all(p.grad is not None for p in layer.parameters())
    assert len(list(layer.parameters())) == 8


def test_lstm_cell_state_propagates(rng):
    cell = nn.LSTMCell(3, 4)
    x = Tensor(rng.standard_normal((2, 3)))
    h, c = cell(x)
    h2, c2 = cell(x, (h, c))
    assert not np.allclose(c.data, c2.data)


def test_rnn_learns_to_sum_sequence(rng):
    x = rng.standard_normal((64, 5, 1))
    y = x.sum(axis=1)
    model = nn.Sequential(nn.RNN(1, 8), nn.Identity())

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(1, 8)
            self.out = nn.Linear(8, 1)

        def forward(self, x):
            seq, _ = self.rnn(x)
            return self.out(seq[:, -1, :])

    loss = _fit(Head(), Tensor(x), Tensor(y), F.mse_loss, steps=300, lr=0.02)
    assert loss < 0.1


def test_multihead_attention_and_transformer(rng):
    mha = nn.MultiheadAttention(8, 2)
    x = Tensor(rng.standard_normal((2, 5, 8)), requires_grad=True)
    out = mha(x, x, x, mask=nn.causal_mask(5))
    assert out.shape == (2, 5, 8)
    out.sum().backward()
    assert x.grad is not None
    block = nn.TransformerEncoderLayer(8, 2, 16)
    assert block(x).shape == (2, 5, 8)


def test_causal_mask_blocks_future(rng):
    q = Tensor(rng.standard_normal((1, 4, 3)))
    v = Tensor(np.arange(4, dtype=float).reshape(1, 4, 1))
    out = nn.scaled_dot_product_attention(q, q, v, nn.causal_mask(4)).data[0, :, 0]
    assert out[0] == pytest.approx(0.0)
    assert np.all(out <= np.arange(4) + 1e-9)


def test_init_functions_shapes_and_scales():
    t = Tensor(np.empty((64, 32)))
    nn.init.xavier_uniform_(t)
    assert abs(t.data.std() - np.sqrt(2.0 / 96)) < 0.02
    nn.init.orthogonal_(t)
    np.testing.assert_allclose(t.data.T @ t.data, np.eye(32), atol=1e-8)
    nn.init.zeros_(t)
    assert not t.data.any()


@pytest.mark.parametrize("cls", [nn.RNN, nn.GRU, nn.LSTM])
def test_bidirectional_recurrent(cls, rng):
    layer = cls(input_size=4, hidden_size=6, num_layers=2, bidirectional=True)
    x = Tensor(rng.standard_normal((3, 5, 4)), requires_grad=True)
    out, state = layer(x)
    assert out.shape == (3, 5, 12) and layer.output_size == 12
    h = state[0] if cls is nn.LSTM else state
    assert h.shape == (4, 3, 6)
    np.testing.assert_allclose(out.data[:, -1, :6], h.data[2])   # last forward step of top layer
    np.testing.assert_allclose(out.data[:, 0, 6:], h.data[3])    # last backward step of top layer
    out.sum().backward()
    assert x.grad.shape == x.shape
    assert len(list(layer.parameters())) == 16


def test_bidirectional_sees_the_future(rng):
    layer = nn.RNN(1, 4, bidirectional=True)
    x = rng.standard_normal((1, 6, 1))
    base = layer(Tensor(x))[0].data
    x2 = x.copy()
    x2[0, -1, 0] += 1.0
    changed = layer(Tensor(x2))[0].data
    assert not np.allclose(base[0, 0, 4:], changed[0, 0, 4:])
    np.testing.assert_allclose(base[0, 0, :4], changed[0, 0, :4])


def test_transformer_encoder_decoder_shapes_and_causality(rng):
    model = nn.Transformer(src_vocab=11, tgt_vocab=13, d_model=16, num_heads=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32)
    src = rng.integers(0, 11, (2, 7))
    tgt = rng.integers(0, 13, (2, 5))
    logits = model(src, tgt)
    assert logits.shape == (2, 5, 13)
    tgt2 = tgt.copy()
    tgt2[:, -1] = (tgt2[:, -1] + 1) % 13
    logits2 = model(src, tgt2)
    np.testing.assert_allclose(logits.data[:, :-1], logits2.data[:, :-1], atol=1e-10)
    assert not np.allclose(logits.data[:, -1], logits2.data[:, -1])
    loss = F.cross_entropy(logits.reshape(-1, 13), tgt.reshape(-1))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_positional_encoding_and_padding_mask():
    pe = nn.PositionalEncoding(8, max_len=10)
    x = Tensor(np.zeros((1, 4, 8)))
    out = pe(x).data[0]
    np.testing.assert_allclose(out[0, 0::2], 0.0)
    np.testing.assert_allclose(out[0, 1::2], 1.0)
    assert not np.allclose(out[1], out[2])
    mask = nn.padding_mask([3, 1], max_len=4)
    assert mask.shape == (2, 1, 1, 4)
    assert mask[0, 0, 0].tolist() == [True, True, True, False]
