from typing import Optional

import numpy as np

from network_nn.autograd import functional as F
from network_nn.nn.layers import Dropout, Embedding, GELU, Linear
from network_nn.nn.module import Module, ModuleList, Sequential
from network_nn.nn.norm import LayerNorm
from network_nn.tensor import Tensor


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
    """q: (..., Lq, D), k/v: (..., Lk, D). mask: boolean broadcastable to (Lq, Lk); False is masked out."""
    scores = (q @ k.transpose(-1, -2)) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~np.asarray(mask, dtype=bool), -1e9)
    return F.softmax(scores, -1) @ v


def causal_mask(length: int) -> np.ndarray:
    return np.tril(np.ones((length, length), dtype=bool))


def padding_mask(lengths, max_len: int) -> np.ndarray:
    """(batch,) lengths -> (batch, 1, 1, max_len) mask usable for multi-head keys."""
    lengths = np.asarray(lengths)
    return (np.arange(max_len)[None, :] < lengths[:, None])[:, None, None, :]


class PositionalEncoding(Module):
    """Adds the fixed sinusoidal position code of "Attention Is All You Need"."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        position = np.arange(max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div)
        pe[:, 1::2] = np.cos(position * div)[:, : d_model // 2]
        self.register_buffer("pe", Tensor(pe))
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + Tensor(self.pe.data[: x.shape[1]]))


class MultiheadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim, self.num_heads, self.head_dim = embed_dim, num_heads, embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias)
        self.dropout = Dropout(dropout)

    def _split(self, x: Tensor) -> Tensor:
        b, l, _ = x.shape
        return x.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)

    def forward(self, query: Tensor, key: Optional[Tensor] = None, value: Optional[Tensor] = None, mask=None) -> Tensor:
        key = query if key is None else key
        value = key if value is None else value
        q, k, v = self._split(self.q_proj(query)), self._split(self.k_proj(key)), self._split(self.v_proj(value))
        context = scaled_dot_product_attention(q, k, v, mask)  # (B, H, Lq, D)
        b, _, l, _ = context.shape
        context = context.transpose(1, 2).reshape(b, l, self.embed_dim)
        return self.dropout(self.out_proj(context))

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}"


def _feed_forward(d_model: int, dim_feedforward: int, dropout: float) -> Sequential:
    return Sequential(Linear(d_model, dim_feedforward), GELU(), Dropout(dropout), Linear(dim_feedforward, d_model))


class TransformerEncoderLayer(Module):
    """Pre-norm block: x + MHA(LN(x)); x + FFN(LN(x))."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = _feed_forward(d_model, dim_feedforward, dropout)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, mask)
        return x + self.dropout(self.ffn(self.norm2(x)))


class TransformerDecoderLayer(Module):
    """Pre-norm block: masked self-attention, cross-attention over `memory`, feed-forward."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.cross_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.norm3 = LayerNorm(d_model)
        self.ffn = _feed_forward(d_model, dim_feedforward, dropout)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor, tgt_mask=None, memory_mask=None) -> Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, tgt_mask)
        x = x + self.cross_attn(self.norm2(x), memory, memory, memory_mask)
        return x + self.dropout(self.ffn(self.norm3(x)))


class TransformerEncoder(Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.layers = ModuleList([TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerDecoder(Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.layers = ModuleList([TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)

    def forward(self, x: Tensor, memory: Tensor, tgt_mask=None, memory_mask=None) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)


class Transformer(Module):
    """Encoder-decoder over token ids: embeddings + positional encoding + stacks + output projection."""

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embed = Embedding(src_vocab, d_model)
        self.tgt_embed = Embedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = TransformerEncoder(d_model, num_heads, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, num_decoder_layers, dim_feedforward, dropout)
        self.out = Linear(d_model, tgt_vocab)

    def encode(self, src, src_mask=None) -> Tensor:
        return self.encoder(self.pos(self.src_embed(src) * np.sqrt(self.d_model)), src_mask)

    def decode(self, tgt, memory: Tensor, tgt_mask=None, memory_mask=None) -> Tensor:
        x = self.pos(self.tgt_embed(tgt) * np.sqrt(self.d_model))
        return self.out(self.decoder(x, memory, tgt_mask, memory_mask))

    def forward(self, src, tgt, src_mask=None, memory_mask=None) -> Tensor:
        tgt_len = np.asarray(tgt).shape[1]
        return self.decode(tgt, self.encode(src, src_mask), causal_mask(tgt_len), memory_mask)
