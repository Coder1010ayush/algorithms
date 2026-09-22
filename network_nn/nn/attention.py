from typing import Optional

import numpy as np

from network_nn.autograd import functional as F
from network_nn.nn.layers import Dropout, GELU, Linear
from network_nn.nn.module import Module, Sequential
from network_nn.nn.norm import LayerNorm
from network_nn.tensor import Tensor


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
    """q, k, v: (..., L, D). mask: broadcastable boolean array, False positions are masked out."""
    scores = (q @ k.transpose(-1, -2)) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~np.asarray(mask, dtype=bool), -1e9)
    return F.softmax(scores, -1) @ v


def causal_mask(length: int) -> np.ndarray:
    return np.tril(np.ones((length, length), dtype=bool))


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
        context = scaled_dot_product_attention(q, k, v, mask)  # (B, H, L, D)
        b, _, l, _ = context.shape
        context = context.transpose(1, 2).reshape(b, l, self.embed_dim)
        return self.dropout(self.out_proj(context))

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}"


class TransformerEncoderLayer(Module):
    """Pre-norm transformer block: x + MHA(LN(x)); x + FFN(LN(x))."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = Sequential(Linear(d_model, dim_feedforward), GELU(), Dropout(dropout), Linear(dim_feedforward, d_model))
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, mask)
        return x + self.dropout(self.ffn(self.norm2(x)))
