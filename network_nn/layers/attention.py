# ------------------------------------------- utf-8 encoding ----------------------------------------------
# this file implements all the varients of attention layer and positional encoding also with its all varients
# Paper link is here ; https://arxiv.org/pdf/1706.03762
import numpy as np
import os , sys 
from math import sqrt
from collections import Counter
from module import Module
import autograd.autodiff as diff
from models import Linear

class Attention(Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.query_layer = Linear(in_feature=d_model, out_feature=d_model)
        self.key_layer = Linear(in_feature=d_model, out_feature=d_model)
        self.value_layer = Linear(in_feature=d_model, out_feature=d_model)
        self.out_proj = Linear(in_feature=d_model, out_feature=d_model)

    def forward(self, query, key, value, mask=None):
        B, L, _ = query.data.shape
        d_k = self.d_model

        Q = self.query_layer(query)  # [B, L, d_model]
        K = self.key_layer(key)
        V = self.value_layer(value)

        # Scaled dot-product attention
        scores = Q.matmul(K.transpose(1, 2)) / sqrt(d_k)  # [B, L, L]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = diff.softmax(scores, dim=-1)       # [B, L, L]
        context = attn_weights.matmul(V)                  # [B, L, d_model]

        output = self.out_proj(context)                   # [B, L, d_model]
        return output
    
class MultiHeadAttention(Module):

    def __init__(self, d_model: int, num_head: int):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        assert d_model % num_head == 0, "d_model must be divisible by num_head"
        self.head_dim = d_model // num_head

        self.query_layer = Linear(in_feature=d_model, out_feature=d_model)
        self.key_layer = Linear(in_feature=d_model, out_feature=d_model)
        self.value_layer = Linear(in_feature=d_model, out_feature=d_model)
        self.out_proj = Linear(in_feature=d_model, out_feature=d_model)

    def forward(self, query, key, value, mask=None):
        B, L, _ = query.data.shape
        H = self.num_head
        D = self.head_dim

        Q = self.query_layer(query)   # [B, L, d_model]
        K = self.key_layer(key)
        V = self.value_layer(value)

        # Split into heads -> [B, L, H, D] -> [B, H, L, D]
        Q = Q.reshape(B, L, H, D).transpose(1, 2)  # [B, H, L, D]
        K = K.reshape(B, L, H, D).transpose(1, 2)
        V = V.reshape(B, L, H, D).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q.matmul(K.transpose(2, 3)) / sqrt(D)  # [B, H, L, L]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = diff.softmax(scores, axis=-1)
        context = attention_weights.matmul(V)  # [B, H, L, D]
        context = context.transpose(1, 2).reshape(B, L, H * D)  # [B, L, d_model]

        output = self.out_proj(context)  # [B, L, d_model]
        return output
