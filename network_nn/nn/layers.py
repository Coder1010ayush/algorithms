from typing import Optional

import numpy as np

from network_nn.autograd import functional as F
from network_nn.nn import init
from network_nn.nn.module import Module
from network_nn.tensor import Parameter, Tensor


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = Parameter(np.empty((in_features, out_features)))
        self.bias = Parameter(np.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / np.sqrt(self.in_features)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = Parameter(np.empty((num_embeddings, embedding_dim)))
        init.normal_(self.weight)

    def forward(self, indices) -> Tensor:
        idx = indices.data if isinstance(indices, Tensor) else np.asarray(indices)
        return self.weight[idx.astype(np.int64)]

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"


class Dropout(Module):
    def __init__(self, p: float = 0.5, rng: Optional[np.random.Generator] = None):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("Dropout probability must be in [0, 1).")
        self.p = p
        self.rng = rng or np.random.default_rng()

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training, self.rng)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Flatten(Module):
    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Residual(Module):
    """y = x + fn(x)"""

    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


class ReLU(Module):
    def forward(self, x):
        return F.relu(x)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)


class Sigmoid(Module):
    def forward(self, x):
        return F.sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        return F.tanh(x)


class GELU(Module):
    def forward(self, x):
        return F.gelu(x)


class SiLU(Module):
    def forward(self, x):
        return F.silu(x)


class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return F.softmax(x, self.axis)


class LogSoftmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return F.log_softmax(x, self.axis)
