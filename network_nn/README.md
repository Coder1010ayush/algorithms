# network_nn

A small PyTorch-style deep learning framework built on NumPy: a reverse-mode autograd
engine, layers, losses, optimizers and learning-rate schedulers.

```python
import numpy as np
from network_nn import Tensor, nn, optim, functional as F

x = Tensor(np.random.randn(64, 2))
y = ((x.data[:, 0] * x.data[:, 1]) > 0).astype(np.int64)

model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
opt = optim.Adam(model.parameters(), lr=0.05)

for step in range(300):
    opt.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
```

## Layout

| Path | Contents |
| --- | --- |
| `tensor.py` | `Tensor`, `Parameter`, `no_grad` / `enable_grad`, topological backward pass |
| `autograd/function.py` | `Function` base class, `unbroadcast`, reduction helpers |
| `autograd/elementwise.py` | `+ - * /`, pow, exp, log, sqrt, trig, abs, clip, masked_fill, where |
| `autograd/reduce.py` | sum, mean, var, std, max, min |
| `autograd/shape.py` | reshape, permute, squeeze, unsqueeze, broadcast, indexing, stack, concat, pad |
| `autograd/linalg.py` | batched `matmul` |
| `autograd/activation.py` | relu, leaky relu, sigmoid, tanh, swish, gelu, softmax, log softmax |
| `autograd/loss.py` | mse, bce, bce-with-logits, cross entropy, huber |
| `autograd/norm.py` | `Normalize`, the primitive behind layer norm and batch norm |
| `autograd/conv.py`, `autograd/pool.py` | N-d convolution and max/avg pooling (1D, 2D and 3D share one implementation) |
| `autograd/functional.py` | the functional API: every op as a plain function (`F.conv2d`, `F.softmax`, ...) |
| `nn/` | `Module`, `Sequential`, `ModuleList`, layers, norms, conv/pool, RNN/GRU/LSTM, attention, losses, `init` |
| `optim/` | `SGD`, `Adam`, `AdamW`, `Nadam`, `RMSprop`, `Adagrad`, `Adadelta` and schedulers |

## Adding an operation

Subclass `Function`, implement `forward` on NumPy arrays and `backward` returning one gradient per
tensor argument, then expose it in `autograd/functional.py`:

```python
class Square(Function):
    def forward(self, a):
        self.a = a
        return a * a

    def backward(self, grad):
        return 2 * self.a * grad

square = Square.apply
```

`Function.apply` records the graph only when gradients are enabled and an input requires them.
Ops that broadcast must reduce their gradient back with `unbroadcast(grad, input_shape)`.

## Tests

```
.venv/bin/python -m pytest tests/network_nn
```

Every op is checked against finite differences in `tests/network_nn/` using
`tests/network_nn/helpers.py::check_gradients`.
