# Autograd

Reverse-mode automatic differentiation. A `Tensor` produced by an op keeps a reference
to the `Function` instance that created it (`tensor._ctx`), and that instance keeps the
input tensors. `Tensor.backward()` sorts the graph topologically and calls each
`Function.backward` exactly once, accumulating gradients into `.grad`.

## Gradient rules used by the ops

| Op | Forward | Backward (grad of loss w.r.t. inputs) |
| --- | --- | --- |
| add | `a + b` | `g`, `g` (each reduced back to its input shape) |
| mul | `a * b` | `g * b`, `g * a` |
| div | `a / b` | `g / b`, `-g * a / b²` |
| pow | `a ** p` | `g * p * a ** (p - 1)` |
| matmul | `a @ b` | `g @ bᵀ`, `aᵀ @ g` (batch dims reduced with `unbroadcast`) |
| sum / mean | reduce over axes | `g` broadcast back over the reduced axes (÷ count for mean) |
| var / std | | `2 (a - μ) g / n`, `(a - μ) g / (n σ)` |
| max / min | | `g` routed to the arg-extreme positions (split evenly on ties) |
| softmax | `s = softmax(a)` | `s * (g - Σ g s)` |
| log_softmax | | `g - exp(out) * Σ g` |
| tan | | `g * (1 + tan²a)` |
| normalize | `x̂ = (a - μ) / σ` | `(g - mean(g) - x̂ · mean(g x̂)) / σ` |
| conv | cross-correlation | `∂w = g ⋆ windows(x)`, `∂x` = scatter of `g · w` over each receptive field |
| max pool | | `g` scattered to the arg-max position of each window |
| avg pool | | `g / kernel_volume` spread over each window |
| getitem | `a[key]` | zeros with `g` added at `key` (`np.add.at`, so repeated indices accumulate) |

## Broadcasting

NumPy broadcasting adds leading axes and stretches size-1 axes. `unbroadcast(grad, shape)`
sums over exactly those axes so the gradient matches the original input.
