# Optimizers

All optimizers take an iterable of parameters and expose `zero_grad()` and `step()`.
`weight_decay` adds `λθ` to the gradient (L2), except in `AdamW` where it is decoupled.

```python
opt = optim.Adam(model.parameters(), lr=1e-3)
opt.zero_grad(); loss.backward(); opt.step()
```

| Class | Update rule | Defaults |
| --- | --- | --- |
| `SGD` | `v = μv + g`, `θ -= η v` (Nesterov: `θ -= η (g + μ v)`) | `lr=0.01, momentum=0` |
| `Adam` | `m = β₁m + (1-β₁)g`, `v = β₂v + (1-β₂)g²`, bias-corrected, `θ -= η m̂ / (√v̂ + ε)` | `lr=1e-3, betas=(0.9, 0.999)` |
| `AdamW` | Adam with `θ -= η λ θ` applied separately | `weight_decay=0` |
| `Nadam` | Adam with a Nesterov-style first moment | as Adam |
| `RMSprop` | `s = αs + (1-α)g²`, `θ -= η g / (√s + ε)` | `lr=0.01, alpha=0.99` |
| `Adagrad` | `s += g²`, `θ -= η g / (√s + ε)` | `lr=0.01` |
| `Adadelta` | `Δ = √(E[Δ²]+ε) / √(E[g²]+ε) · g`, running averages of both | `lr=1.0, rho=0.9` |

# Schedulers

Call `scheduler.step()` after `optimizer.step()`. Each scheduler reads the optimizer's
learning rate at construction as `base_lr`. See `SCHEDULERS.md` for the formulas.

| Class | Arguments |
| --- | --- |
| `ConstantLR` | |
| `StepLR` | `step_size, gamma` |
| `ExponentialLR` | `gamma` |
| `LinearLR` | `total_steps, final_lr` |
| `CosineAnnealingLR` | `total_steps, min_lr` |
| `PolynomialLR` | `total_steps, power` |
| `CyclicLR` | `step_size, max_lr, min_lr` |
| `WarmupLR` | `warmup_steps, after=<another scheduler>` |
| `ReduceLROnPlateau` | `factor, patience, mode`; `step(metric)` takes the monitored value |
