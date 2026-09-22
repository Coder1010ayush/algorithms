# algorithms

Machine learning algorithms and a deep learning framework implemented from scratch in NumPy.

| Package | What it is |
| --- | --- |
| [`ml/`](ml/README.md) | Classical ML: linear and logistic regression, SVM, KNN, naive Bayes, decision trees, random forest, AdaBoost, gradient boosting, XGBoost-style boosting, k-means, k-medoids, agglomerative clustering, Gaussian mixture models, PCA, Gaussian processes. Every model exposes `fit(X, y)` and `predict(X)`. |
| [`network_nn/`](network_nn/README.md) | A PyTorch-style framework: `Tensor` with reverse-mode autograd, `nn` layers (linear, conv 1D/2D/3D, pooling, batch/layer norm, RNN/GRU/LSTM, multi-head attention, transformer block), losses, optimizers and LR schedulers. |
| `tests/` | pytest suite. `tests/network_nn/` checks every autograd op against finite differences; `tests/ml/` trains each model on toy data. |

## Setup

The project uses a local virtualenv managed with [uv](https://github.com/astral-sh/uv).

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Runtime dependencies are NumPy, pandas and SciPy (SciPy is used only for a few numerical
helpers such as Cholesky solves and Bessel functions). No scikit-learn, no PyTorch.

## Tests

```bash
.venv/bin/python -m pytest              # everything
.venv/bin/python -m pytest tests/ml     # one package
.venv/bin/python -m pytest -k conv      # by keyword
```

## Quick start

```python
import numpy as np
from ml.models import RandomForest
from network_nn import Tensor, nn, optim, functional as F

X = np.random.randn(200, 2); y = (X[:, 0] * X[:, 1] > 0).astype(int)

forest = RandomForest(n_trees=20).fit(X, y)
print("forest accuracy:", (forest.predict(X) == y).mean())

model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
opt = optim.Adam(model.parameters(), lr=0.05)
for _ in range(300):
    opt.zero_grad()
    loss = F.cross_entropy(model(Tensor(X)), y)
    loss.backward()
    opt.step()
print("mlp accuracy:", (model(Tensor(X)).data.argmax(1) == y).mean())
```

## Roadmap

Not yet implemented, kept here as the to-do list:

- **ml**: LightGBM/CatBoost-style boosting, quantile regression, DBSCAN, mean shift, spectral
  clustering, t-SNE, UMAP, ICA, LDA, HMM, Apriori/FP-growth, self-training and co-training.
- **network_nn**: dataloaders, model checkpointing, a full transformer decoder, GPU backend.
- **Reinforcement learning**: Q-learning, SARSA, DQN, policy gradient, actor-critic, PPO, MCTS.
- **Classic algorithms**: graph algorithms (Dijkstra, Bellman-Ford, Floyd-Warshall, A*,
  Tarjan SCC, Hopcroft-Karp, Edmonds-Karp, Prim/Kruskal), computational geometry (convex hull,
  sweep-line intersection, closest pair, Voronoi), data structures (red-black tree, AVL, segment
  tree with lazy propagation, suffix tree, Fenwick tree, treap, B-tree, KD-tree, van Emde Boas),
  FFT, simulated annealing, genetic algorithms, Bloom filter.
