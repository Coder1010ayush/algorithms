# Decomposition and manifold learning

All classes follow `fit(X)` / `transform(X)` / `fit_transform(X)`; `predict` is an alias of `transform` where one exists. See also `PCA` in `ml/models/pca.py`.

## `TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, early_exaggeration=12, random_state=None)`

t-distributed stochastic neighbour embedding. For each point a Gaussian kernel width is found by binary search so that the neighbourhood distribution has the requested `perplexity`; the symmetrised joint distribution `P` is then matched by a Student-t distribution `Q` over the low-dimensional points using gradient descent with momentum, adaptive gains and early exaggeration. Only `fit_transform` is meaningful: there is no out-of-sample map. `kl_divergence_` holds the final objective.

## `FastICA(n_components=None, max_iter=200, tol=1e-4, fun="logcosh", whiten=True, random_state=None)`

Independent component analysis. The data is centred and whitened, then an orthogonal unmixing matrix is found by the symmetric fixed-point iteration that maximises the non-Gaussianity contrast (`logcosh`, `exp` or `cube`). `components_` maps observations to sources, `mixing_` is its pseudo-inverse.

## `LinearDiscriminantAnalysis(n_components=None, solver="eigen", reg=1e-6)`

Supervised projection maximising between-class over within-class scatter, solved by whitening with the within-class scatter and taking the leading eigenvectors of the whitened between-class scatter (at most `n_classes - 1` components). `predict` / `predict_proba` use Gaussian class-conditionals with the shared covariance, i.e. the classic LDA classifier.

## `TruncatedSVD(n_components=2)`

Rank-`k` SVD of the raw (uncentred) matrix. `components_` are the top right singular vectors, `singular_values_` the corresponding values; `explained_variance_ratio_` is the fraction of total variance captured by the projected data.

```python
from ml.models.decomposition import TSNE, FastICA, LinearDiscriminantAnalysis, TruncatedSVD

Z = TSNE(perplexity=10, random_state=0).fit_transform(X)
S = FastICA(n_components=2, random_state=0).fit_transform(X_mixed)
lda = LinearDiscriminantAnalysis().fit(X, y); lda.predict(X_test)
U = TruncatedSVD(n_components=5).fit_transform(counts)
```
