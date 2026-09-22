# PCA (`ml/models/pca.py`)

`PCA(n_components=None)` centres the data and takes the right singular vectors of the centred matrix, which are the eigenvectors of the covariance matrix without forming it explicitly.

```python
from ml.models import PCA
pca = PCA(n_components=2).fit(X)
Z = pca.transform(X)          # (n, 2)
X_hat = pca.inverse_transform(Z)
pca.components                # (2, n_features)
pca.explained_variance_ratio  # fraction of variance per component
```

`fit_transform(X)` combines the two steps. `n_components` larger than the number of features raises `ValueError`.
