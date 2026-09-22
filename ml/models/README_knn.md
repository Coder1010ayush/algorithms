# K-nearest neighbours (`ml/models/knn.py`)

`KNearestNeighbour(n_neighbours=3, metric="euclidean", task="classification")` stores the training set on `fit` and, on `predict`, computes the full distance matrix with `ml.utils.distance.pairwise`, takes the `n_neighbours` closest rows and returns the majority label (classification) or the mean target (regression).

Metrics: `"euclidean"`, `"manhattan"`, `"cosine"`.

```python
from ml.models import KNearestNeighbour
knn = KNearestNeighbour(n_neighbours=5).fit(X, y)
knn.predict(X_test)
```
