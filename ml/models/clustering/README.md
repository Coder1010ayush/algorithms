# Clustering (`ml/models/clustering/`)

All distance computations go through `ml.utils.distance.pairwise` (NumPy only).

## `KMeans(n_clusters=2, max_iter=300, init="kmeans++", tol=1e-6, metric="euclidean", random_state=None)`

Lloyd's algorithm: assign each point to the nearest centroid, move centroids to the cluster means, stop when the centroid shift is below `tol`. Initialisation via `initialize_centroids`: `"random"` samples, `"kmeans++"` distance-weighted sampling, `"uniform"` a linspace between the feature minima and maxima.

## `KMedoids(n_clusters=2, max_iter=300, metric="euclidean", random_state=None)`

Same loop but each cluster centre is the member minimising the summed distance to the other members, so centres are always real data points and outliers matter less.

## `AgglomerativeClustering(n_clusters=2, linkage="single", metric="euclidean")`

Starts with one cluster per point and repeatedly merges the closest pair under the chosen linkage (`single`, `complete`, `average`, `centroid`) until `n_clusters` remain. `labels_` holds the training assignments; `predict` assigns new points to the nearest cluster centre.

## `Centroid(kind)`

Helper computing a `"mean"`, `"medoid"` or `"weighted"` centre of a set of points.

```python
from ml.models import KMeans
km = KMeans(n_clusters=3, random_state=0).fit(X)
km.labels_, km.centroids, km.predict(X_test)
```
