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

## `DBSCAN(eps=0.5, min_samples=5, metric="euclidean")`

Density-based clustering. A point with at least `min_samples` neighbours within `eps` is a core point; clusters are the connected components of core points plus the border points they reach. Everything else is noise and gets label `-1`. `predict` assigns a new point to the cluster of its nearest core point if that is within `eps`, otherwise `-1`. Finds arbitrarily shaped clusters and does not need the number of clusters up front.

## `MeanShift(bandwidth=None, max_iter=300, tol=1e-3, bin_seeding=False)`

Every seed (each point, or the occupied grid cells of size `bandwidth` when `bin_seeding=True`) repeatedly moves to the mean of the points within `bandwidth` (flat kernel) until it stops moving. Seeds that end up within one bandwidth of each other are merged into one `cluster_centers_` entry. `bandwidth=None` uses `estimate_bandwidth`, the median distance to the 30 % nearest neighbour.

## `SpectralClustering(n_clusters=2, affinity="rbf", gamma=1.0, n_neighbors=10, random_state=None)`

Ng-Jordan-Weiss algorithm. Build an affinity matrix (`rbf`: `exp(-gamma d²)`, `nearest_neighbors`: symmetrised k-NN graph), form the normalised affinity `D^-1/2 W D^-1/2`, take its top `n_clusters` eigenvectors, normalise the rows and run `KMeans` on them. Separates clusters that are connected but not convex, such as concentric rings. `predict` copies the label of the nearest training point.

## `AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, preference=None)`

Points exchange responsibility and availability messages until a stable set of exemplars emerges; the number of clusters is a result, not an input. `preference` (the self-similarity, default the median similarity) controls how many exemplars appear: larger gives more clusters. `cluster_centers_indices_` are the exemplar rows of `X`.
