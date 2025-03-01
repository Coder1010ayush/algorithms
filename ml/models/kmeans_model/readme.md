# Clustering Algorithms Codebase

## Overview
This codebase implements multiple clustering algorithms, including:

- **K-Means Clustering** (with different initialization methods)
- **K-Medoids Clustering**
- **Agglomerative Hierarchical Clustering**

Each algorithm is implemented as a Python class, using NumPy and SciPy for efficient mathematical computations.

---

## Dependencies
Ensure you have the following libraries installed before running the code:

```bash
pip install numpy scipy
```

---

## KMeansClusterClassification

### Class Definition:
```python
class KMeansClusterClassification(BaseModel)
```
This class implements the K-Means clustering algorithm with multiple initialization strategies.

### Parameters:
- `degree` *(int)*: Number of clusters (k)
- `max_iteration` *(int)*: Maximum iterations before convergence
- `init_type` *(str)*: Initialization method (`"uniform"`, `"kmeans++"`, `"random"`)
- `tolerence` *(float)*: Convergence threshold
- `metric` *(str)*: Distance metric (default: "euclidean")

### Methods:
#### `forward(x: np.ndarray, y)`
- Initializes centroids and iteratively updates them until convergence.
- Uses the centroid update rule based on the mean of assigned points.
- Returns the cluster labels for each data point.

#### `predict(x: np.ndarray)`
- Assigns new data points to the nearest cluster centroid.

---

## KMedoids

### Class Definition:
```python
class KMedoids
```
This class implements the K-Medoids clustering algorithm, which is more robust to outliers than K-Means.

### Parameters:
- `degree` *(int)*: Number of clusters
- `max_iters` *(int, optional)*: Maximum number of iterations
- `metric` *(str)*: Distance metric (default: "euclidean")

### Methods:
#### `forward(X: np.ndarray, y)`
- Initializes medoids randomly.
- Assigns each point to the nearest medoid.
- Updates medoids by selecting the point with the smallest total distance within each cluster.
- Returns the cluster labels.

#### `predict(X: np.ndarray)`
- Assigns new data points to the nearest medoid.

---

## AgglomerativeClustering

### Class Definition:
```python
class AgglomerativeClustering(BaseModel)
```
This class implements Agglomerative Hierarchical Clustering using different linkage strategies.

### Parameters:
- `n_clusters` *(int)*: Desired number of clusters
- `linkage` *(str)*: Linkage criterion (`"single"`, `"average"`, `"complete"`, `"centroid"`)
- `metric` *(str)*: Distance metric (default: "euclidean")

### Methods:
#### `forward(X, y)`
- Iteratively merges the closest clusters until the desired number of clusters is reached.
- Computes distances between clusters based on the chosen linkage method.
- Stores cluster labels and centroids.

#### `predict(X_test)`
- Assigns new data points to the nearest cluster based on the computed centroids.

#### `_find_closest_clusters(distance_matrix, clusters)`
- Identifies the two closest clusters based on the linkage method.

#### `_compute_cluster_distance(cluster1, cluster2, distance_matrix)`
- Computes the distance between two clusters based on the specified linkage strategy.

#### `_merge_clusters(i, j, clusters)`
- Merges two clusters into one.

#### `_update_distances(i, j, clusters, X, distance_matrix)`
- Updates the distance matrix after merging clusters.

---

## Usage Example

### K-Means Example:
```python
import numpy as np
from models.knn_models.centroid import InitializeCentroid

# Generate synthetic data
X = np.random.rand(100, 2)

# Initialize KMeans
kmeans = KMeansClusterClassification(degree=3, init_type='kmeans++')
labels = kmeans.forward(X, None)
print(labels)
```

### K-Medoids Example:
```python
kmedoids = KMedoids(degree=3)
labels = kmedoids.forward(X, None)
print(labels)
```

### Agglomerative Clustering Example:
```python
agglo = AgglomerativeClustering(n_clusters=3, linkage='average')
agglo.forward(X, None)
print(agglo.labels_)
```

---

## Notes:
- The K-Means and K-Medoids implementations use **SciPy's `cdist`** for distance calculations.
- The Agglomerative Clustering algorithm maintains a **custom linkage-based merging process**.
- The `BaseModel` class is assumed to be a generic class that standardizes model implementations.

---

## Future Enhancements:
- Implement **DBSCAN** for density-based clustering.
- Optimize medoid selection using **PAM (Partitioning Around Medoids)**.
- Add GPU acceleration with **CuPy** for large datasets.
- Implement **spectral clustering** for non-linear separable data.

---

