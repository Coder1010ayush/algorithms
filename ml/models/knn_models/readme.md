# Clustering Algorithms Cheat Sheet

## 1. Partition-Based Clustering
### 1.1 k-Means Clustering
**Steps:**
1. Choose k initial centroids.
2. Assign each data point to the nearest centroid.
3. Compute new centroids based on current clusters.
4. Repeat until centroids converge.

**Variants:**
- **k-Medoids**: Uses actual data points as centroids.
- **Mini-Batch k-Means**: Faster approximation using mini-batches.
- **Fuzzy c-Means**: Allows soft clustering (probabilistic assignments).

---
## 2. Hierarchical Clustering
### 2.1 Agglomerative Clustering (Bottom-Up)
**Steps:**
1. Start with each data point as a cluster.
2. Merge the two closest clusters iteratively.
3. Continue until one cluster remains.

**Linkage Methods:**
- **Single Linkage**: Uses the minimum pairwise distance.
- **Complete Linkage**: Uses the maximum pairwise distance.
- **Average Linkage**: Uses the mean pairwise distance.

### 2.2 Divisive Clustering (Top-Down)
**Steps:**
1. Start with all data in one cluster.
2. Recursively split clusters based on distance.

---
## 3. Density-Based Clustering
### 3.1 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
**Steps:**
1. Define **eps** (radius) and **minPts** (minimum points for a dense region).
2. Expand clusters from high-density points.
3. Mark low-density points as noise.

### 3.2 OPTICS (Ordering Points To Identify Clustering Structure)
- Similar to DBSCAN but handles varying density better.

### 3.3 Mean-Shift Clustering
**Steps:**
1. Compute density estimates at each point.
2. Shift each point toward areas of higher density.
3. Stop when convergence is reached.

---
## 4. Graph-Based Clustering
### 4.1 Spectral Clustering
**Steps:**
1. Construct a similarity graph.
2. Compute the Laplacian matrix.
3. Use eigenvectors to transform data and apply k-Means.

### 4.2 Markov Clustering (MCL)
- Simulates random walks on a graph to find clusters.

---
## 5. Model-Based Clustering
### 5.1 Gaussian Mixture Models (GMM)
**Steps:**
1. Initialize k Gaussian distributions.
2. Use the Expectation-Maximization (EM) algorithm to update parameters.
3. Assign probabilities for each data point.

### 5.2 Bayesian Gaussian Mixture Models
- Uses priors to determine the optimal number of clusters.

---
## 6. Grid-Based Clustering
### 6.1 STING (Statistical Information Grid)
- Divides data into a hierarchical grid structure.

### 6.2 CLIQUE (Clustering In QUEst)
- Performs subspace clustering to handle high-dimensional data.

---
## 7. Deep Learning-Based Clustering
### 7.1 Autoencoder-Based Clustering
- Uses autoencoders to learn feature embeddings before clustering.

### 7.2 Self-Supervised Clustering
- Uses contrastive learning for unsupervised clustering.

---
## 8. Evolutionary & Swarm-Based Clustering
### 8.1 Genetic Algorithm Clustering
- Uses evolutionary strategies to optimize clusters.

### 8.2 Particle Swarm Optimization (PSO) Clustering
- Treats cluster centers as particles in an optimization problem.

---
## Choosing the Right Algorithm
| Algorithm | Shape | Density | Speed | Robust to Noise | Handles Outliers |
|-----------|-------|---------|-------|----------------|----------------|
| k-Means | Convex | Uniform | Fast | No | No |
| Hierarchical | Any | Any | Slow | Yes | Yes |
| DBSCAN | Arbitrary | Varying | Medium | Yes | Yes |
| Spectral | Arbitrary | Any | Medium | No | No |
| GMM | Elliptical | Any | Medium | No | No |
| Mean-Shift | Arbitrary | Any | Slow | Yes | Yes |

---
## Implementation Notes
- **Preprocessing**: Normalize data for distance-based clustering (e.g., k-Means, DBSCAN).
- **Parameter Tuning**: Use the **elbow method** for k-Means, **silhouette score**, or **cross-validation**.
- **Scalability**: For large datasets, use **Mini-Batch k-Means** or **Approximate DBSCAN**.

---
### References
- Scikit-Learn: https://scikit-learn.org/
- Clustering Theory: https://en.wikipedia.org/wiki/Cluster_analysis
