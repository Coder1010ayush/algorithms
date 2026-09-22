# ml — machine learning from scratch

Classical ML algorithms implemented with NumPy (SciPy only for Cholesky solves, Bessel functions and L-BFGS in the Gaussian process). No scikit-learn, no PyTorch.

## Layout

```
ml/
  base.py                 BaseModel: fit(X, y) / predict(X) / fit_predict
  optim/
    gradient_descent.py   GradientDescent (batch | stochastic, L1/L2/elastic penalties)
    model_optimizer.py    GradientOptimizer for models exposing forward/compute_loss/compute_gradient
  utils/
    activation.py         relu, sigmoid, tanh, softmax, Activation
    distance.py           euclidean, manhattan, cosine, pairwise, Distance
    metrics.py            RegressionMetric, ClassificationMetric and the underlying functions
    normalization.py      Normalization (min_max, z_score, max_abs, mean, unit_vector, log, box_cox, yeo_johnson, quantile)
  models/
    linear.py             LinearRegression, LogisticRegression
    svm.py                SVMClassifier, SVMRegression (linear / rbf / poly kernels)
    knn.py                KNearestNeighbour
    naive_bayes.py        NaiveBayes (gaussian / multinomial / bernoulli)
    tree.py               DecisionTreeID3, DecisionTreeCART, DecisionTreeRegression
    random_forest.py      RandomForest (bootstrap, max_features, OOB score, multiprocessing)
    boosting/             AdaBoostClassifier, AdaBoostRegressor, GradientBoostClassifier/Regressor, XGBoostClassifier/Regressor
    clustering/           KMeans, KMedoids, AgglomerativeClustering, centroid helpers
    gmm.py                GaussianMixtureModel (EM)
    pca.py                PCA (SVD based)
    gaussian_process.py   GaussianProcessRegression, GaussianProcessClassification, MultiClassGaussianProcessClassification
```

## Conventions

- Every model is constructed with hyperparameters, then `model.fit(X, y)` and `model.predict(X)`. Clustering and PCA accept `fit(X)`.
- Probabilistic classifiers also expose `predict_proba`.
- Anything random takes `random_state`.

```python
from ml.models import LogisticRegression

model = LogisticRegression(optimizer="batch", epochs=200, learning_rate=0.1).fit(X, y)
labels = model.predict(X_test)
```

## Tests

```
.venv/bin/python -m pytest tests/ml
```
