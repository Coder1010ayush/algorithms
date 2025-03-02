# XGBoost Algorithm Implementation

## Overview
This file contains a simple implementation of the XGBoost algorithm from scratch using NumPy. The implementation supports both regression and classification tasks using gradient boosting with decision trees as base learners.

## Features
- **Gradient Boosting Algo**: Implements the core logic of boosting by iteratively fitting decision trees to minimize loss.
- **Tree-Based Model**: Decision trees are constructed using gain-based splitting criteria.
- **Regularization Support**: Includes L2 regularization (`lambda_`) and pruning (`gamma`) to prevent overfitting.
- **Multi-Class Classification**: Implements softmax loss and supports multi-class classification.
- **Efficient Gradient Computation**: Uses first-order gradients and Hessians for optimization.

## Classes and Methods

### `XGTree`
Represents a single decision tree node.
```python
class XGTree:
    def __init__(self, is_leaf=False, value=None, f_idx=None, threshold=None):
```
- `is_leaf`: Boolean indicating whether the node is a leaf.
- `value`: Leaf node prediction value.
- `f_idx`: Feature index used for splitting.
- `threshold`: Threshold value for feature splitting.

---
### `TreeRegressor`
Implements a regression tree used in gradient boosting.
```python
class TreeRegressor:
    def __init__(self, max_depth=6, min_samples_split=5, lambda_=1.0, gamma=0.0):
```
- `max_depth`: Maximum depth of the tree.
- `min_samples_split`: Minimum number of samples required to split.
- `lambda_`: L2 regularization term.
- `gamma`: Minimum gain required to split a node.

#### Methods:
- `_compute_gradients(y, y_pred)`: Computes the gradients and Hessians for squared loss.
- `_get_best_split(X, y, gradients, hessians)`: Determines the best split based on gain.
- `_build_tree(X, y, gradients, hessians, depth)`: Recursively constructs a decision tree.

---
### `TreeClassifier`
Implements a classification tree used in gradient boosting.
```python
class TreeClassifier:
    def __init__(self, n_classes, max_depth=6, min_samples_split=5, lambda_=1.0, gamma=0.0):
```
- `n_classes`: Number of target classes.
- `max_depth`, `min_samples_split`, `lambda_`, `gamma`: Similar to `TreeRegressor`.

#### Methods:
- `_softmax(logits)`: Computes softmax probabilities.
- `_compute_gradients(y_one_hot, y_pred)`: Computes gradients and Hessians for softmax loss.
- `_get_best_split(X, y, gradients, hessians)`: Determines the best split.
- `_build_tree(X, y, gradients, hessians, depth)`: Constructs a classification tree.

---
### `XgBoostModelRegressor`
Implements gradient boosting for regression.
```python
class XgBoostModelRegressor(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, min_samples_split=5, lambda_=1.0, gamma=0.0):
```
- `n_estimators`: Number of boosting iterations.
- `learning_rate`: Controls step size.
- `max_depth`, `min_samples_split`, `lambda_`, `gamma`: Similar to `TreeRegressor`.

#### Methods:
- `forward(X, y)`: Fits the model to training data.
- `_predict_sample(node, sample)`: Recursively predicts a sample.
- `predict(X)`: Returns predictions for input data.

---
### `XgBoostModelClassifier`
Implements gradient boosting for classification.
```python
class XgBoostModelClassifier(BaseModel):
    def __init__(self, n_classes, n_estimators=100, learning_rate=0.1, max_depth=6, min_samples_split=5, lambda_=1.0, gamma=0.0):
```
- `n_classes`: Number of target classes.
- `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `lambda_`, `gamma`: Similar to `TreeClassifier`.

#### Methods:
- `forward(X, y)`: Fits the model to training data.
- `_predict_sample(node, sample)`: Recursively predicts a sample.
- `predict_proba(X)`: Returns class probabilities.
- `predict(X)`: Returns class predictions.

## Usage
### Training a Regressor:
```python
xgb_regressor = XgBoostModelRegressor(n_estimators=50, learning_rate=0.1)
xgb_regressor.forward(X_train, y_train)
preds = xgb_regressor.predict(X_test)
```

### Training a Classifier:
```python
xgb_classifier = XgBoostModelClassifier(n_classes=3, n_estimators=50, learning_rate=0.1)
xgb_classifier.forward(X_train, y_train)
preds = xgb_classifier.predict(X_test)
```

## Notes
- The implementation does not support missing value handling or feature importance computation.
- Further optimizations such as parallelization and histogram-based splitting can improve efficiency.
