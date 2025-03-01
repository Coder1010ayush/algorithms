# Random Forest Implementation

## Overview
This document describes the implementation of a **Random Forest** algorithm that supports both **classification** and **regression** tasks. The implementation is designed for efficiency and includes options for **parallel processing** and **out-of-bag (OOB) score computation**.

## Features
- Supports **classification** and **regression** tasks.
- Implements **bootstrapping** for training.
- Supports **parallel training** using multiprocessing.
- Allows computation of **out-of-bag (OOB) score**.
- Uses **Decision Trees** (CART or ID3 for classification, Regression Tree for regression).

## Class Definition
```python
class RandomForest(BaseModel):
```
### Parameters:
- `n_trees` *(int, default=50)*: Number of trees in the forest.
- `min_sample_split` *(int, default=5)*: Minimum samples required to split a node.
- `max_depth` *(int, default=10)*: Maximum depth of each decision tree.
- `max_features` *(int, default=None)*: Maximum number of features to consider for the best split.
- `task` *(str, default='classification')*: Type of task ('classification' or 'regression').
- `oob_score` *(bool, default=False)*: Whether to compute Out-of-Bag (OOB) score.
- `parallel` *(bool, default=True)*: Whether to use parallel processing.
- `trees` *(list)*: Stores trained decision trees.
- `oob_predictions` *(dict)*: Stores OOB predictions for score calculation.

## Methods

### `_bootstrap_sample(X, y)`
Creates a bootstrap sample from the dataset for training individual trees.

#### Returns:
- `X_sample` *(array)*: Sampled training features.
- `y_sample` *(array)*: Sampled training labels.
- `oob_indices` *(list)*: Out-of-Bag indices (not included in the sample).

---

### `_train_tree(args)`
Trains a single decision tree on the bootstrap sample.

#### Parameters:
- `args` *(tuple)*: Contains `X_sample`, `y_sample`, and `oob_indices`.

#### Returns:
- `tree` *(DecisionTreeCART or DecisionTreeRegression)*: Trained decision tree.

---

### `forward(X, y)`
Trains the Random Forest by constructing multiple decision trees using bootstrap samples.

#### Parameters:
- `X` *(array)*: Feature matrix.
- `y` *(array)*: Target labels.

#### Steps:
1. Generates bootstrap samples for each tree.
2. Trains each tree in parallel (if enabled).
3. Stores OOB predictions for OOB score computation.

---

### `predict(X)`
Predicts labels for given input samples using majority voting (classification) or averaging (regression).

#### Parameters:
- `X` *(array)*: Feature matrix for prediction.

#### Returns:
- `predictions` *(array)*: Predicted labels or values.

---

### `compute_oob_score(y)`
Computes the Out-of-Bag (OOB) score for the trained forest.

#### Parameters:
- `y` *(array)*: True target labels.

#### Returns:
- `oob_score` *(float or None)*: OOB score (accuracy for classification, MSE for regression).

## Conclusion
This implementation provides an efficient and scalable **Random Forest** model with key functionalities such as **parallel processing** and **OOB score computation**. It can be used for both classification and regression tasks, making it a versatile tool in machine learning.

