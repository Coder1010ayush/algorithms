# Decision Tree Implementation

## Overview
This code implements decision tree algorithms, including:
- **ID3 (Iterative Dichotomiser 3)**: Uses entropy and information gain for classification.
- **CART (Classification and Regression Trees)**: Supports both classification (Gini index) and regression (Mean Squared Error).
- **Regression Trees**: Uses Mean Squared Error (MSE) for splitting criteria.

The implementation is structured into three classes:
1. `DecisionTreeID3` - Implements the ID3 algorithm for classification.
2. `DecisionTreeCART` - Implements the CART algorithm for classification and regression.
3. `DecisionTreeRegression` - Implements decision trees specifically for regression.

Additionally, a `Tree` class is used to represent nodes in the decision tree.

---

## Class Details

### Tree (Node Representation)
```python
class Tree:
    def __init__(self, f_index=None, threshold=None, is_leaf=False, label=None, value=None):
        self.f_index = f_index  # Index of the feature used for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.is_leaf = is_leaf  # Indicates whether the node is a leaf
        self.label = label  # Class label for classification tasks
        self.value = value  # Predicted value for regression tasks
        self.left = None  # Left child
        self.right = None  # Right child
```

---

## Decision Tree Algorithms

### 1. DecisionTreeID3

#### **Constructor**
```python
class DecisionTreeID3(BaseModel):
    def __init__(self, min_sample_split=5, max_depth=10):
```
- `min_sample_split`: Minimum number of samples required to split a node.
- `max_depth`: Maximum depth of the tree.

#### **Methods**
- `__get_best_split_criterion(x, y)`: Finds the best feature and threshold based on information gain.
- `gini_index(y)`: Calculates Gini impurity.
- `calculate_gain(y_parent, y_left, y_right)`: Computes information gain.
- `__build_decision_tree(x, y, depth)`: Recursively builds the decision tree.
- `forward(x, y)`: Trains the decision tree on given dataset.
- `predict(x)`: Predicts class labels for input samples.

---

### 2. DecisionTreeCART

#### **Constructor**
```python
class DecisionTreeCART(BaseModel):
    def __init__(self, min_sample_split=5, max_depth=10, task="classification"):
```
- `min_sample_split`: Minimum number of samples to split a node.
- `max_depth`: Maximum depth of the tree.
- `task`: Determines whether the tree is for classification or regression.

#### **Methods**
- `__get_best_split_criterion(x, y)`: Finds the best feature and threshold based on Gini impurity (classification) or MSE (regression).
- `gini_index(y_left, y_right)`: Computes Gini impurity for classification.
- `mse_index(y_left, y_right)`: Computes MSE for regression.
- `__build_decision_tree(x, y, depth)`: Recursively constructs the decision tree.
- `forward(x, y)`: Trains the decision tree.
- `predict(x)`: Predicts the class label or regression value.

---

### 3. DecisionTreeRegression

#### **Constructor**
```python
class DecisionTreeRegression(BaseModel):
    def __init__(self, min_sample_split=5, max_depth=10):
```
- `min_sample_split`: Minimum number of samples to split a node.
- `max_depth`: Maximum depth of the tree.

#### **Methods**
- `__get_best_split_criterion(x, y)`: Finds the best split using MSE.
- `calculate_mse(y_left, y_right)`: Computes MSE to evaluate splits.
- `__build_regression_tree(x, y, depth)`: Recursively builds the regression tree.
- `forward(x, y)`: Trains the regression tree.
- `predict(x)`: Predicts numerical values for input data.

---

## Summary
This implementation provides a robust framework for decision trees, covering classification (ID3, CART) and regression trees. The models support:
- Feature selection using information gain (ID3) or Gini impurity (CART).
- Handling classification and regression tasks (CART and Regression Trees).
- Tree depth control and stopping criteria based on `min_sample_split` and `max_depth`.

This code can be extended further to incorporate:
- Pruning techniques (e.g., pre-pruning and post-pruning).
- Handling missing values in features.
- Optimized tree traversal for large datasets.

