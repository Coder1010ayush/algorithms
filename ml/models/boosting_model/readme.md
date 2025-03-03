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
- Further optimizations such as parallelization and histogram-based splitting can improve efficiency(Tried to do but got a lot of issue so may be in future i will added).




# AdaBoost Classification Implementation

## Overview
This file contains an implementation of the AdaBoost algorithm from scratch for classification. The implementation utilizes decision trees as weak learners and updates sample weights iteratively to improve performance.

## Dependencies
This implementation relies on the following Python libraries:

```python
import numpy as np
import os
from models.basemodel import BaseModel
from models.decision_tree.decision_tree import DecisionTreeRegression, DecisionTreeCART
```

## Class: `AdaboostClassification`

### **Constructor**
```python
def __init__(
    self,
    n_estimators: int = 100,
    min_sample_split: int = 5,
    max_depth: int = 1,
):
```
- **Parameters:**
  - `n_estimators`: Number of weak learners (default: 100)
  - `min_sample_split`: Minimum samples required to split a node (default: 5)
  - `max_depth`: Maximum depth of each weak learner (default: 1)
- **Attributes:**
  - `self.model_w`: Stores model weights for each weak learner.
  - `self.trees`: Stores the trained weak learners.

### **Error Calculation**
```python
def __get_error(self, weights: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
```
- Computes the weighted classification error.

### **Updating Sample Weights**
```python
def __update_data_point_weightage(
    self, old_w, current_model_w, y: np.ndarray, y_pred: np.ndarray
):
```
- Updates sample weights based on misclassification.

### **Updating Model Weights**
```python
def __update_model_weights(self, error):
```
- Computes weight of the weak learner based on error.

### **Dataset Resampling**
```python
def __get_upsampled_data(self, normal_w: np.ndarray, y: np.ndarray, x: np.ndarray):
```
- Uses weighted sampling to create a new dataset for the next iteration.

### **Training Function**
```python
def forward(self, x: np.ndarray, y: np.ndarray):
```
- Trains multiple weak learners iteratively.
- Updates sample weights and resamples dataset.
- Stops training if a weak learner has an error rate >= 0.5.

### **Prediction Functions**
```python
def __predict_sample(self, x):
```
- Aggregates predictions from weak learners using weighted voting.

```python
def predict(self, x: np.ndarray):
```
- Predicts class labels for input samples.

## Notes
- Uses `DecisionTreeCART` as a weak learner.
- Handles sample weighting and dataset resampling effectively.
- Stops early if the weak learner is too weak (error ≥ 0.5).




# AdaBoost Regression from Scratch

## Overview
This file contains an implementation of AdaBoost for regression, built from scratch using NumPy. The weak learner used in this implementation is a Decision Tree Regressor (CART). The AdaBoost algorithm iteratively trains weak models, assigns them weights based on their performance, and updates sample weights to focus more on difficult examples.

## Features
- Implements AdaBoost for regression using Decision Trees.
- Uses weighted sampling to improve weak learners iteratively.
- Implements custom loss and weight update mechanisms.
- Supports hyperparameter tuning for `n_estimators`, `min_sample_split`, and `max_depth`.


## Implementation Details
### Core Functions
1. **Error Calculation:**
   - Computes weighted squared error for weak learners.

2. **Weight Update Mechanism:**
   - Adjusts weights of incorrectly predicted samples.
   - Uses exponential update rules for adjusting data point weights.

3. **Weighted Sampling:**
   - Samples training data based on updated weight distribution.

4. **Final Prediction:**
   - Uses a weighted sum of weak learners' outputs.

### Model Workflow
1. Initialize sample weights.
2. Train weak learners sequentially on weighted samples.
3. Compute weighted error for each weak model.
4. Assign model weight based on performance.
5. Update sample weights to emphasize harder examples.
6. Repeat until `n_estimators` is reached.

## Hyperparameters
| Parameter          | Description                                    | Default Value |
|-------------------|--------------------------------|---------------|
| `n_estimators`    | Number of weak learners         | 100           |
| `min_sample_split` | Minimum samples to split a node | 5             |
| `max_depth`       | Maximum depth of weak learners | 1             |

-----
-----
-----



# Gradient Boosting Algorithm Implementation

## Overview
This file contains a simple implementation of the Gradient Boosting algorithm from scratch using NumPy. The implementation supports both regression and classification tasks using gradient boosting with decision trees as base learners.

## Features
- **Gradient Boosting Algo**: Implements the core logic of boosting by iteratively fitting decision trees to minimize loss.
- **Tree-Based Model**: Decision trees are constructed using gain-based splitting criteria.
- **Regularization Support**: Includes L2 regularization (`lambda_`) and pruning (`gamma`) to prevent overfitting.
- **Multi-Class Classification**: Implements softmax loss and supports multi-class classification.
- **Efficient Gradient Computation**: Uses first-order gradients and Hessians for optimization.

## Classes and Methods

### `GradientBoostRegression`
Implements gradient boosting for regression.
```python
class GradientBoostRegression(BaseModel):
    def __init__(
        self,
        n_estimators=100,
        min_split_sample=5,
        max_depth=10,
        learning_rate=1e-3,
        tol=1e-6,
    ):
```
- `n_estimators`: Number of boosting iterations.
- `min_split_sample`: Minimum number of samples required to split a node.
- `max_depth`: Maximum depth of the decision trees.
- `learning_rate`: Controls the step size of updates.
- `tol`: Early stopping tolerance.

#### Methods:
- `forward(X, y)`: Trains the gradient boosting model on regression data.
- `predict(X)`: Returns regression predictions.

---
### `GradientBoostClassification`
Implements gradient boosting for multi-class classification.
```python
class GradientBoostClassification(BaseModel):
    def __init__(
        self,
        n_estimators=100,
        min_split_sample=5,
        max_depth=10,
        learning_rate=1e-3,
        tol=1e-6,
    ):
```
- `n_estimators`, `min_split_sample`, `max_depth`, `learning_rate`, `tol`: Similar to `GradientBoostRegression`.
- `n_classes`: Number of classes in the target variable.

#### Methods:
- `forward(X, y)`: Fits the model to classification data using softmax loss.
- `predict_proba(X)`: Returns class probabilities using softmax.
- `predict(X)`: Returns class predictions based on highest probability.

## Usage
### Training a Regressor:
```python
gb_regressor = GradientBoostRegression(n_estimators=50, learning_rate=0.1)
gb_regressor.forward(X_train, y_train)
preds = gb_regressor.predict(X_test)
```

### Training a Classifier:
```python
gb_classifier = GradientBoostClassification(n_estimators=50, learning_rate=0.1)
gb_classifier.forward(X_train, y_train)
preds = gb_classifier.predict(X_test)
```

## Notes
- The implementation does not support missing value handling or feature importance computation.
- Further optimizations such as parallelization and histogram-based splitting can improve efficiency.