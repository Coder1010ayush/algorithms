# Boosting (`ml/models/boosting/`)

## AdaBoost (`adaboost.py`)

`AdaBoostClassifier(n_estimators=50, max_depth=1, min_samples_split=2, random_state=None)` implements SAMME: each round resamples the data by the current sample weights, fits a CART stump, computes the weighted error, derives the tree weight `alpha = log((1-err)/err) + log(k-1)` and up-weights misclassified samples. Works for any label set (binary or multi-class); `predict` returns the class with the largest summed `alpha`.

`AdaBoostRegressor(n_estimators=50, max_depth=3, ...)` implements AdaBoost.R2 with a weighted-median prediction.

## Gradient boosting (`gradient_boost.py`)

`GradientBoostRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=5)` starts from the target mean and fits a regression tree to the residuals each round.

`GradientBoostClassifier(...)` keeps one logit column per class, starts from log class priors and, each round, fits one **regression** tree per class to `onehot - softmax(logits)`. `predict_proba` re-applies the same trees; `rounds[i][c]` is the tree for class `c` in round `i`.

## XGBoost-style boosting (`xgboost.py`)

`XGTree(max_depth, min_samples_split, reg_lambda, gamma)` builds a tree from first- and second-order gradients: split gain is
`0.5 * (G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)) - γ` and each leaf weight is `-G/(H+λ)`. Gains for all thresholds of a feature are computed with cumulative sums.

`XGBoostRegressor` uses squared loss (`g = pred - y`, `h = 1`). `XGBoostClassifier` uses softmax loss (`g = p - onehot`, `h = p(1-p)`) with one tree per class per round, stored as `rounds[i][c]` so prediction applies the right tree to the right class.

```python
from ml.models import XGBoostClassifier
model = XGBoostClassifier(n_estimators=20, learning_rate=0.3, max_depth=3).fit(X, y)
model.predict_proba(X_test)
```
