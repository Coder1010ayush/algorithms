# Random forest (`ml/models/random_forest.py`)

`RandomForest` trains `n_trees` CART trees, each on a bootstrap sample of the data with `max_features` features considered per split, and aggregates by majority vote (classification) or mean (regression).

```python
from ml.models import RandomForest
rf = RandomForest(n_trees=50, max_depth=10, max_features="sqrt", task="classification",
                  oob_score=True, n_jobs=1, random_state=0).fit(X, y)
rf.predict(X_test)
rf.oob_score_   # accuracy (classification) or MSE (regression) on out-of-bag samples
```

- `max_features`: `"sqrt"` (default), an int, or `None` for all features.
- `n_jobs`: `1` runs serially; any other value uses a `multiprocessing.Pool` for fitting and prediction.
- `oob_score=True` evaluates each sample only with trees that did not see it.
