# Decision trees (`ml/models/tree.py`)

All trees share `_Tree`, which grows a binary tree greedily: for every feature and every distinct threshold it scores the weighted impurity of the two children and keeps the lowest. Growth stops at `max_depth`, when fewer than `min_samples_split` samples remain, or when a node is pure.

| Class | Impurity | Leaf value | Use |
|---|---|---|---|
| `DecisionTreeID3` | entropy (information gain) | majority class | classification |
| `DecisionTreeCART(task="classification")` | Gini | majority class | classification |
| `DecisionTreeCART(task="regression")` | variance | mean | regression |
| `DecisionTreeRegression` | variance | mean | regression |

Constructor: `(min_samples_split=5, max_depth=10, max_features=None, random_state=None)`. `max_features` subsamples the features tried at each split (used by random forests).

```python
from ml.models import DecisionTreeCART
tree = DecisionTreeCART(max_depth=5).fit(X, y)
tree.predict(X_test)
tree.root        # Node(feature, threshold, left, right, value)
```

Impurity helpers `gini`, `entropy`, `variance` and `majority` are module-level functions.
