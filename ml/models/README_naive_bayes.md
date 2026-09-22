# Naive Bayes (`ml/models/naive_bayes.py`)

`NaiveBayes(model_type="gaussian", smoothing=1.0)` computes log class priors on `fit` and per-class likelihood parameters:

- `gaussian`: per-feature mean and variance, log-likelihood of a normal density.
- `multinomial`: Laplace-smoothed log feature probabilities, suited to count features.
- `bernoulli`: smoothed log probabilities of a feature being present or absent.

`predict` returns the class with the highest log posterior; `predict_proba` normalises the posteriors.

```python
from ml.models import NaiveBayes
nb = NaiveBayes("gaussian").fit(X, y)
nb.predict_proba(X_test)
```
