# Gaussian mixture model (`ml/models/gmm.py`)

`GaussianMixtureModel(n_components=2, max_iter=100, tol=1e-6, init="kmeans", reg_covar=1e-6, random_state=None)` fits a mixture of full-covariance Gaussians with expectation-maximisation.

- E-step: responsibilities are computed in log space (Cholesky of each covariance, `logaddexp` normalisation) so small densities do not underflow.
- M-step: weights, means and covariances are the responsibility-weighted statistics; `reg_covar` is added to every covariance diagonal.
- Initialisation: `"kmeans"` (default) uses `ml.models.clustering.KMeans` for the means, `"random"` picks random samples.
- Stops when the log-likelihood changes by less than `tol`; `log_likelihood_` and `n_iter_` are stored.

```python
from ml.models import GaussianMixtureModel
gmm = GaussianMixtureModel(n_components=3, random_state=0).fit(X)
gmm.predict(X)          # hard assignments
gmm.predict_proba(X)    # responsibilities
```
