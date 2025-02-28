# ------------------------------ utf-8 encoding -------------------------------
# this file contains implementation of naive bayes algorithm with its varients

import numpy as np
from multiprocessing import Pool, cpu_count
from collections import defaultdict


class NaiveBayes:
    def __init__(self, model_type="gaussian", smoothing=1.0, parallel=False):
        self.model_type = model_type.lower()
        self.smoothing = smoothing
        self.parallel = parallel
        self.classes = None
        self.priors = None
        self.likelihoods = None

    def fit(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.priors = {
            cls: np.log(class_counts[i] / len(y)) for i, cls in enumerate(self.classes)
        }

        if self.model_type == "gaussian":
            self._fit_gaussian(X, y)
        elif self.model_type == "multinomial":
            self._fit_multinomial(X, y)
        elif self.model_type == "bernoulli":
            self._fit_bernoulli(X, y)
        else:
            raise ValueError(
                "Unsupported model type. Choose from 'gaussian', 'multinomial', or 'bernoulli'"
            )

    def _fit_gaussian(self, X, y):
        self.likelihoods = {}
        for cls in self.classes:
            X_cls = X[y == cls]
            mean = np.mean(X_cls, axis=0)
            var = np.var(X_cls, axis=0) + self.smoothing
            self.likelihoods[cls] = (mean, var)

    def _fit_multinomial(self, X, y):
        self.likelihoods = {}
        for cls in self.classes:
            X_cls = X[y == cls]
            word_counts = np.sum(X_cls, axis=0) + self.smoothing
            self.likelihoods[cls] = np.log(word_counts / np.sum(word_counts))

    def _fit_bernoulli(self, X, y):
        self.likelihoods = {}
        for cls in self.classes:
            X_cls = X[y == cls]
            prob = (np.sum(X_cls, axis=0) + self.smoothing) / (
                X_cls.shape[0] + 2 * self.smoothing
            )
            self.likelihoods[cls] = np.log(prob), np.log(1 - prob)

    def _compute_posterior(self, sample):
        posteriors = {}
        for cls in self.classes:
            posterior = self.priors[cls]
            if self.model_type == "gaussian":  # normal distribution assumption
                mean, var = self.likelihoods[cls]
                likelihood = -0.5 * np.sum(
                    ((sample - mean) ** 2) / var + np.log(2 * np.pi * var)
                )
            elif self.model_type == "multinomial":
                likelihood = np.sum(sample * self.likelihoods[cls])
            elif self.model_type == "bernoulli":  # burnauli distribution assumption
                prob_1, prob_0 = self.likelihoods[cls]
                likelihood = np.sum(sample * prob_1 + (1 - sample) * prob_0)
            posterior += likelihood
            posteriors[cls] = posterior
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        if self.parallel:
            with Pool(cpu_count()) as pool:
                return np.array(pool.map(self._compute_posterior, X))
        else:
            return np.array([self._compute_posterior(sample) for sample in X])
