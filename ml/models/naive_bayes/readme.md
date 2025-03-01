# Naive Bayes Algorithm Implementation

## Overview
This document provides a detailed explanation of the Naive Bayes algorithm and its implementation. The Naive Bayes classifier is a probabilistic machine learning model used for classification tasks based on Bayes' Theorem. This implementation includes three variants:
- **Gaussian Naive Bayes** (for continuous data assuming normal distribution)
- **Multinomial Naive Bayes** (for text classification and discrete features)
- **Bernoulli Naive Bayes** (for binary/boolean features)

---

## Implementation Details

### Class: `NaiveBayes`
#### Constructor: `__init__()`
```python
class NaiveBayes(BaseModel):
    def __init__(self, model_type="gaussian", smoothing=1.0, parallel=False):
```
- `model_type`: Specifies the type of Naive Bayes model ("gaussian", "multinomial", or "bernoulli").
- `smoothing`: Laplace smoothing parameter (default: 1.0) to handle zero probabilities.
- `parallel`: Boolean flag to enable parallel processing for faster predictions.

#### Training: `forward(X, y)`
```python
def forward(self, X, y):
```
- `X`: Feature matrix (numpy array).
- `y`: Labels (numpy array).
- Computes class priors and likelihoods based on the chosen model type.

#### Gaussian Naive Bayes: `_fit_gaussian(X, y)`
```python
def _fit_gaussian(self, X, y):
```
- Computes mean and variance for each feature within each class.
- Assumes normal distribution for feature values.

#### Multinomial Naive Bayes: `_fit_multinomial(X, y)`
```python
def _fit_multinomial(self, X, y):
```
- Computes probability of each feature occurring in a given class.
- Suitable for text classification (e.g., word frequencies in documents).

#### Bernoulli Naive Bayes: `_fit_bernoulli(X, y)`
```python
def _fit_bernoulli(self, X, y):
```
- Computes probability of a feature being present (1) or absent (0) in a given class.
- Suitable for binary feature data.

#### Computing Posterior Probabilities: `_compute_posterior(sample)`
```python
def _compute_posterior(self, sample):
```
- Uses Bayes' Theorem to compute class probabilities for a given sample.
- Chooses the class with the highest posterior probability.

#### Prediction: `predict(X)`
```python
def predict(self, X):
```
- Takes a set of feature samples and predicts their class labels.
- If `parallel=True`, uses multiprocessing for faster computation.

---

---

## Summary
- This implementation provides a flexible and efficient Naive Bayes classifier supporting Gaussian, Multinomial, and Bernoulli models.
- It includes parallel computation for faster inference.
- Laplace smoothing is applied to handle zero probabilities.
