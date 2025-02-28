# Naive Bayes Algorithm - Implementation Summary

## Overview
Naive Bayes is a probabilistic classifier based on Bayes' Theorem with an assumption of independence among features. It is commonly used for classification tasks, especially in text classification, spam detection, and medical diagnosis. {Source : ChatGpt}

## Implemented Variants {Source : ChatGpt}
I have implemented the following variants of Naive Bayes:
1. **Gaussian Naive Bayes**: Assumes features follow a normal distribution.
2. **Multinomial Naive Bayes**: Suitable for text classification and count-based data.
3. **Bernoulli Naive Bayes**: Works well for binary/boolean feature vectors.

## Optimizations
1. **Parallelization**
   - Implemented an optional parameter to enable multiprocessing for faster computations.
   - Useful when dealing with large datasets.

2. **Vectorized Computation**
   - Leveraged NumPy for efficient probability calculations.
   - Reduced redundant computations for improved performance.

3. **Laplace Smoothing**
   - Added smoothing to prevent zero probability issues in categorical data.
   - Helps in handling unseen words/features in text classification.

4. **Efficient Probability Calculation**
   - Log probabilities are used to avoid numerical underflow.
   - Reduced redundant calculations by precomputing values where possible.

## Implementation Details
- Follows a modular approach to support different Naive Bayes variants.
- Provides flexibility with custom hyperparameters for smoothing and parallelization.
