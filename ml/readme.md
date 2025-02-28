# Machine Learning from Scratch

## Overview
This project is focused on implementing fundamental machine learning algorithms from scratch without relying on high-level libraries like scikit-learn. The goal is to gain a deeper understanding of how these models work and build utility functions similar to scikit-learn for easy usability.

---

## Implemented Features
### 1. Optimization Techniques
- **Gradient Descent Variants**
  - Batch Gradient Descent
  - Stochastic Gradient Descent (SGD)
  - Mini-Batch Gradient Descent
- **Regularization Techniques**
  - L1 Regularization (Lasso)
  - L2 Regularization (Ridge)
  - Elastic Net Regularization (Combination of L1 & L2)
- **Gradient Calculation & Update Mechanisms**
  - Implemented gradient update logic with convergence checks
  - Supports custom learning rates, tolerances, and debugging modes

### 2. Linear Models
- **Linear Regression**
  - Implemented using gradient descent (batch and stochastic)
  - Supports different normalization techniques
  - Can accept different loss functions dynamically
- **Loss Functions Implemented**
  - Mean Squared Error (MSE)
  - Binary Cross-Entropy Loss (for future classification models)
  - Modular loss function support for flexibility

### 3. Utility Functions
- **Evaluation Metrics**
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R-squared Score
  - Precision, Recall, and F1 Score (to be used for classification models)
- **Data Normalization Techniques**
  - Min-Max Scaling
  - Standardization (Z-score normalization)

---

## Implemented Algorithm
### 1. Logistic Regression
- Implement logistic regression for binary classification
- Support gradient-based optimization with regularization
- Implement softmax for multi-class classification

### 2. Support Vector Machines (SVM)
- Implement primal SVM with gradient descent
- Explore kernelized SVM techniques

### 3. Decision Trees
- Implement decision tree training using information gain
- Support Gini impurity and entropy-based splitting criteria

### 4. Ensemble Methods
- Implement Random Forest using multiple decision trees
- Implement Gradient Boosting and AdaBoost

### 5. Neural Networks (Basic)
- Implement a simple feedforward neural network with backpropagation
- Support multiple activation functions like ReLU, Sigmoid, and Tanh

### 6. Model Evaluation & Hyperparameter Tuning
- Cross-validation techniques
- Grid search and random search for hyperparameter optimization

---

## Contributing
If you’d like to contribute to this project, feel free to fork the repository and submit a pull request with improvements or new implementations.

---