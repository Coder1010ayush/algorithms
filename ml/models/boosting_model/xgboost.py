# ------------------------ utf-8 encoding -------------------------------
# this file contains simple implementation of xgboost algorithm

import numpy as np
from models.basemodel import BaseModel
from typing import List, Literal, Union
from utils.activation import ActivationFunction
from utils.metrics import RegressionMetric, ClassificationMetric


class XGTree:
    def __init__(
        self,
        is_leaf: bool = False,
        value: float = None,
        f_idx: int = None,
        threshold: float = None,
    ):
        self.f_index = f_idx
        self.value = value
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.left = None
        self.right = None


class TreeRegressor:
    def __init__(self, max_depth=6, min_samples_split=5, lambda_=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_sample_split = min_samples_split
        self.lamba_ = lambda_
        self.gamma = gamma

    def _compute_gradients(self, y, y_pred):
        """**golden heart hai**"""
        gradients = y_pred - y  # First derivative
        hessians = np.ones_like(y)  # Second derivative is 1 for squared loss
        return gradients, hessians

    def _get_best_split(self, X, y, gradients, hessians):
        best_split = {"score": -float("inf"), "f_idx": None, "thresh": None}

        for f_idx in range(X.shape[1]):
            x_column = X[:, f_idx]
            sorted_indices = np.argsort(x_column)
            x_sorted, g_sorted, h_sorted = (
                x_column[sorted_indices],
                gradients[sorted_indices],
                hessians[sorted_indices],
            )

            G_L, H_L = 0, 0
            G_total, H_total = np.sum(g_sorted), np.sum(h_sorted)

            for i in range(1, len(y)):
                G_L += g_sorted[i - 1]
                H_L += h_sorted[i - 1]
                G_R, H_R = G_total - G_L, H_total - H_L

                if x_sorted[i] == x_sorted[i - 1]:  # Skip duplicate threshold
                    continue

                gain = (
                    0.5
                    * (
                        (G_L**2) / (H_L + self.lambda_)
                        + (G_R**2) / (H_R + self.lambda_)
                        - (G_total**2) / (H_total + self.lambda_)
                    )
                    - self.gamma
                )  # Regularization penalty

                if gain > best_split["score"]:
                    best_split.update(
                        {"score": gain, "f_idx": f_idx, "thresh": x_sorted[i]}
                    )

        return best_split

    def _build_tree(self, X, y, gradients, hessians, depth=0):
        if len(y) < self.min_samples_split or depth >= self.max_depth:
            return XGTree(
                is_leaf=True,
                value=-np.sum(gradients) / (np.sum(hessians) + self.lambda_),
            )  # Leaf weight formula

        split = self._get_best_split(X, y, gradients, hessians)
        if split["f_idx"] is None:
            return XGTree(
                is_leaf=True,
                value=-np.sum(gradients) / (np.sum(hessians) + self.lambda_),
            )

        f_idx, thresh = split["f_idx"], split["thresh"]
        left_mask, right_mask = X[:, f_idx] < thresh, X[:, f_idx] >= thresh
        X_left, X_right, y_left, y_right = (
            X[left_mask],
            X[right_mask],
            y[left_mask],
            y[right_mask],
        )
        grad_left, grad_right, hess_left, hess_right = (
            gradients[left_mask],
            gradients[right_mask],
            hessians[left_mask],
            hessians[right_mask],
        )

        node = XGTree(f_index=f_idx, threshold=thresh)
        node.left = self._build_tree(X_left, y_left, grad_left, hess_left, depth + 1)
        node.right = self._build_tree(
            X_right, y_right, grad_right, hess_right, depth + 1
        )
        return node


class TreeClassifier:
    def __init__(
        self,
        n_classes,
        max_depth=6,
        min_samples_split=5,
        lambda_=1.0,
        gamma=0.0,
    ):
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_ = lambda_
        self.gamma = gamma

    def _softmax(self, logits):
        """Compute softmax probabilities."""
        exp_logits = np.exp(
            logits - np.max(logits, axis=1, keepdims=True)
        )  # Prevent overflow
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _compute_gradients(self, y_one_hot, y_pred):
        """Compute gradients and hessians for softmax loss."""
        """ **golden heart hai** """
        probs = self._softmax(y_pred)  # Convert logits to probabilities
        gradients = probs - y_one_hot  # First derivative
        hessians = probs * (1 - probs)  # Second derivative (diagonal of Hessian)
        return gradients, hessians

    def _get_best_split(self, X, y, gradients, hessians):
        best_split = {"score": -float("inf"), "f_idx": None, "thresh": None}

        for f_idx in range(X.shape[1]):
            x_column = X[:, f_idx]
            sorted_indices = np.argsort(x_column)
            x_sorted, g_sorted, h_sorted = (
                x_column[sorted_indices],
                gradients[sorted_indices],
                hessians[sorted_indices],
            )

            G_L, H_L = 0, 0
            G_total, H_total = np.sum(g_sorted), np.sum(h_sorted)

            for i in range(1, len(y)):
                G_L += g_sorted[i - 1]
                H_L += h_sorted[i - 1]
                G_R, H_R = G_total - G_L, H_total - H_L

                if x_sorted[i] == x_sorted[i - 1]:  # Skip duplicate threshold
                    continue

                gain = (
                    0.5
                    * (
                        (G_L**2) / (H_L + self.lambda_)
                        + (G_R**2) / (H_R + self.lambda_)
                        - (G_total**2) / (H_total + self.lambda_)
                    )
                    - self.gamma
                )  # Regularization penalty

                if gain > best_split["score"]:
                    best_split.update(
                        {"score": gain, "f_idx": f_idx, "thresh": x_sorted[i]}
                    )

        return best_split

    def _build_tree(self, X, y, gradients, hessians, depth=0):
        if len(y) < self.min_samples_split or depth >= self.max_depth:
            return XGTree(
                is_leaf=True,
                value=-np.sum(gradients) / (np.sum(hessians) + self.lambda_),
            )

        split = self._get_best_split(X, y, gradients, hessians)
        if split["f_idx"] is None:
            return XGTree(
                is_leaf=True,
                value=-np.sum(gradients) / (np.sum(hessians) + self.lambda_),
            )

        f_idx, thresh = split["f_idx"], split["thresh"]
        left_mask, right_mask = X[:, f_idx] < thresh, X[:, f_idx] >= thresh
        X_left, X_right, y_left, y_right = (
            X[left_mask],
            X[right_mask],
            y[left_mask],
            y[right_mask],
        )
        grad_left, grad_right, hess_left, hess_right = (
            gradients[left_mask],
            gradients[right_mask],
            hessians[left_mask],
            hessians[right_mask],
        )

        node = XGTree(f_index=f_idx, threshold=thresh)
        node.left = self._build_tree(X_left, y_left, grad_left, hess_left, depth + 1)
        node.right = self._build_tree(
            X_right, y_right, grad_right, hess_right, depth + 1
        )
        return node


class XgBoostModelRegressor(BaseModel):
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        lambda_=1.0,
        gamma=0.0,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_sample_split = min_samples_split
        self.lambda_ = lambda_
        self.gamma = gamma
        self.base_predictions = None

    def forward(self, X: np.ndarray, y: np.ndarray):
        self.base_prediction = np.mean(y)  # Initial prediction
        y_pred = np.full(y.shape, self.base_prediction)

        for _ in range(self.n_estimators):
            gradients, hessians = self._compute_gradients(y, y_pred)
            tree = self._build_tree(X, y, gradients, hessians)
            self.trees.append(tree)

            # Update predictions with learning rate
            y_pred += self.learning_rate * np.array(
                [self._predict_sample(tree, sample) for sample in X]
            )

    def _predict_sample(self, node, sample):
        if node.is_leaf:
            return node.value
        if sample[node.f_index] < node.threshold:
            return self._predict_sample(node.left, sample)
        else:
            return self._predict_sample(node.right, sample)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * np.array(
                [self._predict_sample(tree, sample) for sample in X]
            )
        return y_pred


class XgBoostModelClassifier(BaseModel):
    def __init__(
        self,
        n_classes,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        lambda_=1.0,
        gamma=0.0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_sample_split = min_samples_split
        self.lambda_ = lambda_
        self.gamma = gamma
        self.base_predictions = None

    def forward(self, X: np.ndarray, y: np.ndarray):
        y_one_hot = np.eye(self.n_classes)[y]  # One-hot encode labels

        self.base_predictions = np.full(
            (X.shape[0], self.n_classes), 1.0 / self.n_classes
        )  # Initialize with uniform probabilities
        y_pred = np.log(self.base_predictions)  # Convert to log-odds

        for _ in range(self.n_estimators):
            gradients, hessians = self._compute_gradients(y_one_hot, y_pred)

            for k in range(self.n_classes):
                tree = self._build_tree(X, y, gradients[:, k], hessians[:, k])
                self.trees[k].append(tree)

                # Update class predictions
                y_pred[:, k] += self.learning_rate * np.array(
                    [self._predict_sample(tree, sample) for sample in X]
                )

    def _predict_sample(self, node, sample):
        if node.is_leaf:
            return node.value
        if sample[node.f_index] < node.threshold:
            return self._predict_sample(node.left, sample)
        else:
            return self._predict_sample(node.right, sample)

    def predict_proba(self, X):
        y_pred = np.zeros((X.shape[0], self.n_classes))
        for k in range(self.n_classes):
            for tree in self.trees[k]:
                y_pred[:, k] += self.learning_rate * np.array(
                    [self._predict_sample(tree, sample) for sample in X]
                )
        return self._softmax(y_pred)

    def predict(self, X: np.ndarray):
        return np.argmax(self.predict_proba(X), axis=1)
