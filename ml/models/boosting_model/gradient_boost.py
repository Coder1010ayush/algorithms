# ---------------------------------- utf-8 encoding ------------------------------
# this file contains the implementation for gradient boosting algorithm for regression and classification

import numpy as np
from models.decision_tree.decision_tree import DecisionTreeCART
from models.basemodel import BaseModel
from utils.activation import ActivationFunction
from utils.metrics import RegressionMetric, ClassificationMetric
from scipy.special import softmax


class GradientBoostRegression(BaseModel):
    def __init__(
        self,
        n_estimators: int = 100,
        min_split_sample: int = 5,
        max_depth: int = 10,
        learning_rate: float = 1e-3,
        tol: float = 1e-6,  # Early stopping tolerance
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.min_split_sample = min_split_sample
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.tol = tol
        self.trees = []
        self.initial_prediction = None

    def forward(self, x: np.ndarray, y: np.ndarray):

        # initialize the initial prediction used at time of prediciton
        self.initial_prediction = y.mean()
        out = np.full(y.shape, fill_value=self.initial_prediction, dtype=np.float32)

        for _ in range(self.n_estimators):
            residuals = y - out

            tree = DecisionTreeCART(
                min_sample_split=self.min_split_sample,
                max_depth=self.max_depth,
                task="regression",
            )
            tree.forward(x=x, y=residuals)
            y_pred = tree.predict(x)

            out += self.learning_rate * y_pred
            self.trees.append(tree)

            if np.abs(residuals).mean() < self.tol:
                break

    def predict(self, x: np.ndarray):
        out = np.full(
            shape=x.shape[0], fill_value=self.initial_prediction, dtype=np.float32
        )
        for tree in self.trees:
            out += self.learning_rate * tree.predict(x)
        return out


class GradientBoostClassification(BaseModel):
    def __init__(
        self,
        n_estimators: int = 100,
        min_split_sample: int = 5,
        max_depth: int = 10,
        learning_rate: float = 1e-3,
        tol: float = 1e-6,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.min_split_sample = min_split_sample
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.tol = tol
        self.trees = []
        self.initial_prediction = None
        self.n_classes = None

    def forward(self, x: np.ndarray, y: np.ndarray):

        # One-hot encode the labels
        self.n_classes = len(np.unique(y))
        y_one_hot = np.eye(self.n_classes)[y]

        # Initialize predictions with log priors
        class_priors = np.bincount(y) / len(y)
        self.initial_prediction = np.log(class_priors + 1e-6)
        out = np.tile(self.initial_prediction, (y.shape[0], 1))

        for _ in range(self.n_estimators):
            residuals = y_one_hot - softmax(out, axis=1)

            trees_per_class = []
            for c in range(self.n_classes):
                tree = DecisionTreeCART(
                    min_sample_split=self.min_split_sample,
                    max_depth=self.max_depth,
                    task="classification",
                )
                tree.forward(x=x, y=residuals[:, c])
                trees_per_class.append(tree)

            self.trees.append(trees_per_class)
            for c in range(self.n_classes):
                out[:, c] += self.learning_rate * trees_per_class[c].predict(x)

            if np.abs(residuals).mean() < self.tol:
                break

    def predict_proba(self, x: np.ndarray):
        out = np.tile(self.initial_prediction, (x.shape[0], 1))
        for trees_per_class in self.trees:
            for c in range(self.n_classes):
                out[:, c] += self.learning_rate * trees_per_class[c].predict(x)
        return softmax(out, axis=1)

    def predict(self, x: np.ndarray):
        return np.argmax(self.predict_proba(x), axis=1)
