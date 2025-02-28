# ------------------------------------------- utf-8 encoding -----------------------------
# this file contains all kinds of Random Forest algorithm for regression and classification both
import numpy as np
import multiprocessing
from collections import Counter
from models.basemodel import BaseModel
from models.decision_tree.decision_tree import (
    DecisionTreeCART,
    DecisionTreeID3,
    DecisionTreeRegression,
)


class RandomForest(BaseModel):
    def __init__(
        self,
        n_trees=50,
        min_sample_split=5,
        max_depth=10,
        max_features=None,
        task="classification",
        oob_score=False,
        parallel=True,
    ):
        super().__init__()
        self.n_trees = n_trees
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.task = task
        self.oob_score = oob_score
        self.parallel = parallel
        self.trees = []
        self.oob_predictions = {}

    def _bootstrap_sample(self, X, y):
        n_samples = len(y)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        oob_indices = list(set(range(n_samples)) - set(indices))
        return X[indices], y[indices], oob_indices

    def _train_tree(self, args):
        X_sample, y_sample, _ = args
        if self.task == "classification":
            tree = DecisionTreeCART(
                self.min_sample_split, self.max_depth, task=self.task
            )
        else:
            tree = DecisionTreeRegression(self.min_sample_split, self.max_depth)
        tree.forward(X_sample, y_sample)
        return tree

    def forward(self, X, y):
        self.trees = []
        self.oob_predictions = (
            {i: [] for i in range(len(y))} if self.oob_score else None
        )

        tasks = [self._bootstrap_sample(X, y) for _ in range(self.n_trees)]

        if self.parallel:
            with multiprocessing.Pool() as pool:
                self.trees = pool.map(self._train_tree, tasks)
        else:
            self.trees = [self._train_tree(task) for task in tasks]

        if self.oob_score:
            for i, (_, _, oob_indices) in enumerate(tasks):
                for idx in oob_indices:
                    self.oob_predictions[idx].append(
                        self.trees[i].predict_sample(self.trees[i].tree, X[idx])
                    )

    def predict(self, X):
        if self.parallel:
            with multiprocessing.Pool() as pool:
                predictions = np.array(
                    pool.map(
                        lambda tree: [
                            tree.predict_sample(tree.tree, sample) for sample in X
                        ],
                        self.trees,
                    )
                )
        else:
            predictions = np.array(
                [
                    [tree.predict_sample(tree.tree, sample) for sample in X]
                    for tree in self.trees
                ]
            )

        if self.task == "classification":
            return np.array(
                [
                    Counter(predictions[:, i]).most_common(1)[0][0]
                    for i in range(X.shape[0])
                ]
            )
        else:
            return np.mean(predictions, axis=0)

    def compute_oob_score(self, y):
        if not self.oob_score:
            return None
        oob_predictions = np.array(
            [
                (
                    np.mean(self.oob_predictions[i])
                    if len(self.oob_predictions[i]) > 0
                    else np.nan
                )
                for i in range(len(y))
            ]
        )
        valid_idx = ~np.isnan(oob_predictions)
        if self.task == "classification":
            return np.mean(oob_predictions[valid_idx] == y[valid_idx])
        else:
            return np.mean((oob_predictions[valid_idx] - y[valid_idx]) ** 2)
