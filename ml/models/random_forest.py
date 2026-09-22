from multiprocessing import Pool
from typing import Literal

import numpy as np

from ml.base import BaseModel
from ml.models.tree import DecisionTreeCART, DecisionTreeRegression, majority


def _fit_tree(args):
    tree, X, y = args
    return tree.fit(X, y)


def _predict_tree(args):
    tree, X = args
    return tree.predict(X)


class RandomForest(BaseModel):
    def __init__(
        self,
        n_trees: int = 50,
        min_samples_split: int = 5,
        max_depth: int = 10,
        max_features: int | Literal["sqrt"] | None = "sqrt",
        task: Literal["classification", "regression"] = "classification",
        oob_score: bool = False,
        n_jobs: int | None = 1,
        random_state: int | None = None,
    ):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.task = task
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.rng = np.random.default_rng(random_state)
        self.trees: list = []
        self.oob_score_: float | None = None

    def _make_tree(self, n_features: int):
        mf = self.max_features
        if mf == "sqrt":
            mf = max(1, int(np.sqrt(n_features)))
        seed = int(self.rng.integers(2**31))
        if self.task == "classification":
            return DecisionTreeCART(self.min_samples_split, self.max_depth, task="classification", max_features=mf, random_state=seed)
        return DecisionTreeRegression(self.min_samples_split, self.max_depth, max_features=mf, random_state=seed)

    def _run(self, fn, jobs):
        if self.n_jobs == 1:
            return [fn(j) for j in jobs]
        with Pool(self.n_jobs) as pool:
            return pool.map(fn, jobs)

    def _aggregate(self, preds: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return np.array([majority(col) for col in preds.T])
        return preds.mean(axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        n = len(y)
        samples = [self.rng.choice(n, n, replace=True) for _ in range(self.n_trees)]
        jobs = [(self._make_tree(X.shape[1]), X[idx], y[idx]) for idx in samples]
        self.trees = self._run(_fit_tree, jobs)

        if self.oob_score:
            oob = [[] for _ in range(n)]
            for tree, idx in zip(self.trees, samples):
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                for i, p in zip(np.flatnonzero(mask), tree.predict(X[mask])):
                    oob[i].append(p)
            has = np.array([len(o) > 0 for o in oob])
            agg = np.array([self._aggregate(np.array(o)[:, None])[0] for o, h in zip(oob, has) if h])
            if self.task == "classification":
                self.oob_score_ = float(np.mean(agg == y[has]))
            else:
                self.oob_score_ = float(np.mean((agg - y[has]) ** 2))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        preds = np.array(self._run(_predict_tree, [(t, X) for t in self.trees]))
        return self._aggregate(preds)
