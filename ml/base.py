from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Common interface: ``fit`` learns from data, ``predict`` produces outputs."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).predict(X)
