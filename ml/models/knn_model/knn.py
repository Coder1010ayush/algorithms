# ------------------------------------- utf-8 encoding --------------------------------------
# this file contains implementation of K Nearest Neighbour algorithm
from models.basemodel import BaseModel
import numpy as np
from typing import Literal
from utils.distant_matric import Distant
from collections import Counter


class KNearestNeighbour(BaseModel):

    def __init__(
        self,
        num_neighbours: int = 3,
        dist_func_type: Literal["eucledian", "manhattan"] = "eucledian",
        task: Literal["classification", "regression"] = "classification",
    ):
        super().__init__()
        self.num_neighbours = num_neighbours
        self.task = task
        self.dist_func_type = dist_func_type
        assert self.dist_func_type in {
            "eucledian",
            "manhattan",
        }, "Unsupported distance function is given"
        self.dist_func = Distant(method=self.dist_func_type)

    def forward(self, x: np.ndarray, y: np.ndarray):
        self.x_train = x
        self.y_train = y

    def __get_label(self, x: np.ndarray):
        distances = [self.dist_func.forward(x=x, y=x_t) for x_t in self.x_train]
        nearest_n = np.argsort(distances)[: self.num_neighbours]
        labels = [self.y_train[idx] for idx in nearest_n]
        return Counter(labels).most_common()[0][0]

    def __get_value(self, x: np.ndarray):
        distances = [self.dist_func.forward(x=x, y=x_t) for x_t in self.x_train]
        nearest_n = np.argsort(distances)[: self.num_neighbours]
        values = [self.y_train[idx] for idx in nearest_n]
        return np.mean(np.array(values))

    def predict(self, x_test: np.ndarray):
        if self.task == "classification":
            return np.array([self.__get_label(x) for x in x_test])

        elif self.task == "regression":
            return np.array([self.__get_value(x) for x in x_test])

        else:
            raise ValueError(f"Unsupported task value {self.task} is given")
