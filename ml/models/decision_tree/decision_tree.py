# ------------------------- utf-8 encoding --------------------------
# This file contains decision tree algorithm implementation (ID3, CART, CHAID)

import numpy as np
import math
from models.basemodel import BaseModel


class Tree:
    def __init__(
        self, f_index=None, threshold=None, is_leaf=False, label=None, value=None
    ):
        self.f_index = f_index
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.label = label
        self.left = None
        self.right = None
        self.value = value


class DecisionTreeID3(BaseModel):
    def __init__(self, min_sample_split: int = 5, max_depth: int = 10):
        super().__init__()
        self.num_features = -1
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.tree = None
        self.classes = []

    def __get_best_split_criterion(self, x, y):
        splitting_val = {"gain": float("-inf"), "f_idx": None, "thresh": None}

        for f_idx in range(self.num_features):
            x_loc = [x_loc[f_idx] for x_loc in x]
            for element in sorted(set(x_loc)):
                y_left = [yi for xi, yi in zip(x_loc, y) if xi < element]
                y_right = [yi for xi, yi in zip(x_loc, y) if xi >= element]
                if len(y_left) > 0 and len(y_right) > 0:
                    current_gain = self.calculate_gain(y, y_left, y_right)
                    if current_gain > splitting_val["gain"]:
                        splitting_val.update(
                            {"gain": current_gain, "f_idx": f_idx, "thresh": element}
                        )
        return splitting_val

    def gini_index(self, y):
        gini = 0
        for cls in self.classes:
            count = sum(1 for yi in y if yi == cls)
            prob = count / len(y)
            gini += prob**2
        return 1 - gini

    def calculate_gain(self, y_parent, y_left, y_right):
        gini_parent = self.gini_index(y_parent)
        gini_left = self.gini_index(y_left)
        gini_right = self.gini_index(y_right)
        weight_left = len(y_left) / len(y_parent)
        weight_right = len(y_right) / len(y_parent)
        return gini_parent - (weight_left * gini_left + weight_right * gini_right)

    def __build_decision_tree(self, x, y, depth=0):

        if len(y) < self.min_sample_split or depth >= self.max_depth:
            label = max(set(y), key=y.count)
            leaft_node = Tree(is_leaf=True, f_index=-1, threshold=-1, label=label)
            # leaft_node.positive_proba = len([yi for yi in y if yi > 0]) / len(y)
            # leaft_node.negative_proba = 1 - leaft_node.positive_proba
            return leaft_node

        split_val = self.__get_best_split_criterion(x, y)
        f_idx, thresh = split_val["f_idx"], split_val["thresh"]
        if f_idx is None:
            return Tree(is_leaf=True, label=max(set(y), key=y.count))

        x_left, x_right, y_left, y_right = [], [], [], []
        for x_loc, y_loc in zip(x, y):
            if x_loc[f_idx] >= thresh:
                x_right.append(x_loc)
                y_right.append(y_loc)
            else:
                x_left.append(x_loc)
                y_left.append(y_loc)

        node = Tree(f_index=f_idx, threshold=thresh)
        node.left = self.__build_decision_tree(x_left, y_left, depth + 1)
        node.right = self.__build_decision_tree(x_right, y_right, depth + 1)
        return node

    def forward(self, x, y):
        self.num_features = x.shape[1]
        self.classes = set(y)
        self.tree = self.__build_decision_tree(x, y)

    def predict_sample(self, node, sample):
        if node.is_leaf:
            return node.label

        if sample[node.f_index] < node.threshold:
            return self.predict_sample(node.left, sample)
        else:
            return self.predict_sample(node.right, sample)

    def predict_helper_alternative(self, node, x):
        if node.is_leaf:
            return 1 if node.positive_proba > node.negative_proba else -1
        f_idx = node.f_index
        thresh = node.threshold
        if x[f_idx] >= thresh:
            return self.predict_helper_alternative(node=node.right, x=x)
        return self.predict_helper_alternative(node=node.left, x=x)

    def predict(self, x):
        """Predict for multiple samples."""
        op1 = np.array([self.predict_sample(self.tree, sample) for sample in x])
        # op2 = np.array(
        #     [self.predict_helper_alternative(self.tree, sample) for sample in x]
        # )
        return op1

    def compute_loss(self, y, y_pred):
        pass

    def compute_gradient(self, X, y, y_pred):
        pass

    def update_parameters(self, grad_w, grad_b, lr):
        pass


class DecisionTreeCART(BaseModel):
    def __init__(self, min_sample_split=5, max_depth=10, task="classification"):
        super().__init__()
        self.num_features = -1
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.tree = None
        self.classes = []
        self.task = task

    def __get_best_split_criterion(self, x, y):
        best_split = {"score": float("inf"), "f_idx": None, "thresh": None}

        for f_idx in range(self.num_features):
            x_column = [row[f_idx] for row in x]
            for threshold in sorted(set(x_column)):
                y_left = [yi for xi, yi in zip(x_column, y) if xi < threshold]
                y_right = [yi for xi, yi in zip(x_column, y) if xi >= threshold]

                if len(y_left) > 0 and len(y_right) > 0:
                    if self.task == "classification":
                        score = self.gini_index(y_left, y_right)
                    else:
                        score = self.mse_index(y_left, y_right)

                    if score < best_split["score"]:  # Minimize impurity/MSE
                        best_split.update(
                            {"score": score, "f_idx": f_idx, "thresh": threshold}
                        )

        return best_split

    def gini_index(self, y_left, y_right):
        """Gini impurity reduction."""

        def gini(y):
            classes = set(y)
            return 1 - sum((y.count(cls) / len(y)) ** 2 for cls in classes)

        weight_left = len(y_left) / (len(y_left) + len(y_right))
        weight_right = len(y_right) / (len(y_left) + len(y_right))
        return weight_left * gini(y_left) + weight_right * gini(y_right)

    def mse_index(self, y_left, y_right):
        """Mean Squared Error (MSE) reduction for regression."""

        def mse(y):
            if len(y) == 0:
                return 0
            mean_y = sum(y) / len(y)
            return sum((yi - mean_y) ** 2 for yi in y) / len(y)

        weight_left = len(y_left) / (len(y_left) + len(y_right))
        weight_right = len(y_right) / (len(y_left) + len(y_right))
        return weight_left * mse(y_left) + weight_right * mse(y_right)

    def __build_decision_tree(self, x, y, depth=0):
        if len(y) < self.min_sample_split or depth >= self.max_depth:
            if self.task == "classification":
                label = max(set(y), key=y.count)
                return Tree(is_leaf=True, label=label)
            else:
                value = sum(y) / len(y)
                return Tree(is_leaf=True, value=value)

        split_val = self.__get_best_split_criterion(x, y)
        f_idx, thresh = split_val["f_idx"], split_val["thresh"]

        if f_idx is None:
            if self.task == "classification":
                return Tree(is_leaf=True, label=max(set(y), key=y.count))
            else:
                return Tree(is_leaf=True, value=sum(y) / len(y))

        x_left, x_right, y_left, y_right = [], [], [], []
        for x_row, y_val in zip(x, y):
            if x_row[f_idx] < thresh:
                x_left.append(x_row)
                y_left.append(y_val)
            else:
                x_right.append(x_row)
                y_right.append(y_val)

        node = Tree(f_index=f_idx, threshold=thresh)
        node.left = self.__build_decision_tree(x_left, y_left, depth + 1)
        node.right = self.__build_decision_tree(x_right, y_right, depth + 1)
        return node

    def forward(self, x, y):
        self.num_features = x.shape[1]
        if self.task == "classification":
            self.classes = set(y)
        self.tree = self.__build_decision_tree(x, y)

    def predict_sample(self, node, sample):
        if node.is_leaf:
            return node.label if self.task == "classification" else node.value

        if sample[node.f_index] < node.threshold:
            return self.predict_sample(node.left, sample)
        else:
            return self.predict_sample(node.right, sample)

    def predict(self, x):
        return np.array([self.predict_sample(self.tree, sample) for sample in x])

    def compute_loss(self, y, y_pred):
        pass

    def compute_gradient(self, X, y, y_pred):
        pass

    def update_parameters(self, grad_w, grad_b, lr):
        pass


class DecisionTreeRegression(BaseModel):
    def __init__(self, min_sample_split: int = 5, max_depth: int = 10):
        super().__init__()
        self.num_features = -1
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.tree = None

    def __get_best_split_criterion(self, x, y):
        splitting_val = {"mse": float("inf"), "f_idx": None, "thresh": None}

        for f_idx in range(self.num_features):
            x_loc = [x_loc[f_idx] for x_loc in x]
            for element in sorted(set(x_loc)):
                y_left = [yi for xi, yi in zip(x_loc, y) if xi < element]
                y_right = [yi for xi, yi in zip(x_loc, y) if xi >= element]
                if len(y_left) > 0 and len(y_right) > 0:
                    current_mse = self.calculate_mse(y_left, y_right)
                    if current_mse < splitting_val["mse"]:
                        splitting_val.update(
                            {"mse": current_mse, "f_idx": f_idx, "thresh": element}
                        )
        return splitting_val

    def calculate_mse(self, y_left, y_right):
        def mse(y):
            if len(y) == 0:
                return 0
            mean_y = np.mean(y)
            return np.mean((y - mean_y) ** 2)

        mse_left = mse(y_left)
        mse_right = mse(y_right)
        weight_left = len(y_left) / (len(y_left) + len(y_right))
        weight_right = 1 - weight_left
        return weight_left * mse_left + weight_right * mse_right

    def __build_regression_tree(self, x, y, depth=0):
        if len(y) < self.min_sample_split or depth >= self.max_depth:
            return Tree(is_leaf=True, value=np.mean(y))

        split_val = self.__get_best_split_criterion(x, y)
        f_idx, thresh = split_val["f_idx"], split_val["thresh"]
        if f_idx is None:
            return Tree(is_leaf=True, value=np.mean(y))

        x_left, x_right, y_left, y_right = [], [], [], []
        for x_loc, y_loc in zip(x, y):
            if x_loc[f_idx] >= thresh:
                x_right.append(x_loc)
                y_right.append(y_loc)
            else:
                x_left.append(x_loc)
                y_left.append(y_loc)

        node = Tree(f_index=f_idx, threshold=thresh)
        node.left = self.__build_regression_tree(x_left, y_left, depth + 1)
        node.right = self.__build_regression_tree(x_right, y_right, depth + 1)
        return node

    def forward(self, x, y):
        self.num_features = x.shape[1]
        self.tree = self.__build_regression_tree(x, y)

    def predict_sample(self, node, sample):
        if node.is_leaf:
            return node.value
        if sample[node.f_index] < node.threshold:
            return self.predict_sample(node.left, sample)
        return self.predict_sample(node.right, sample)

    def predict(self, x):
        return np.array([self.predict_sample(self.tree, sample) for sample in x])

    def compute_loss(self, y, y_pred):
        pass

    def compute_gradient(self, X, y, y_pred):
        pass

    def update_parameters(self, grad_w, grad_b, lr):
        pass
