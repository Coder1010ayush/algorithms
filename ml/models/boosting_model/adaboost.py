# ------------------------- utf-8 encoding -------------------------------------
# this file contains implementation of adabosst from skretch for regression and classification also both
import numpy as np
import os
from models.basemodel import BaseModel
from models.decision_tree.decision_tree import DecisionTreeRegression, DecisionTreeCART


class AdaboostClassification(BaseModel):

    def __init__(
        self,
        n_estimators: int = 100,
        min_sample_split: int = 5,
        max_depth: int = 1,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.model_w = []
        self.trees = []

    def __get_error(self, weights: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        error_indices = y != y_pred
        return np.sum(weights[error_indices])

    def __update_data_point_weightage(
        self, old_w, current_model_w, y: np.ndarray, y_pred: np.ndarray
    ):
        new_w = np.where(
            y == y_pred,
            old_w * np.exp(-current_model_w),
            old_w * np.exp(current_model_w),
        )
        return new_w

    def __update_model_weights(self, error):
        epsilon = 1e-10
        return 0.5 * np.log((1 - error + epsilon) / (error + epsilon))

    def __get_h(self, normal_w: np.ndarray):
        upper_h = np.cumsum(normal_w)
        lower_h = upper_h - normal_w
        return upper_h, lower_h

    # def __get_upsampled_data(
    #     self, upper_h: np.ndarray, lower_h: np.ndarray, y: np.ndarray, x: np.ndarray
    # ):
    #     selected_index = []
    #     for i in range(y.shape[0]):
    #         num = np.random.random()
    #         if upper_h[i] > num and num > lower_h[i]:
    #             selected_index.append(i)
    #     return x[np.array(selected_index)], y[np.array(selected_index)]

    def __get_upsampled_data(self, normal_w: np.ndarray, y: np.ndarray, x: np.ndarray):
        indices = np.random.choice(np.arange(y.shape[0]), size=y.shape[0], p=normal_w)
        return x[indices], y[indices]

    def forward(self, x: np.ndarray, y: np.ndarray):

        # weight initialization (will be same for all the iterations)
        weights = np.ones(shape=y.shape, dtype=np.float32) / y.shape[0]
        for i in range(self.n_estimators):
            x_train, y_train = x, y
            tree = DecisionTreeCART(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                task="classification",
            )
            tree.forward(x_train, y_train)
            y_pred = tree.predict(x=x_train)
            error = self.__get_error(weights, y_train, y_pred)
            if error >= 0.5:
                break
            update_error = self.__update_model_weights(error=error)
            self.model_w.append(update_error)
            update_model_weight = self.__update_data_point_weightage(
                old_w=weights, current_model_w=update_error, y=y_train, y_pred=y_pred
            )
            normalized_w = update_model_weight / update_model_weight.sum()
            # this weight will use in upsampling the dataset
            # generate y.shape[0] time random integer number
            # upper_h, lower_h = self.__get_h(normalized_w)
            # x_sel, y_sel = self.__get_upsampled_data(upper_h, lower_h, y, x)
            x_sel, y_sel = self.__get_upsampled_data(normalized_w, y_train, x_train)
            x_train = x_sel
            y_train = y_sel
            self.trees.append(tree)

    def __predict_sample(self, x: np.ndarray):
        x = x.reshape((1, x.shape[0]))
        class_scores = {}
        for tree, weight in zip(self.trees, self.model_w):

            pred = tree.predict(x)
            if pred.item() not in class_scores:
                class_scores[pred.item()] = 0
            class_scores[pred.item()] += weight

        return max(class_scores, key=class_scores.get)

    def predict(self, x: np.ndarray):
        return np.array([self.__predict_sample(x_m) for x_m in x])


class AdaboostRegression(BaseModel):

    def __init__(
        self,
        n_estimators: int = 100,
        min_sample_split: int = 5,
        max_depth: int = 1,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.model_w = []
        self.trees = []

    def __get_error(self, weights: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        return np.sum(weights * (y - y_pred) ** 2) / np.sum(weights)

    def __update_data_point_weightage(
        self, old_w, current_model_w, y: np.ndarray, y_pred: np.ndarray
    ):
        new_w = np.where(
            y == y_pred,
            old_w * np.exp(-current_model_w),
            old_w * np.exp(current_model_w),
        )
        return new_w

    def __update_model_weights(self, error):
        epsilon = 1e-10
        return 0.5 * np.log((1 - error + epsilon) / (error + epsilon))

    def __get_h(self, normal_w: np.ndarray):
        upper_h = np.cumsum(normal_w)
        lower_h = upper_h - normal_w
        return upper_h, lower_h

    # def __get_upsampled_data(
    #     self, upper_h: np.ndarray, lower_h: np.ndarray, y: np.ndarray, x: np.ndarray
    # ):
    #     selected_index = []
    #     for i in range(y.shape[0]):
    #         num = np.random.random()
    #         if upper_h[i] > num and num > lower_h[i]:
    #             selected_index.append(i)
    #     return x[np.array(selected_index)], y[np.array(selected_index)]

    def __get_upsampled_data(self, normal_w: np.ndarray, y: np.ndarray, x: np.ndarray):
        indices = np.random.choice(np.arange(y.shape[0]), size=y.shape[0], p=normal_w)
        return x[indices], y[indices]

    def forward(self, x: np.ndarray, y: np.ndarray):

        # weight initialization (will be same for all the iterations)
        weights = np.ones(shape=y.shape, dtype=np.float32) / y.shape[0]
        for i in range(self.n_estimators):
            x_train, y_train = x, y
            tree = DecisionTreeCART(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                task="regression",
            )
            tree.forward(x_train, y_train)
            y_pred = tree.predict(x=x_train)
            error = self.__get_error(weights, y_train, y_pred)
            if error >= 0.8:
                break
            update_error = self.__update_model_weights(error=error)
            self.model_w.append(update_error)
            update_model_weight = self.__update_data_point_weightage(
                old_w=weights, current_model_w=update_error, y=y_train, y_pred=y_pred
            )
            normalized_w = update_model_weight / update_model_weight.sum()
            # this weight will use in upsampling the dataset
            # generate y.shape[0] time random integer number
            # upper_h, lower_h = self.__get_h(normalized_w)
            # x_sel, y_sel = self.__get_upsampled_data(upper_h, lower_h, y, x)
            x_sel, y_sel = self.__get_upsampled_data(normalized_w, y_train, x_train)
            x_train = x_sel
            y_train = y_sel
            self.trees.append(tree)

    def predict(self, x: np.ndarray):
        weighted_preds = np.zeros(x.shape[0])
        for tree, weight in zip(self.trees, self.model_w):
            weighted_preds += weight * tree.predict(x)
        return weighted_preds / np.sum(self.model_w)
