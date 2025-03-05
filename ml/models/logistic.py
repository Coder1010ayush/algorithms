# ------------------------- utf-8 encoding ----------------------------
# implementation logistic model
# this model is used for classification task

import numpy as np
from typing import Union, Literal
from optimizer.gradient_descent_batch import GradientDescentBatch
from optimizer.gradient_descent_stochastic import GradientDescentStochastic
from models.linear_utils import linear_derivative_function, linear_function
from utils.activation import ActivationFunction
from utils.metrics import ClassificationMetric


class LogisticModel:
    def __init__(
        self,
        optimizer=Literal["stochastic", "batch"],
        epochs=50,
        learning_rate=0.001,
        tolerance=1e-8,
        debug_mode="off",
        debug_step=10,
        norm="",
        l1_lambda=0.2,
        l2_lambda=0.1,
        el_param=0.9,
    ):
        self.coeff = None
        self.intercept = None
        self.loss_hist = None
        self.loss_func = ClassificationMetric(method="log_loss")
        self.sigmoid = ActivationFunction(activation_fnc="sigmoid")
        self.model_function = linear_function
        self.derivative_function = linear_derivative_function
        if optimizer == "batch":
            self.optimizer = GradientDescentBatch(
                epochs,
                learning_rate,
                tolerance,
                debug_mode,
                debug_step,
                norm,
                l1_lambda,
                l2_lambda,
                el_param,
            )
        elif optimizer == "stochastic":
            self.optimizer = GradientDescentStochastic(
                epochs,
                learning_rate,
                tolerance,
                debug_mode,
                debug_step,
                norm,
                l1_lambda,
                l2_lambda,
                el_param,
            )
        else:
            raise ValueError("Invalid optimizer choice. Use 'batch' or 'stochastic'.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.coeff, self.intercept, self.loss_hist = self.optimizer.forward(
            x_train=X,
            y_train=y,
            model_function=self.model_function,
            loss_function=self.loss_func.forward,
            derivative_function=self.derivative_function,
        )

    def predict(self, x: np.ndarray, threshold=0.5):
        out = self.model_function(x_val=x, coeff=self.coeff, intercept=self.intercept)
        out = self.sigmoid.forward(x=out)
        return (out >= threshold).astype(int)

    def show_loss(self):
        for idx, itm in enumerate(self.loss_hist):
            print(f"Epoch {idx} loss is : {itm}")
