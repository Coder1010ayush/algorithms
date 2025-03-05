# ------------------------- utf-8 encoding ----------------------------
# implementation linear model

import numpy as np
from typing import Literal
from optimizer.gradient_descent_batch import GradientDescentBatch
from optimizer.gradient_descent_stochastic import GradientDescentStochastic
from utils.metrics import RegressionMetric
from models.linear_utils import linear_derivative_function, linear_function
from models.basemodel import BaseModel


class Linear(BaseModel):

    def __init__(
        self,
        optimizer: Literal["stochastic", "batch"] = "stochastic",
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
        self.loss_func = RegressionMetric(method="mse")
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

    def forward(self, X, y):
        self.coeff, self.intercept, self.loss_hist = self.optimizer.forward(
            x_train=X,
            y_train=y,
            model_function=self.model_function,
            loss_function=self.loss_func.forward,
            derivative_function=self.derivative_function,
        )

    def predict(self, X):
        return self.model_function(X, self.coeff, self.intercept)

    def show_loss(self):
        for idx, itm in enumerate(self.loss_hist):
            print(f"Epoch {idx} loss is : {itm}")
