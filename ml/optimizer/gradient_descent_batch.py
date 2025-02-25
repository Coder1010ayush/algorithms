# -------------------- utf-8 encoding ----------------------------
# gradient calculation function is implemented for batched data
import math
import numpy as np


class GradientDescentBatch:

    def __init__(
        self,
        coeff_shape,
        epochs=50,
        learning_rate=1e-3,
        tolerence=1e-6,
        debug_mode="off",
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tol = tolerence
        self.debug_mode = debug_mode
        self.intercept = 0.0
        self.coeff = np.random.uniform(low=0.0, high=1.0, size=coeff_shape)
