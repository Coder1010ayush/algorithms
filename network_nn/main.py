# ---------------------------------------------- utf-8 encoding ------------------------------------
# This file contains the test cases for matrix multiplication
import numpy as np


if __name__ == "__main__":
    from tests.autograd.matmul_c import compute_gradient

    compute_gradient()
    from tests.autograd.matmul import compute_gradient

    compute_gradient()
