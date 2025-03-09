# ---------------------------------------------- utf-8 encoding ------------------------------------
# This file contains the test cases for matrix multiplication
import numpy as np
from tests.autograd.matmul_c import compute_gradient_matmul
from tests.autograd.addition_c import compute_gradient_add
from tests.autograd.division_c import compute_gradient_division
from tests.autograd.multiply_c import compute_gradient_multiply
from tests.autograd.utils import compute_gradient_g

if __name__ == "__main__":

    # compute_gradient_add()
    # compute_gradient_matmul()
    # compute_gradient_division()
    # compute_gradient_multiply()
    compute_gradient_g()
