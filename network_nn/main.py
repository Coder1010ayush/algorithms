# ---------------------------------------------- utf-8 encoding ------------------------------------
# This file contains the test cases for matrix multiplication
import numpy as np
from autograd.autodiff import relu
from tests.autograd_test.matmul_c import compute_gradient_matmul, create_tensor
from tests.autograd_test.addition_c import compute_gradient_add
from tests.autograd_test.division_c import compute_gradient_division
from tests.autograd_test.multiply_c import compute_gradient_multiply
from tests.autograd_test.utils import compute_gradient_g
from tests.autograd_test.conv_c import compute_gradient_conv
from layers.models import Linear

if __name__ == "__main__":

    # compute_gradient_add()
    # compute_gradient_matmul()
    # compute_gradient_division()
    # compute_gradient_multiply()
    # compute_gradient_g()
    # compute_gradient_conv()
    x = create_tensor(shape=(50, 2))
    model = Linear(in_feature=2, out_feature=1)
    z = model(x)
    out = relu(f=z)
    out.backprop()
    print(out)
