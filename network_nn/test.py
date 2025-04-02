# -------------------------------------- utf-8 encoding -----------------------------------------
# this file contains kinds of layers test cases that are implemented here
import numpy as np
import pandas as pd 
import os 
import sys
import pathlib 
from autograd.autodiff import mse
from optimizer.optim import AdamOptimizer
from tensor import Tensor
from tests.model_tests.linear_unit_test import LinearModel 
from tests.model_tests.cnn_unit_test import CustomCNN_One , CustomCNN_Three , CustomCNN_Two
from init_w import Initializer

def test_linear():
    init_w = Initializer()
    x = np.linspace(-10, 10, 50).reshape(-1, 1)
    # noise = np.random.normal(0, 2, size=x.shape)
    y = (3 * x + 2).reshape(-1, 1)
    x = Tensor(data=x, retain_grad=True)
    y = Tensor(data=y, retain_grad=True)

    model = LinearModel(in_features=1)
    optimiser = AdamOptimizer(lr=0.001)
    for epoch in range(1):
        out = model(x)
        loss = mse(predictions=out, targets=y)
        loss.backprop()
        print(f"Loss at Epoch {epoch} is : {loss}")
        optimiser.step(params=model.parameters())

def test_cnn():
    init_w = Initializer()
    # input tensor with batch size ( b , num_channel , width , height )
    input_x = init_w.kaiming_normal(tensor_or_shape=( 2 , 3 , 256  ) )
    input_x_2 = init_w.kaiming_normal(tensor_or_shape=( 2 , 3 , 256 , 256 ))
    input_x_3 = init_w.kaiming_normal(tensor_or_shape=( 2 , 3 , 16 , 16 , 16))
    model_1d = CustomCNN_One(num_channels= 3, length=256 )
    model_2d = CustomCNN_Two(num_channels=3 )
    model_3d = CustomCNN_Three(num_channels=3)
    out_1d = model_1d(input_x)
    # print(f"Shape of output is {out_1d.shape}")
    # out_2d = model_2d(input_x_2)
    # print(f"Shape of output is {out_2d.shape}")
    out_3d = model_3d(input_x_3)
    print(f"Shape of output is {out_3d.shape}")


if __name__ == "__main__":
    # defining input data for this
    test_cnn()
