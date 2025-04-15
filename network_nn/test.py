# -------------------------------------- utf-8 encoding -----------------------------------------
# this file contains kinds of layers test cases that are implemented here
# test like tom and jerry
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
from layers.models import *
def test_stack_operation():
    import numpy as np
    from tensor import Tensor
    from autograd.autodiff import stack  

    # Test Case 1: Stacking along axis 0
    t1 = Tensor(np.array([1, 2, 3] , dtype=float), retain_grad=True)
    t2 = Tensor(np.array([4, 5, 6] , dtype=float), retain_grad=True)
    t3 = Tensor(np.array([7, 8, 9] , dtype=float), retain_grad=True)
    stacked_tensor = stack([t1, t2, t3], axis=0)

    expected_output_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]] , dtype=float)
    assert np.array_equal(stacked_tensor.data, expected_output_1), "Test Case 1 Failed"

    # Test Case 2: Stacking along axis 1
    t4 = Tensor(np.array([[1, 2], [3, 4]] , dtype=float), retain_grad=True)
    t5 = Tensor(np.array([[5, 6], [7, 8]] , dtype=float), retain_grad=True)
    stacked_tensor_2 = stack([t4, t5], axis=1)

    expected_output_2 = np.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]]] , dtype=float)
    assert np.array_equal(stacked_tensor_2.data, expected_output_2), "Test Case 2 Failed"

    # Test Case 3: Backward Pass for axis 0
    grad = np.ones_like(stacked_tensor.data)
    stacked_tensor.grad = grad 
    stacked_tensor.backprop()

    expected_grad_1 = np.ones_like(t1.data)
    assert np.array_equal(t1.grad, expected_grad_1), "Gradient mismatch for t1"
    assert np.array_equal(t2.grad, expected_grad_1), "Gradient mismatch for t2"
    assert np.array_equal(t3.grad, expected_grad_1), "Gradient mismatch for t3"

    # Test Case 4: Backward Pass for axis 1
    grad_2 = np.ones_like(stacked_tensor_2.data)
    stacked_tensor_2.grad = grad_2
    stacked_tensor_2.backprop()

    expected_grad_2 = np.ones_like(t4.data)
    assert np.array_equal(t4.grad, expected_grad_2), "Gradient mismatch for t4"
    assert np.array_equal(t5.grad, expected_grad_2), "Gradient mismatch for t5"

    print("All test cases passed!")

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

def test_rnn():
    rnn = RNN(input_size=10, hidden_size=20, num_layers=3)
    x = Tensor(np.random.randn(5, 8, 10) ,retain_grad=True)  
    output, h_n = rnn(x)

    print(output.shape) 
    print(h_n.shape) 

def test_gru():
    rnn = GRULayer(input_size=10, hidden_size=20, num_layers=3)
    x = Tensor(np.random.randn(5, 8, 10) ,retain_grad=True)  
    output, h_n = rnn(x)

    print(output.shape) 
    print(h_n.shape) 

def test_lstm():
    batch_size, seq_len, input_size, hidden_size, num_layers = 32, 10, 64, 128, 2
    x = Tensor(data= np.random.rand(batch_size, seq_len, input_size)  ,retain_grad=True)
    h0 = [Tensor(data= np.random.rand(batch_size, hidden_size) ,retain_grad=True) for _ in range(num_layers)]
    c0 = [Tensor(data= np.random.rand(batch_size, hidden_size) ,retain_grad=True) for _ in range(num_layers)]

    lstm_layer = LSTMLayer(input_size, hidden_size, num_layers)
    output, (h_n, c_n) = lstm_layer.forward(x, [h0, c0])
    print("---------------------------------- done -------------------------------------")

if __name__ == "__main__":
    # defining input data for this
    # test_cnn()

    # test_rnn() 
    # test_gru()
    # x = Tensor(np.random.randn(5, 8, 10) ,retain_grad=True)  
    # print(x[: , 1 , :])   
    # test_lstm()

    import torch
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    mask = torch.tensor([[False, True], [True, False]])
    out = torch.masked_fill(x, mask, -1)
    print(out)
