# --------------------------------- utf-8 encoding --------------------------------------
# this file contains testing code for convulation layer
# import all the essential modules which will be used in this file.
import os
import sys
import numpy as np
import pandas as pd 
import pathlib
from collections import Counter 
from tensor import Tensor
from init_w import Initializer
from layers.models import Conv1DLayer , Conv2DLayer , Conv3DLayer
from autograd.autodiff import relu
from layers.module import Module

class CustomCNN_One(Module):

    def __init__(self , num_channels : int = 3 , length : int = 256 ):
        super().__init__()
        self.l = length 
        self.n_c = num_channels

        self.conv_layer_1 = Conv1DLayer(in_channels=3 , out_channels=3 , kernel_size= 3, stride= 2, padding="same" )
        self.conv_layer_2 = Conv1DLayer(in_channels=3 , out_channels=3 , kernel_size= 3, stride= 2, padding= "same" )

    def forward(self , x ):
        out = self.conv_layer_1(x)
        out1 = self.conv_layer_2(out)
        return out1

class CustomCNN_Two(Module):

    def __init__(self , num_channels : int = 3 , length : int = 256 ):
        super().__init__()
        self.l = length 
        self.n_c = num_channels

        self.conv_layer_1 = Conv2DLayer(in_channels=3 , out_channels=3 , kernel_size= 3, stride= 2 , padding= "same" )
        self.conv_layer_2 = Conv2DLayer(in_channels=3 , out_channels=3 , kernel_size= 3, stride= 2 , padding= "same" )

    def forward(self , x ):
        out = self.conv_layer_1(x)
        out1 = self.conv_layer_2(out)
        return out1
    
class CustomCNN_Three(Module):

    def __init__(self , num_channels : int = 3 , length : int = 256 ):
        super().__init__()
        self.l = length 
        self.n_c = num_channels

        self.conv_layer_1 = Conv3DLayer(in_channels=3 , out_channels=3 , kernel_size= 3, stride= 2 , padding= "same" )
        # self.conv_layer_2 = Conv3DLayer(in_channels=3 , out_channels=3 , kernel_size= 3, stride= 2 , padding="same" )

    def forward(self , x ):
        out = self.conv_layer_1(x)
        # out1 = self.conv_layer_2(out)
        return out

if __name__ == "__main__":
    pass