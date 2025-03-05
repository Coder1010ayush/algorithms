# ------------------------------------- utf-8 encoding ---------------------------------
import os
import sys
import numpy as np
import zen


def check_tensor_type(dta):
    if isinstance(dta, np.ndarray):
        return True
    if isinstance(dta, int) or isinstance(dta, float):
        return True
    try:
        import cupy as cp

        if isinstance(dta, cp.ndarray):
            return True
    except ImportError:
        return False


def asTensor(x):
    if isinstance(x, zen.Tensor):
        return x
    return zen.Tensor(data=x)
