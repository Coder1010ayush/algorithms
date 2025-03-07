# --------------------* utf-8 encoding* ---------------
import numpy as np


def check_tensor_type(dta):
    if isinstance(dta, np.ndarray):
        return True
    if isinstance(dta, int) or isinstance(dta, float) or isinstance(dta, list):
        return True
    try:
        import cupy as cp

        if isinstance(dta, cp.ndarray):
            return True
    except ImportError:
        return False


def asTensor(x):
    if isinstance(x):
        return x
    from tensor import Tensor

    return Tensor(data=x)
