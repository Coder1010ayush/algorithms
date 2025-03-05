import numpy as np
from zen import Tensor


def define_tensor():

    a = Tensor(data=np.array([1, 2, 3], dtype=float))
    slic = a[0:2]
    print(f"sliced part is {slic}")
    print(a)
