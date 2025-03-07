# ------------------------- *utf-8 encoding* ---------------------------
import os
import json
from tensor import Tensor, use_grad
import numpy as np

if __name__ == "__main__":
    with use_grad():
        tensor1 = Tensor([[1, 2], [3, 4]], dtype=np.float32)
        tensor2 = Tensor([[1, 2], [3, 4]], dtype=np.float32)
        tensor3 = Tensor([[1, 2]], dtype=np.float32)

        out = tensor1 + tensor2
        out_f = out + tensor3
        out_f.backpropogate()
        print(tensor1.grad)
        print()
        print(tensor2.grad)
        print()
        print(out.grad)
        print()
        print(out_f.grad)
        print()
        print(tensor3.grad)
