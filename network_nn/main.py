# ------------------------- *utf-8 encoding* ---------------------------
import os
import json
from tensor import Tensor, use_grad
import numpy as np


def custom_conv1d(input_seq, filters, stride=1, padding="valid"):
    batch_size, channels, input_len = input_seq.shape
    num_filters, _, filter_len = filters.shape

    if padding == "same":
        pad = (filter_len - 1) // 2
        input_seq = np.pad(input_seq, ((0, 0), (0, 0), (pad, pad)), mode="constant")
    elif padding != "valid":
        raise ValueError("Padding must be 'same' or 'valid'.")

    output_len = (input_seq.shape[2] - filter_len) // stride + 1
    output = np.zeros((batch_size, num_filters, output_len))

    for b in range(batch_size):
        for f in range(num_filters):
            for i in range(output_len):
                for c in range(channels):
                    segment = input_seq[b, c, i * stride : i * stride + filter_len]
                    output[b, f, i] += np.sum(segment * filters[f, c])

    return output


if __name__ == "__main__":
    # with use_grad():
    #     tensor1 = Tensor([[1, 2], [3, 4]], dtype=np.float32)
    #     tensor2 = Tensor([[1, 2], [3, 4]], dtype=np.float32)
    #     tensor3 = Tensor([[[1, 2]]], dtype=np.float32)

    #     out = tensor2 - tensor1
    #     out_f = out - tensor3
    #     out_f.backpropogate()
    #     print(tensor1.grad)
    #     print()
    #     print(tensor2.grad)
    #     print()
    #     print(out.grad)
    #     print()
    #     print(out_f.grad)
    #     print()
    #     print(tensor3.grad)

    input_seq = np.array([[[1, 2, 3, 4, 5], [0, 1, 0, 1, 0]]])
    filters = np.array([[[1, 0, -1], [0, 1, 0]], [[0, 1, 0], [1, 0, -1]]])

    output = custom_conv1d(input_seq, filters, stride=2, padding="same")
    print("Conv1D Output:\n", output)

    import torch
    import torch.nn.functional as F

    input_seq = torch.tensor([[[1, 2, 3, 4, 5], [0, 1, 0, 1, 0]]], dtype=torch.float32)
    filters = torch.tensor(
        [
            [[1, 0, -1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, -1]],
        ],
        dtype=torch.float32,
    )
    output = F.conv1d(input_seq, filters, stride=2, padding=1)
    print("PyTorch Conv1D Output:\n", output)
