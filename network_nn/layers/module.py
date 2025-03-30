# ------------------------ utf-8 encoding ----------------------------------------
import numpy as np
from torch import Tensor


class Module:
    def __init__(self):
        self._parameters = {}  # Stores trainable parameters
        self._buffers = {}     # Stores non-trainable buffers
        self._modules = {}     # Stores submodules

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def __setattr__(self, name, value):
        """Automatically registers submodules as attributes."""
        # solves a major headache for me
        if isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def register_parameter(self, name, param):
        if param is not None:
            self._parameters[name] = param

    def register_buffer(self, name, buffer):
        self._buffers[name] = buffer

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self):
        for name, param in self._parameters.items():
            yield name, param
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param

    def buffers(self):
        for buffer in self._buffers.values():
            yield buffer
        for module in self._modules.values():
            yield from module.buffers()

    def named_buffers(self):
        for name, buffer in self._buffers.items():
            yield name, buffer
        for module_name, module in self._modules.items():
            for name, buffer in module.named_buffers():
                yield f"{module_name}.{name}", buffer

    def modules(self):
        return self._modules.values()

    def named_modules(self):
        for name, module in self._modules.items():
            yield name, module
            for sub_name, sub_module in module.named_modules():
                yield f"{name}.{sub_name}", sub_module

class Sequential(Module):
    """
    A general sequential class which will be used for any kind of layer that will be
    defined in nn module
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        for layer in self.layers:
            yield from layer.parameters()

    # def train(self):
    #     self.training = True
    #     for layer in self.layers:
    #         layer.train()

    # def eval(self):
    #     self.training = False
    #     for layer in self.layers:
    #         layer.eval()

    def __repr__(self) -> str:
        strg = ""
        for layer in self.layers:
            strg += repr(layer) + ",\n"
        return strg
