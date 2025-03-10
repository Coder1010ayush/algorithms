# ------------------------ utf-8 encoding ----------------------------------------
import numpy as np


class Module:
    def __init__(self):
        self._parameters = {}  # Stores trainable parameters
        self._buffers = {}  # Stores non-trainable buffers
        self._modules = {}  # Stores submodules

    def forward(self, *inputs):
        """Override this method to define forward pass."""
        raise NotImplementedError

    def __call__(self, *inputs):
        """Allows calling model(input) directly."""
        return self.forward(*inputs)

    def register_parameter(self, name, param):
        """Registers a trainable parameter."""
        if param is not None:
            self._parameters[name] = param

    def register_buffer(self, name, buffer):
        """Registers a non-trainable buffer."""
        self._buffers[name] = buffer

    def add_module(self, name, module):
        """Adds a submodule."""
        if isinstance(module, Module):
            self._modules[name] = module

    def parameters(self):
        """Returns an iterator over trainable parameters."""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self):
        """Returns an iterator over named trainable parameters."""
        for name, param in self._parameters.items():
            yield name, param
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param

    def zero_grad(self):
        """Resets gradients for all parameters."""
        for param in self.parameters():
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)

    def buffers(self):
        """Returns an iterator over buffers."""
        for buffer in self._buffers.values():
            yield buffer
        for module in self._modules.values():
            yield from module.buffers()

    def named_buffers(self):
        """Returns an iterator over named buffers."""
        for name, buffer in self._buffers.items():
            yield name, buffer
        for module_name, module in self._modules.items():
            for name, buffer in module.named_buffers():
                yield f"{module_name}.{name}", buffer

    def modules(self):
        """Returns an iterator over submodules."""
        return self._modules.values()

    def named_modules(self):
        """Returns an iterator over named submodules."""
        for name, module in self._modules.items():
            yield name, module
            for sub_name, sub_module in module.named_modules():
                yield f"{name}.{sub_name}", sub_module
