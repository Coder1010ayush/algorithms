import numpy as np


class Module:
    def __init__(self):
        self._parameters = {}
        self._buffers = {}
        self._modules = {}

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def register_parameter(self, name, value):
        if value is not None:
            self._parameters[name] = {"value": value, "grad": np.zeros_like(value)}

    def register_buffer(self, name, value):
        self._buffers[name] = value

    def add_module(self, name, module):
        if isinstance(module, Module):
            self._modules[name] = module

    def get_parameter(self, name):
        return self._parameters.get(name, None)

    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def named_parameters(self):
        for name, param in self._parameters.items():
            yield name, param
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param

    def buffers(self):
        buffers = list(self._buffers.values())
        for module in self._modules.values():
            buffers.extend(module.buffers())
        return buffers

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

    def zero_grad(self):
        for param in self.parameters():
            if param["grad"] is not None:
                param["grad"] = np.zeros_like(param["grad"])
