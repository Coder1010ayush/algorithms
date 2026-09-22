from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterator, Tuple

from network_nn.tensor import Parameter, Tensor


class Module:
    """Base class for all layers. Parameters and sub-modules are registered on attribute assignment."""

    def __init__(self):
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------ registration
    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        for registry in (self._parameters, self._modules, self._buffers):
            if name in registry and not isinstance(value, type(registry[name])):
                del registry[name]
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        for registry in ("_parameters", "_modules", "_buffers"):
            store = self.__dict__.get(registry, {})
            if name in store:
                return store[name]
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def add_module(self, name: str, module: "Module") -> None:
        setattr(self, name, module)

    # ---------------------------------------------------------------- traversal
    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            yield prefix + name, p
        for name, m in self._modules.items():
            yield from m.named_parameters(prefix + name + ".")

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        yield prefix.rstrip("."), self
        for name, m in self._modules.items():
            yield from m.named_modules(prefix + name + ".")

    def modules(self) -> Iterator["Module"]:
        for _, m in self.named_modules():
            yield m

    def children(self) -> Iterator["Module"]:
        return iter(self._modules.values())

    def named_buffers(self, prefix: str = "") -> Iterator[Tuple[str, Tensor]]:
        for name, b in self._buffers.items():
            yield prefix + name, b
        for name, m in self._modules.items():
            yield from m.named_buffers(prefix + name + ".")

    # ------------------------------------------------------------------- state
    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def state_dict(self) -> Dict[str, Tensor]:
        state = OrderedDict(self.named_parameters())
        state.update(self.named_buffers())
        return state

    def load_state_dict(self, state: Dict[str, Tensor]) -> None:
        own = self.state_dict()
        missing = set(own) - set(state)
        if missing:
            raise KeyError(f"Missing keys in state_dict: {sorted(missing)}")
        for key, tensor in own.items():
            tensor.data[...] = state[key].data if isinstance(state[key], Tensor) else state[key]

    # -------------------------------------------------------------------- repr
    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}({self.extra_repr()}"]
        children = [f"  ({name}): {repr(m).replace(chr(10), chr(10) + '  ')}" for name, m in self._modules.items()]
        if children:
            lines[0] = lines[0].rstrip(")") + ")" if not self.extra_repr() else lines[0]
            return lines[0].rstrip(")") + "\n" + "\n".join(children) + "\n)"
        return lines[0] + ")"


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)

    def forward(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())


class ModuleList(Module):
    def __init__(self, modules=()):
        super().__init__()
        for m in modules:
            self.append(m)

    def append(self, module: Module) -> "ModuleList":
        self.add_module(str(len(self._modules)), module)
        return self

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())
