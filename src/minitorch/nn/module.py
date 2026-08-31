"""Module：神经网络模块基类（Ch5）。

__call__ 调 forward（留 hooks）；__setattr__ 拦截 Parameter/Module 自动注册到
_parameters/_modules；__getattr__ 从这些 dict 取出。state_dict 递归收集。
train/eval 递归切模式。对应真实 PyTorch 的 nn/modules/module.py。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from ..tensor import Tensor
from .parameter import Parameter


class Module:
    def __init__(self):
        self._parameters: dict[str, Parameter] = {}
        self._modules: dict[str, Module] = {}
        self._buffers: dict[str, Tensor] = {}
        self.training: bool = True
        self._forward_pre_hooks: dict[int, Callable] = {}
        self._forward_hooks: dict[int, Callable] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_parameters", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        params = self.__dict__.get("_parameters", {})
        if name in params:
            return params[name]
        modules = self.__dict__.get("_modules", {})
        if name in modules:
            return modules[name]
        buffers = self.__dict__.get("_buffers", {})
        if name in buffers:
            return buffers[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __call__(self, *args, **kwargs) -> Any:
        for hook in self._forward_pre_hooks.values():
            hook(self, args)
        result = self.forward(*args, **kwargs)
        for hook in self._forward_hooks.values():
            hook(self, args, result)
        return result

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def register_parameter(self, name: str, param: Parameter) -> None:
        self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        self._buffers[name] = tensor

    def parameters(self) -> Iterator[Parameter]:
        yield from self._parameters.values()
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            yield prefix + name, p
        for name, m in self._modules.items():
            yield from m.named_parameters(prefix + name + ".")

    def state_dict(self, prefix: str = "") -> dict[str, Tensor]:
        state: dict[str, Tensor] = {}
        for name, p in self._parameters.items():
            state[prefix + name] = p
        for name, b in self._buffers.items():
            state[prefix + name] = b
        for name, m in self._modules.items():
            state.update(m.state_dict(prefix + name + "."))
        return state

    def load_state_dict(self, state_dict: dict[str, Tensor], prefix: str = "") -> None:
        for name, p in self._parameters.items():
            key = prefix + name
            if key in state_dict:
                src = state_dict[key]
                p._storage._data[:] = src._numpy_view().ravel()
        for name, m in self._modules.items():
            m.load_state_dict(state_dict, prefix + name + ".")

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def register_forward_pre_hook(self, hook: Callable) -> None:
        self._forward_pre_hooks[id(hook)] = hook

    def register_forward_hook(self, hook: Callable) -> None:
        self._forward_hooks[id(hook)] = hook

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        for name, p in self._parameters.items():
            lines.append(f"  {name}: {p}")
        for name, m in self._modules.items():
            lines.append(f"  {name}: {m}")
        lines.append(")")
        return "\n".join(lines)
