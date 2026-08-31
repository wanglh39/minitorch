"""containers：模块容器（Ch5）。

Sequential / ModuleList，通过 setattr 自动注册子模块。
对应真实 PyTorch 的 nn/modules/container.py。
"""

from __future__ import annotations

from collections.abc import Iterator

from .module import Module


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, str(i), m)

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


class ModuleList(Module):
    def __init__(self, modules: list[Module] | None = None):
        super().__init__()
        if modules is not None:
            for i, m in enumerate(modules):
                setattr(self, str(i), m)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x
