"""_C：算子分发路由表（Ch2 / Ch8）。

阶段一：CPU 单后端，Python dict 路由。
阶段二：C++ 注册的 kernel 通过 pybind11 暴露到同一接口。
对应真实 PyTorch 的 DispatchKey + kernel 函数指针。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_dispatch_table: dict[tuple[str, str], Callable] = {}


def register(op_name: str, kernel: Callable, device: str = "cpu") -> None:
    _dispatch_table[(op_name, device)] = kernel


def dispatch(op_name: str, *args: Any, device: str = "cpu", **kwargs: Any) -> Any:
    key = (op_name, device)
    if key not in _dispatch_table:
        raise RuntimeError(f"no kernel registered for op '{op_name}' on device '{device}'")
    return _dispatch_table[key](*args, **kwargs)


def has_kernel(op_name: str, device: str = "cpu") -> bool:
    return (op_name, device) in _dispatch_table
