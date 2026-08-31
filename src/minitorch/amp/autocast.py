"""autocast：自动混合精度上下文（Ch11）。

上下文内前向自动转 fp16。matmul/conv 转 fp16，reduction 保留 fp32。
教学版用 numpy float16 模拟，主要讲原理。
对应真实 PyTorch 的 amp/autocast_mode.py。
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor


class Autocast:
    def __init__(self, enabled: bool = True, dtype=np.float16):
        self.enabled = enabled
        self.dtype = dtype
        self._prev_enabled = False

    def __enter__(self):
        self._prev_enabled = _autocast_enabled.global_enabled
        if self.enabled:
            _autocast_enabled.global_enabled = True
            _autocast_enabled.global_dtype = self.dtype
        return self

    def __exit__(self, *args):
        _autocast_enabled.global_enabled = self._prev_enabled


class _AutocastState:
    global_enabled: bool = False
    global_dtype = np.float16


_autocast_enabled = _AutocastState()


def is_autocast_enabled() -> bool:
    return _autocast_enabled.global_enabled


def get_autocast_dtype():
    return _autocast_enabled.global_dtype


def autocast_tensor(t: Tensor) -> Tensor:
    """如果 autocast 开启，把 tensor 转为 autocast dtype。"""
    if not is_autocast_enabled():
        return t
    arr = t._numpy_view().astype(get_autocast_dtype())
    return Tensor.from_numpy(arr)
