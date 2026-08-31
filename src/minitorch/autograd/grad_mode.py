"""grad_mode：梯度计算模式上下文（Ch3 基础 / Ch4 扩展）。

no_grad/enable_grad 上下文管理器，控制 Function.apply 是否建图。
反向传播在 no_grad 下执行，避免梯度运算二次建图。
对应真实 PyTorch 的 csrc/autograd/grad_mode.h。
"""

from __future__ import annotations

_grad_enabled: bool = True


def is_grad_enabled() -> bool:
    return _grad_enabled


class no_grad:
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._prev


class enable_grad:
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = True
        return self

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._prev
