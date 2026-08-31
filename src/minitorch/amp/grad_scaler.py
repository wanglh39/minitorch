"""GradScaler：梯度缩放器（Ch11）。

放大 loss 防小梯度下溢。反向后检查 inf/NaN 决定是否 skip step。
动态调整 scale。对应真实 PyTorch 的 amp/grad_scaler.py。
"""

from __future__ import annotations

import numpy as np


class GradScaler:
    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._found_inf = False
        self._growth_tracker = 0

    def get_scale(self) -> float:
        return self._scale

    def scale(self, loss):
        """放大 loss。"""
        from ..tensor import Tensor

        if isinstance(loss, Tensor):
            return Tensor.from_numpy(loss._numpy_view() * self._scale)
        return loss * self._scale

    def unscale_(self, optimizer) -> None:
        """把梯度除回 scale，并检查 inf/NaN。"""
        self._found_inf = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad._numpy_view()
                if np.any(np.isinf(grad)) or np.any(np.isnan(grad)):
                    self._found_inf = True
                p.grad._numpy_view()[:] = grad / self._scale

    def step(self, optimizer) -> None:
        """如果无 inf/NaN，执行 optimizer.step()。"""
        if not self._found_inf:
            optimizer.step()

    def update(self) -> None:
        """动态调整 scale。"""
        if self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
