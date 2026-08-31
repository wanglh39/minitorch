"""Optimizer：优化器基类（Ch6）。

持有 param_groups（每组含 params + lr + weight_decay 等），step() 更新参数，
zero_grad() 清梯度。Adam 的 exp_avg/exp_avg_sq 存优化器 state（非参数）。
对应真实 PyTorch 的 optim/optimizer.py。
"""

from __future__ import annotations

from collections.abc import Iterable


class Optimizer:
    def __init__(self, params: Iterable, defaults: dict):
        self.defaults = defaults
        self.param_groups: list[dict] = []
        params = list(params)
        if len(params) == 0:
            raise ValueError("optimizer got empty param list")
        if isinstance(params[0], dict):
            for group in params:
                self.param_groups.append({**defaults, **group})
        else:
            self.param_groups = [{"params": params, **defaults}]
        self.state: dict = {}

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None

    def step(self) -> None:
        raise NotImplementedError
