"""lr_scheduler：学习率调度器（Ch6）。

每个 epoch（或 step）后调用 scheduler.step()，根据策略调整 optimizer
中每个 param_group 的 lr。

对应真实 PyTorch 的 optim/lr_scheduler.py。
"""

from __future__ import annotations

import math


class LRScheduler:
    def __init__(self, optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self._initial_lr = [g["lr"] for g in optimizer.param_groups]
        self.step()

    def get_lr(self) -> list[float]:
        raise NotImplementedError

    def step(self) -> None:
        self.last_epoch += 1
        values = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, values, strict=True):
            group["lr"] = lr


class LambdaLR(LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch: int = -1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        epoch = self.last_epoch
        return [base * self.lr_lambda(epoch) for base in self._initial_lr]


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base * factor for base in self._initial_lr]


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch == 0:
            return self._initial_lr
        return [
            self.eta_min + (base - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base in self._initial_lr
        ]
