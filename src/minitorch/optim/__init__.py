"""optim：优化器包（Ch6）。

子模块：
  - optimizer:    Optimizer 基类（param_groups / step / zero_grad）。
  - sgd:          SGD（动量 / nesterov / weight_decay）。
  - adam:         Adam（一阶二阶矩 + bias correction）。
  - lr_scheduler: LambdaLR / StepLR / CosineAnnealingLR 等。
"""

from .adam import Adam
from .lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler, StepLR
from .optimizer import Optimizer
from .sgd import SGD

__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "LRScheduler",
    "LambdaLR",
    "StepLR",
    "CosineAnnealingLR",
]
