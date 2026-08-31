"""nn：神经网络模块包（Ch5 / Ch7）。

子模块：
  - module:     Module 基类。
  - parameter:  Parameter（requires_grad=True 的特殊 Tensor）。
  - containers: Sequential / ModuleList / ModuleDict。
  - linear:     Linear 层。
  - functional: 无状态算子 functional API。
  - loss:       损失函数（MSE / CrossEntropy）。
"""

from . import functional as F
from .containers import ModuleList, Sequential
from .linear import Linear
from .loss import CrossEntropyLoss, MSELoss, NLLLoss
from .module import Module
from .parameter import Parameter

__all__ = [
    "CrossEntropyLoss",
    "F",
    "Linear",
    "MSELoss",
    "Module",
    "ModuleList",
    "NLLLoss",
    "Parameter",
    "Sequential",
]