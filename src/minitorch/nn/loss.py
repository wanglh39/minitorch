"""loss：损失函数（Ch7）。

设计要点：
  - CrossEntropyLoss = log_softmax + nll_loss，拆分以暴露数值稳定技巧（减 max）。
  - MSELoss 由现有 autograd 算子组合，图自动构建。
  - 与真实 PyTorch 的 nn/modules/loss.py 对应。
"""

from __future__ import annotations

from ..tensor import Tensor
from . import functional as F
from .module import Module


class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(pred, target)


class NLLLoss(Module):
    def forward(self, log_probs: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(log_probs, target)


class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(logits, target)
