"""Linear：全连接层（Ch5）。

forward: x @ weight.T + bias。权重用 uniform 初始化（贴近 PyTorch 默认）。
对应真实 PyTorch 的 nn/modules/linear.py。
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from .module import Module
from .parameter import Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        bound = 1.0 / (in_features**0.5)
        w = np.random.uniform(-bound, bound, (out_features, in_features))
        self.weight = Parameter(Tensor.from_numpy(w))
        if bias:
            b = np.random.uniform(-bound, bound, (out_features,))
            self.bias = Parameter(Tensor.from_numpy(b))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out
