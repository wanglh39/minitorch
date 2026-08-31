"""variable：Tensor.backward 入口与计算图管理（Ch3 入口 / Ch4 扩展）。

动态图：每次前向重建图，backward 后默认释放（Ch4 加 retain_graph）。
对应真实 PyTorch 的 csrc/autograd/variable.h。
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from .engine import run_backward


def backward(
    tensor: Tensor,
    gradient: Tensor | None = None,
    retain_graph: bool = False,
    retain_grad: bool = False,
) -> None:
    if tensor.grad_fn is None:
        raise RuntimeError(
            "backward() called on a tensor with no grad_fn "
            "(is it a non-leaf or created without requires_grad?)"
        )
    if gradient is None:
        if tensor.size != 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        gradient = Tensor.from_numpy(np.ones(tensor.shape, dtype=tensor.dtype))
    run_backward(
        tensor.grad_fn, gradient, retain_graph=retain_graph, retain_grad=retain_grad
    )
    if not retain_graph:
        tensor.grad_fn = None
