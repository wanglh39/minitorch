"""SGD：随机梯度下降（Ch6）。

更新规则（逐参数）：
  1. weight_decay:  g ← g + wd * p          (L2 正则)
  2. momentum:      v ← μ * v + g           (动量缓冲)
  3. nesterov:      g ← g + μ * v           (Nesterov 预测)
  4. update:        p ← p - lr * g

不参与计算图——纯 in-place 数值更新，在 no_grad 下执行。
对应真实 PyTorch 的 optim/sgd.py。
"""

from __future__ import annotations

import numpy as np

from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad._numpy_view()
                param = p._numpy_view()

                if weight_decay != 0:
                    grad = grad + weight_decay * param

                if momentum != 0:
                    state = self.state.setdefault(id(p), {})
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = np.zeros_like(param)
                    buf = state["momentum_buffer"]
                    buf *= momentum
                    buf += grad if momentum == 0 else (1 - dampening) * grad

                    if nesterov:
                        grad = grad + momentum * buf
                    else:
                        grad = buf

                param -= lr * grad
