"""Adam：自适应矩估计优化器（Ch6）。

更新规则（逐参数，step t 从 1 开始）：
  1. weight_decay:  g ← g + wd * p
  2. 一阶矩:        m ← β₁ * m + (1 - β₁) * g
  3. 二阶矩:        v ← β₂ * v + (1 - β₂) * g²
  4. bias correction:
                     m̂ = m / (1 - β₁ᵗ)
                     v̂ = v / (1 - β₂ᵗ)
  5. update:        p ← p - lr * m̂ / (√v̂ + eps)

等价化简（PyTorch 实际写法）：
  step_size = lr / (1 - β₁ᵗ)
  denom = √v / √(1 - β₂ᵗ) + eps
  p ← p - step_size * m / denom

对应真实 PyTorch 的 optim/adam.py。
"""

from __future__ import annotations

import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad._numpy_view()
                param = p._numpy_view()

                if weight_decay != 0:
                    grad = grad + weight_decay * param

                state = self.state.setdefault(id(p), {})
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = np.zeros_like(param)
                    state["exp_avg_sq"] = np.zeros_like(param)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg *= beta1
                exp_avg += (1 - beta1) * grad
                exp_avg_sq *= beta2
                exp_avg_sq += (1 - beta2) * (grad * grad)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr / bias_correction1
                bias_correction2_sqrt = np.sqrt(bias_correction2)

                denom = np.sqrt(exp_avg_sq) / bias_correction2_sqrt + eps
                param -= step_size * exp_avg / denom
