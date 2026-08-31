"""functional：无状态算子 functional API（Ch7）。

包含需要自定义 backward 的算子（Relu / LogSoftmax / Softmax / NLLLoss）
以及可由现有 autograd 算子组合的函数（mse_loss / cross_entropy）。

对应真实 PyTorch 的 torch.nn.functional。
"""

from __future__ import annotations

import numpy as np

from ..autograd.function import Function
from ..tensor import Tensor


class Relu(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return Tensor.from_numpy(np.maximum(0, x._numpy_view()))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        mask = (x._numpy_view() > 0).astype(np.float64)
        return Tensor.from_numpy(grad_output._numpy_view() * mask)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        arr = x._numpy_view()
        max_val = np.max(arr, axis=dim, keepdims=True)
        shifted = arr - max_val
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
        result = shifted - log_sum_exp
        ctx.save_for_backward(Tensor.from_numpy(np.exp(result)))
        ctx.dim = dim
        return Tensor.from_numpy(result)

    @staticmethod
    def backward(ctx, grad_output):
        softmax = ctx.saved_tensors[0]
        dim = ctx.dim
        sum_grad = np.sum(grad_output._numpy_view(), axis=dim, keepdims=True)
        grad_x = grad_output._numpy_view() - softmax._numpy_view() * sum_grad
        return Tensor.from_numpy(grad_x)


class Softmax(Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        arr = x._numpy_view()
        max_val = np.max(arr, axis=dim, keepdims=True)
        exp_arr = np.exp(arr - max_val)
        result = exp_arr / np.sum(exp_arr, axis=dim, keepdims=True)
        ctx.save_for_backward(Tensor.from_numpy(result))
        ctx.dim = dim
        return Tensor.from_numpy(result)

    @staticmethod
    def backward(ctx, grad_output):
        softmax = ctx.saved_tensors[0]
        dim = ctx.dim
        grad = grad_output._numpy_view()
        sm = softmax._numpy_view()
        dot = np.sum(grad * sm, axis=dim, keepdims=True)
        grad_x = sm * (grad - dot)
        return Tensor.from_numpy(grad_x)


class NLLLoss(Function):
    @staticmethod
    def forward(ctx, log_probs, target):
        n = log_probs.shape[0]
        target_arr = target._numpy_view().astype(int)
        lp = log_probs._numpy_view()
        loss = -np.mean(lp[np.arange(n), target_arr])
        ctx.n = n
        ctx.num_classes = log_probs.shape[1]
        ctx.target = target_arr
        return Tensor.from_numpy(np.array(loss))

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.n
        target = ctx.target
        C = ctx.num_classes
        grad = np.zeros((n, C))
        grad[np.arange(n), target] = -1.0 / n
        g = Tensor.from_numpy(grad * grad_output.item())
        return g, None


def relu(x: Tensor) -> Tensor:
    return Relu.apply(x)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    return LogSoftmax.apply(x, dim=dim)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return Softmax.apply(x, dim=dim)


def nll_loss(log_probs: Tensor, target: Tensor) -> Tensor:
    if not isinstance(target, Tensor):
        target = Tensor.from_numpy(np.asarray(target))
    return NLLLoss.apply(log_probs, target)


def cross_entropy(logits: Tensor, target: Tensor, dim: int = -1) -> Tensor:
    if not isinstance(target, Tensor):
        target = Tensor.from_numpy(np.asarray(target))
    log_probs = LogSoftmax.apply(logits, dim=dim)
    return NLLLoss.apply(log_probs, target)


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    if not isinstance(target, Tensor):
        target = Tensor.from_numpy(np.asarray(target))
    diff = pred - target
    return (diff**2).mean()
