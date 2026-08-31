"""arithmetic：基础算子（Ch2）。

每个算子是 Function 子类，forward/backward 用 Tensor 底层方法（_add/_mul 等）
避免与公开算子（走 apply）递归。反向执行在 Ch3 引擎。
"""

from __future__ import annotations

import numpy as np

from ..autograd.function import Function, _reduce_grad
from ..tensor import Tensor


class Add(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        a, b = Tensor.broadcast_tensors(a, b)
        return a._add(b)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b = ctx.saved_tensors
        return (_reduce_grad(grad_output, a.shape), _reduce_grad(grad_output, b.shape))


class Sub(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        a, b = Tensor.broadcast_tensors(a, b)
        return a._sub(b)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b = ctx.saved_tensors
        return (_reduce_grad(grad_output, a.shape), _reduce_grad(grad_output._neg(), b.shape))


class Mul(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        a, b = Tensor.broadcast_tensors(a, b)
        return a._mul(b)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b = ctx.saved_tensors
        grad_a = _reduce_grad(grad_output._mul(b), a.shape)
        grad_b = _reduce_grad(grad_output._mul(a), b.shape)
        return (grad_a, grad_b)


class Div(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        a, b = Tensor.broadcast_tensors(a, b)
        return a._div(b)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b = ctx.saved_tensors
        grad_a = _reduce_grad(grad_output._div(b), a.shape)
        grad_b = _reduce_grad(grad_output._mul(a)._div(b._mul(b))._neg(), b.shape)
        return (grad_a, grad_b)


class Pow(Function):
    @staticmethod
    def forward(ctx, a: Tensor, exponent: Tensor | float | int) -> Tensor:
        exp = exponent if isinstance(exponent, Tensor) else Tensor.from_numpy(np.asarray(exponent))
        ctx.save_for_backward(a, exp)
        return a._pow(exp)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, exp = ctx.saved_tensors
        grad_a = grad_output._mul(exp)._mul(a._pow(exp._sub(Tensor.from_numpy(np.asarray(1.0)))))
        return (grad_a, None)


class Neg(Function):
    @staticmethod
    def forward(ctx, a: Tensor) -> Tensor:
        return a._neg()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return (grad_output._neg(),)


class Matmul(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a._matmul(b)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b = ctx.saved_tensors
        grad_a = grad_output._matmul(b.transpose())
        grad_b = a.transpose()._matmul(grad_output)
        return (grad_a, grad_b)


class Sum(Function):
    @staticmethod
    def forward(ctx, a: Tensor, dim=None, keepdim: bool = False) -> Tensor:
        ctx.save_for_backward(a)
        ctx.meta["dim"] = dim
        ctx.meta["keepdim"] = keepdim
        return a._sum(dim=dim, keepdim=keepdim)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (a,) = ctx.saved_tensors
        grad = grad_output
        if ctx.meta["dim"] is not None and not ctx.meta["keepdim"]:
            grad = grad.unsqueeze(ctx.meta["dim"])
        grad = grad.broadcast_to(a.shape)
        return (grad,)


class Mean(Function):
    @staticmethod
    def forward(ctx, a: Tensor, dim=None, keepdim: bool = False) -> Tensor:
        ctx.save_for_backward(a)
        ctx.meta["n"] = a.size
        return a._mean(dim=dim, keepdim=keepdim)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (a,) = ctx.saved_tensors
        n = Tensor.from_numpy(np.asarray(1.0 / ctx.meta["n"]))
        grad = grad_output._mul(n).broadcast_to(a.shape)
        return (grad,)


class Transpose(Function):
    @staticmethod
    def forward(ctx, a: Tensor, dim0: int = 1, dim1: int = 0) -> Tensor:
        ctx.meta["dim0"] = dim0
        ctx.meta["dim1"] = dim1
        return a._transpose(dim0, dim1)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return (grad_output._transpose(ctx.meta["dim0"], ctx.meta["dim1"]),)
