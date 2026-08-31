"""Function：自动微分算子基类（Ch2 建图 / Ch3 反向）。

每个算子是 Function 子类，实现 forward/backward 静态方法。
Function.apply 调 forward 并在任一输入 requires_grad 时构建 Node 挂到输出.grad_fn。
Node 持有 backward_fn 与 next_edges（输入的 grad_fn 列表）。
AccumulateGrad 是叶子节点的 Node，反向时把梯度累加到 variable.grad。
对应真实 PyTorch 的 torch.autograd.Function / csrc/autograd/function.h。
"""

from __future__ import annotations

from ..tensor import Tensor
from .grad_mode import is_grad_enabled


class Context:
    """forward 与 backward 之间传递信息的上下文。"""

    def __init__(self):
        self.saved_tensors: tuple = ()
        self.meta: dict = {}

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class Node:
    """计算图节点，持有 backward 函数与 next_edges。"""

    def __init__(self, backward_fn, next_edges: list, name: str = ""):
        self.backward_fn = backward_fn
        self.next_edges = next_edges
        self.name = name
        self.output: Tensor | None = None

    def __repr__(self) -> str:
        return f"Node({self.name})"


class AccumulateGrad(Node):
    """叶子节点的 Node：反向时把梯度累加到 variable.grad。"""

    def __init__(self, variable: Tensor):
        self.variable = variable
        super().__init__(self._accumulate, [], name="AccumulateGrad")

    def _accumulate(self, grad: Tensor):
        if self.variable.grad is None:
            self.variable.grad = grad
        else:
            self.variable.grad = self.variable.grad + grad


def _reduce_grad(grad: Tensor, shape: tuple[int, ...]) -> Tensor:
    """把 grad 的 shape reduce 回 shape（处理前向广播的反向）。"""
    while grad.ndim > len(shape):
        grad = grad.sum(dim=0)
    for i in range(len(shape)):
        if grad.shape[i] != shape[i] and shape[i] == 1:
            grad = grad.sum(dim=i, keepdim=True)
    return grad


class Function:
    """自动微分算子基类。子类实现 forward/backward 静态方法。"""

    @staticmethod
    def forward(ctx: Context, *args) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, *grad_outputs) -> tuple:
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor:
        ctx = Context()
        result = cls.forward(ctx, *args, **kwargs)

        if not is_grad_enabled():
            return result

        needs_grad = any(isinstance(a, Tensor) and a.requires_grad for a in args)
        if not needs_grad:
            return result

        next_edges: list[Node | None] = []
        for a in args:
            if isinstance(a, Tensor) and a.requires_grad:
                if a.grad_fn is not None:
                    next_edges.append(a.grad_fn)
                else:
                    next_edges.append(AccumulateGrad(a))
            else:
                next_edges.append(None)

        node = Node(
            backward_fn=lambda *grads: cls.backward(ctx, *grads),
            next_edges=next_edges,
            name=cls.__name__,
        )
        node.output = result
        result.requires_grad = True
        result.grad_fn = node
        return result
