"""Engine：反向传播调度引擎（Ch3）。

从输出 Node 出发，DFS 后序逆序做拓扑排序，按序逐个调用 backward_fn，
梯度沿 next_edges 传给前驱，多后继梯度求和。AccumulateGrad 累加到 variable.grad。
教学版单线程；真实 PyTorch 多线程用 ReadyQueue 并行无依赖 Node。
对应真实 PyTorch 的 csrc/autograd/engine.cpp。
"""

from __future__ import annotations

from ..tensor import Tensor
from .function import AccumulateGrad, Node
from .grad_mode import no_grad


def _topological_sort(root: Node) -> list[Node]:
    topo: list[Node] = []
    visited: set[int] = set()

    def dfs(node: Node | None) -> None:
        if node is None or id(node) in visited:
            return
        visited.add(id(node))
        for edge in node.next_edges:
            dfs(edge)
        topo.append(node)

    dfs(root)
    return topo


def run_backward(
    root: Node,
    root_grad: Tensor,
    retain_graph: bool = False,
    retain_grad: bool = False,
) -> None:
    with no_grad():
        topo = _topological_sort(root)
        grad_map: dict[int, Tensor | None] = {id(root): root_grad}

        for node in reversed(topo):
            grad = grad_map.get(id(node))
            if grad is None:
                continue

            if isinstance(node, AccumulateGrad):
                node.backward_fn(grad)
                continue

            if retain_grad and node.output is not None:
                if node.output.grad is None:
                    node.output.grad = grad
                else:
                    node.output.grad = node.output.grad + grad

            grads = node.backward_fn(grad)
            if not isinstance(grads, tuple):
                grads = (grads,)

            for edge, g in zip(node.next_edges, grads, strict=True):
                if edge is None or g is None:
                    continue
                prev = grad_map.get(id(edge))
                grad_map[id(edge)] = g if prev is None else prev + g

        if not retain_graph:
            for node in topo:
                node.next_edges = []
