"""tracer：符号追踪器（Ch12）。

symbolic_trace(func) 用 Proxy 替换输入运行 func，记录运算为 Graph Node。
对应真实 PyTorch 的 fx/_symbolic_trace.py。
"""

from __future__ import annotations

from collections.abc import Callable

from .graph import Graph
from .proxy import Proxy


def symbolic_trace(func: Callable, n_inputs: int = 1) -> Graph:
    graph = Graph()
    proxies = [Proxy(graph.placeholder(f"x{i}"), graph) for i in range(n_inputs)]
    result = func(*proxies)
    if isinstance(result, Proxy):
        graph.output(result.node)
    elif isinstance(result, tuple | list):
        output_node = graph.create_node("output", "output", tuple(r.node for r in result))
        output_node.name = "output"
    return graph
