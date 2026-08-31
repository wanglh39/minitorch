"""GraphModule：可执行图模块（Ch12）。

持有 Graph，forward 时解释执行每个 Node。
对应真实 PyTorch 的 fx/graph_module.py。
"""

from __future__ import annotations

from .graph import Graph, Node


def _resolve(arg, env: dict[str, object]):
    if isinstance(arg, Node):
        return env[arg.name]
    return arg


class GraphModule:
    def __init__(self, graph: Graph):
        self.graph = graph

    def forward(self, *inputs):
        env: dict[str, object] = {}
        for node in self.graph.nodes:
            if node.op == "placeholder":
                idx = int(node.name[1:]) if len(node.name) > 1 else 0
                env[node.name] = inputs[idx]
            elif node.op == "call_function":
                args = [_resolve(a, env) for a in node.args]
                kwargs = {k: _resolve(v, env) for k, v in node.kwargs.items()}
                env[node.name] = node.target(*args, **kwargs)
            elif node.op == "call_method":
                method = node.target
                obj = _resolve(node.args[0], env)
                args = [_resolve(a, env) for a in node.args[1:]]
                kwargs = {k: _resolve(v, env) for k, v in node.kwargs.items()}
                env[node.name] = getattr(obj, method)(*args, **kwargs)
            elif node.op == "output":
                return _resolve(node.args[0], env)
        return None

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def code(self) -> str:
        return self.graph.codegen()
