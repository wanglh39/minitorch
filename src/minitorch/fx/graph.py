"""Graph：计算图数据结构（Ch12）。

Node 持有 op/target/args/kwargs/name/users。
Graph 持有 Node 序列，可 codegen forward。
对应真实 PyTorch 的 fx/graph.py（简化为 call_function/call_method）。
"""

from __future__ import annotations

from collections.abc import Callable


class Node:
    def __init__(
        self,
        name: str,
        op: str,
        target: Callable | str,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        self.name = name
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.users: set[Node] = set()
        self._update_users()

    def _update_users(self):
        for arg in self.args:
            if isinstance(arg, Node):
                arg.users.add(self)
        for arg in self.kwargs.values():
            if isinstance(arg, Node):
                arg.users.add(self)

    def __repr__(self):
        return f"Node({self.name}, op={self.op})"


class Graph:
    def __init__(self):
        self.nodes: list[Node] = []

    def create_node(self, op: str, target, args=(), kwargs=None) -> Node:
        name = f"{op}_{len(self.nodes)}"
        node = Node(name, op, target, args, kwargs)
        self.nodes.append(node)
        return node

    def placeholder(self, name: str = "x") -> Node:
        node = self.create_node("placeholder", name)
        node.name = name
        return node

    def call_function(self, target: Callable, args=(), kwargs=None) -> Node:
        return self.create_node("call_function", target, args, kwargs)

    def call_method(self, method: str, args=(), kwargs=None) -> Node:
        return self.create_node("call_method", method, args, kwargs)

    def output(self, result: Node) -> Node:
        return self.create_node("output", "output", (result,))

    def codegen(self) -> str:
        lines = ["def forward(self, x):"]
        for node in self.nodes:
            if node.op == "placeholder":
                lines.append(f"    {node.name} = x")
            elif node.op == "call_function":
                args_str = self._format_args(node.args)
                kwargs_str = self._format_kwargs(node.kwargs)
                target_name = getattr(node.target, "__name__", str(node.target))
                lines.append(f"    {node.name} = {target_name}({args_str}{kwargs_str})")
            elif node.op == "call_method":
                method = node.target
                args_str = self._format_args(node.args[1:])
                lines.append(f"    {node.name} = {node.args[0].name}.{method}({args_str})")
            elif node.op == "output":
                lines.append(f"    return {node.args[0].name}")
        return "\n".join(lines)

    def _format_args(self, args) -> str:
        parts = []
        for arg in args:
            parts.append(self._format_value(arg))
        return ", ".join(parts)

    def _format_kwargs(self, kwargs: dict) -> str:
        if not kwargs:
            return ""
        parts = []
        for k, v in kwargs.items():
            parts.append(f"{k}={self._format_value(v)}")
        return ", " + ", ".join(parts) if parts else ""

    def _format_value(self, val) -> str:
        if isinstance(val, Node):
            return val.name
        return repr(val)

    def __repr__(self):
        return f"Graph({len(self.nodes)} nodes)"
