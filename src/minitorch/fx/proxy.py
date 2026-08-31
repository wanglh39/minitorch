"""Proxy：符号追踪代理对象（Ch12）。

用 Proxy 替换输入，拦截所有运算记录为 Graph Node。
对应真实 PyTorch 的 fx/proxy.py。
"""

from __future__ import annotations

from .graph import Graph, Node


class Proxy:
    def __init__(self, node: Node, graph: Graph):
        self.node = node
        self.graph = graph

    def _record(self, op: str, target, args, kwargs=None) -> Proxy:
        proxy_args = tuple(a.node if isinstance(a, Proxy) else a for a in args)
        proxy_kwargs = {
            k: v.node if isinstance(v, Proxy) else v for k, v in (kwargs or {}).items()
        }
        node = self.graph.create_node(op, target, proxy_args, proxy_kwargs)
        return Proxy(node, self.graph)

    def __add__(self, other):
        return self._record("call_function", _add, (self, other))

    def __mul__(self, other):
        return self._record("call_function", _mul, (self, other))

    def __sub__(self, other):
        return self._record("call_function", _sub, (self, other))

    def __truediv__(self, other):
        return self._record("call_function", _div, (self, other))

    def __matmul__(self, other):
        return self._record("call_function", _matmul, (self, other))

    def __neg__(self):
        return self._record("call_function", _neg, (self,))

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def method(*args, **kwargs):
            return self._record("call_method", name, (self, *args), kwargs)

        return method

    def __repr__(self):
        return f"Proxy({self.node})"


def _add(a, b):
    return a + b


def _mul(a, b):
    return a * b


def _sub(a, b):
    return a - b


def _div(a, b):
    return a / b


def _matmul(a, b):
    return a @ b


def _neg(a):
    return -a
