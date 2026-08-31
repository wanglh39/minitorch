"""fusion：算子融合 pass（Ch12）。

相邻算子合并（如 add+mul），减少 kernel launch 与访存。
融合前后数值不变，节点数减少。
对应真实 PyTorch 的 fx/passes/fusion.py。
"""

from __future__ import annotations

from ..graph import Graph, Node
from ..proxy import _add, _mul


def _fused_add_mul(a, b, c):
    return (a + b) * c


def fuse_add_mul(graph: Graph) -> Graph:
    """把 x = add(a, b); y = mul(x, c) 融合为 y = fused_add_mul(a, b, c)。"""
    fuse_pairs: dict[str, str] = {}
    skip_nodes: set[str] = set()

    for node in graph.nodes:
        if node.op == "call_function" and node.target is _mul:
            prev_node = node.args[0]
            if (
                isinstance(prev_node, Node)
                and prev_node.op == "call_function"
                and prev_node.target is _add
            ):
                fuse_pairs[prev_node.name] = node.name
                skip_nodes.add(prev_node.name)

    new_graph = Graph()
    node_map: dict[str, Node] = {}

    for node in graph.nodes:
        if node.name in skip_nodes:
            continue
        if node.op == "placeholder":
            new_node = new_graph.placeholder(node.name)
            node_map[node.name] = new_node
        elif node.op == "call_function" and node.target is _mul and node.name in fuse_pairs.values():
            prev_name = next(k for k, v in fuse_pairs.items() if v == node.name)
            prev_node = next(n for n in graph.nodes if n.name == prev_name)
            add_a = _remap_arg(prev_node.args[0], node_map)
            add_b = _remap_arg(prev_node.args[1], node_map)
            mul_c = _remap_arg(node.args[1], node_map)
            new_node = new_graph.call_function(_fused_add_mul, (add_a, add_b, mul_c))
            new_node.name = f"fused_{prev_name}_{node.name}"
            node_map[node.name] = new_node
        elif node.op == "output":
            mapped = _remap_arg(node.args[0], node_map)
            new_graph.output(mapped)
        else:
            new_node = _copy_node(new_graph, node, node_map)
            node_map[node.name] = new_node

    return new_graph


def _remap_arg(arg, node_map: dict[str, Node]):
    if isinstance(arg, Node):
        return node_map.get(arg.name, arg)
    return arg


def _copy_node(new_graph: Graph, node: Node, node_map: dict[str, Node]) -> Node:
    args = tuple(_remap_arg(a, node_map) for a in node.args)
    kwargs = {k: _remap_arg(v, node_map) for k, v in node.kwargs.items()}
    new_node = new_graph.create_node(node.op, node.target, args, kwargs)
    new_node.name = node.name
    return new_node
