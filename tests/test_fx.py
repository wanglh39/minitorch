"""Ch12 图与编译测试。"""

import numpy as np

from minitorch import Tensor
from minitorch.fx import GraphModule, symbolic_trace
from minitorch.fx.passes.fusion import fuse_add_mul


def test_trace_node_count_matches_calls():
    def f(x):
        return (x + 1) * 2

    graph = symbolic_trace(f, n_inputs=1)
    assert len(graph.nodes) == 4
    assert graph.nodes[0].op == "placeholder"
    assert graph.nodes[1].op == "call_function"
    assert graph.nodes[2].op == "call_function"
    assert graph.nodes[3].op == "output"


def test_graph_module_forward_equivalent():
    def f(x):
        return (x + 1) * 2

    graph = symbolic_trace(f, n_inputs=1)
    gm = GraphModule(graph)
    x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    expected = f(x)
    actual = gm(x)
    assert np.allclose(actual.numpy(), expected.numpy())


def test_graph_module_matmul():
    def f(x):
        return x @ x + 1

    graph = symbolic_trace(f, n_inputs=1)
    gm = GraphModule(graph)
    x = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    expected = f(x)
    actual = gm(x)
    assert np.allclose(actual.numpy(), expected.numpy())


def test_fusion_preserves_values():
    def f(x):
        return (x + 1) * 2

    graph = symbolic_trace(f, n_inputs=1)
    fused_graph = fuse_add_mul(graph)
    assert len(fused_graph.nodes) < len(graph.nodes)

    gm_orig = GraphModule(graph)
    gm_fused = GraphModule(fused_graph)
    x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(gm_orig(x).numpy(), gm_fused(x).numpy())


def test_fusion_reduces_node_count():
    def f(x):
        return (x + 1) * 2

    graph = symbolic_trace(f, n_inputs=1)
    original_count = len(graph.nodes)
    fused_graph = fuse_add_mul(graph)
    fused_count = len(fused_graph.nodes)
    assert fused_count == original_count - 1


def test_codegen():
    def f(x):
        return (x + 1) * 2

    graph = symbolic_trace(f, n_inputs=1)
    code = graph.codegen()
    assert "def forward" in code
    assert "return" in code


def test_multi_input_trace():
    def f(x0, x1):
        return x0 + x1

    graph = symbolic_trace(f, n_inputs=2)
    gm = GraphModule(graph)
    a = Tensor.from_numpy(np.array([1.0, 2.0]))
    b = Tensor.from_numpy(np.array([3.0, 4.0]))
    result = gm(a, b)
    assert np.allclose(result.numpy(), [4.0, 6.0])
