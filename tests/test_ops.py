"""Ch2 算子与分发测试：前向数值、建图、requires_grad 传播、dispatcher。"""

import numpy as np
import pytest

from minitorch import Tensor
from minitorch._C import dispatch, has_kernel, register


def test_add_forward():
    a = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    b = Tensor.from_numpy(np.array([4.0, 5.0, 6.0]))
    c = a + b
    assert c.tolist() == [5.0, 7.0, 9.0]


def test_mul_forward():
    a = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    b = Tensor.from_numpy(np.array([4.0, 5.0, 6.0]))
    assert (a * b).tolist() == [4.0, 10.0, 18.0]


def test_matmul_forward():
    a = Tensor.from_numpy(np.arange(6).reshape(2, 3).astype(float))
    b = Tensor.from_numpy(np.arange(12).reshape(3, 4).astype(float))
    c = a @ b
    np.testing.assert_allclose(c.numpy(), np.arange(6).reshape(2, 3) @ np.arange(12).reshape(3, 4))


def test_sum_forward():
    a = Tensor.from_numpy(np.arange(6).reshape(2, 3).astype(float))
    assert a.sum().item() == 15
    assert a.sum(dim=1).tolist() == [3, 12]


def test_requires_grad_propagation():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    x.requires_grad = True
    y = x + x
    assert y.requires_grad
    z = y * y
    assert z.requires_grad


def test_grad_fn_built():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    x.requires_grad = True
    y = x + x
    assert y.grad_fn is not None
    assert y.grad_fn.name == "Add"


def test_no_graph_when_not_required():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    y = x + x
    assert not y.requires_grad
    assert y.grad_fn is None


def test_computation_graph_chain():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    x.requires_grad = True
    y = x * x
    z = y + y
    assert z.grad_fn.name == "Add"
    assert z.grad_fn.next_edges[0] is y.grad_fn
    assert y.grad_fn.name == "Mul"


def test_broadcast_in_apply():
    a = Tensor.from_numpy(np.ones((3, 4)))
    a.requires_grad = True
    b = Tensor.from_numpy(np.arange(4).astype(float))
    c = a + b
    assert c.shape == (3, 4)
    assert c[0].tolist() == [1, 2, 3, 4]


def test_neg_and_sub_forward():
    x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    assert (-x).tolist() == [-1.0, -2.0, -3.0]
    assert (x - 1).tolist() == [0.0, 1.0, 2.0]


def test_pow_forward():
    x = Tensor.from_numpy(np.array([2.0, 3.0]))
    assert (x**2).tolist() == [4.0, 9.0]


def test_dispatcher_register_and_dispatch():
    register("add_cpu", lambda a, b: a + b)
    assert has_kernel("add_cpu")
    assert dispatch("add_cpu", 1, 2) == 3


def test_dispatcher_missing_kernel():
    assert not has_kernel("nonexistent_op")
    with pytest.raises(RuntimeError):
        dispatch("nonexistent_op")


def test_leaf_accumulate_grad_node():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    x.requires_grad = True
    y = x * x
    edge = y.grad_fn.next_edges[0]
    assert edge.name == "AccumulateGrad"
    assert edge.variable is x