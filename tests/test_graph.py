"""Ch4 计算图机制测试：图释放、retain_graph、retain_grad、no_grad。"""

import numpy as np

from minitorch import Tensor
from minitorch.autograd.grad_mode import no_grad


def test_graph_freed_after_backward():
    x = Tensor.from_numpy(np.array([2.0]))
    x.requires_grad = True
    y = x * x
    y.backward()
    assert y.grad_fn is None


def test_retain_graph_allows_second_backward():
    x = Tensor.from_numpy(np.array([2.0]))
    x.requires_grad = True
    y = x * x
    y.backward(retain_graph=True)
    assert x.grad.tolist() == [4.0]
    y.backward()
    assert x.grad.tolist() == [8.0]


def test_no_grad_skips_graph():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    x.requires_grad = True
    with no_grad():
        y = x + 1
        assert y.grad_fn is None
        assert not y.requires_grad


def test_retain_grad():
    x = Tensor.from_numpy(np.array([2.0]))
    x.requires_grad = True
    y = x * x
    y.backward(retain_grad=True)
    assert y.grad is not None
    assert y.grad.item() == 1.0


def test_default_no_retain_grad():
    x = Tensor.from_numpy(np.array([2.0]))
    x.requires_grad = True
    y = x * x
    y.backward()
    assert y.grad is None


def test_retain_graph_keeps_grad_fn():
    x = Tensor.from_numpy(np.array([2.0]))
    x.requires_grad = True
    y = x * x
    y.backward(retain_graph=True)
    assert y.grad_fn is not None


def test_gradient_argument():
    x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    x.requires_grad = True
    y = x * x
    g = Tensor.from_numpy(np.array([1.0, 0.0, 0.0]))
    y.backward(g)
    assert x.grad.tolist() == [2.0, 0.0, 0.0]
