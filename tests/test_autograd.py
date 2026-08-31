"""Ch3 自动微分引擎测试：反向传播、链式法则、梯度累加、数值对照。"""

import numpy as np

from minitorch import Tensor
from minitorch.autograd.grad_mode import enable_grad, is_grad_enabled, no_grad


def test_chain_rule():
    x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    x.requires_grad = True
    y = (x * 2).sum()
    y.backward()
    assert x.grad.tolist() == [2.0, 2.0, 2.0]


def test_shared_leaf_accumulation():
    x = Tensor.from_numpy(np.array([2.0]))
    x.requires_grad = True
    y = x * x
    y.sum().backward()
    assert x.grad.tolist() == [4.0]


def test_mul_backward():
    a = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    a.requires_grad = True
    b = Tensor.from_numpy(np.array([4.0, 5.0, 6.0]))
    b.requires_grad = True
    (a * b).sum().backward()
    assert a.grad.tolist() == [4.0, 5.0, 6.0]
    assert b.grad.tolist() == [1.0, 2.0, 3.0]


def test_matmul_backward():
    a = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    a.requires_grad = True
    b = Tensor.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0]]))
    b.requires_grad = True
    (a @ b).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 2)) @ np.eye(2).T)
    np.testing.assert_allclose(b.grad.numpy(), np.array([[1.0, 3.0], [2.0, 4.0]]) @ np.ones((2, 2)))


def test_sum_backward():
    x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    x.requires_grad = True
    x.sum().backward()
    assert x.grad.tolist() == [1.0, 1.0, 1.0]


def test_broadcast_backward():
    a = Tensor.from_numpy(np.ones((3, 4)))
    a.requires_grad = True
    b = Tensor.from_numpy(np.arange(4).astype(float))
    b.requires_grad = True
    (a + b).sum().backward()
    assert a.grad.shape == (3, 4)
    assert b.grad.tolist() == [3.0, 3.0, 3.0, 3.0]


def test_numerical_grad_comparison(numerical_grad):
    def f(v):
        t = Tensor.from_numpy(v)
        t.requires_grad = True
        return ((t * t).sum()).item()

    x0 = np.array([1.0, 2.0, 3.0])
    x = Tensor.from_numpy(x0)
    x.requires_grad = True
    ((x * x).sum()).backward()
    expected = numerical_grad(f, x0)
    np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-4)


def test_complex_graph_accumulation():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    x.requires_grad = True
    y = x * x
    z = y + y
    z.sum().backward()
    assert x.grad.tolist() == [4.0, 8.0]


def test_no_grad_context():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    x.requires_grad = True
    with no_grad():
        y = x + x
        assert not y.requires_grad
        assert y.grad_fn is None
    assert not is_grad_enabled() or is_grad_enabled()


def test_enable_grad_context():
    with no_grad():
        assert not is_grad_enabled()
        with enable_grad():
            assert is_grad_enabled()
        assert not is_grad_enabled()
    assert is_grad_enabled()


def test_backward_sets_grad():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    x.requires_grad = True
    y = x * 3
    y.sum().backward()
    assert x.grad.tolist() == [3.0, 3.0]


def test_pow_backward():
    x = Tensor.from_numpy(np.array([2.0, 3.0]))
    x.requires_grad = True
    (x**3).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [12.0, 27.0])
