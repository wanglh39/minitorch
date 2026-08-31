"""CppTensor 包装层测试。

验证 CppTensor 提供与 Tensor 一致的接口，
算子和 autograd 委托给 C++ 后端。
"""

import numpy as np
import pytest

from minitorch import _has_cpp

pytestmark = pytest.mark.skipif(not _has_cpp, reason="C++ 扩展未编译")


@pytest.fixture
def T():
    from minitorch.cpp_tensor import CppTensor
    return CppTensor


def approx(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b, strict=True))


# ── 创建与属性 ──────────────────────────────────────────

def test_from_list(T):
    a = T([1.0, 2.0, 3.0])
    assert a.shape == (3,)
    assert a.numel == 3
    assert a.ndim == 1


def test_from_nested_list(T):
    a = T([[1.0, 2.0], [3.0, 4.0]])
    assert a.shape == (2, 2)
    assert a.numel == 4


def test_from_numpy(T):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = T.from_numpy(arr)
    assert a.shape == (2, 2)
    assert np.allclose(a.numpy(), arr)


def test_zeros_ones(T):
    z = T.zeros(2, 3)
    assert z.shape == (2, 3)
    assert all(v == 0 for v in z.to_vector())
    o = T.ones(3)
    assert all(v == 1 for v in o.to_vector())


# ── 算子 ────────────────────────────────────────────────

def test_add(T):
    a = T([1.0, 2.0, 3.0])
    b = T([4.0, 5.0, 6.0])
    c = a + b
    assert approx(c.to_vector(), [5.0, 7.0, 9.0])


def test_sub(T):
    a = T([4.0, 5.0, 6.0])
    b = T([1.0, 2.0, 3.0])
    c = a - b
    assert approx(c.to_vector(), [3.0, 3.0, 3.0])


def test_mul(T):
    a = T([1.0, 2.0, 3.0])
    b = T([4.0, 5.0, 6.0])
    c = a * b
    assert approx(c.to_vector(), [4.0, 10.0, 18.0])


def test_div(T):
    a = T([4.0, 6.0])
    b = T([2.0, 3.0])
    c = a / b
    assert approx(c.to_vector(), [2.0, 2.0])


def test_neg(T):
    a = T([1.0, -2.0, 3.0])
    c = -a
    assert approx(c.to_vector(), [-1.0, 2.0, -3.0])


def test_matmul(T):
    a = T([[1.0, 2.0], [3.0, 4.0]])
    b = T([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    assert approx(c.to_vector(), [19.0, 22.0, 43.0, 50.0])


def test_relu(T):
    a = T([-1.0, 2.0, -3.0, 4.0])
    c = a.relu()
    assert approx(c.to_vector(), [0.0, 2.0, 0.0, 4.0])


def test_sum(T):
    a = T([1.0, 2.0, 3.0, 4.0])
    s = a.sum()
    assert s.item() == 10.0


def test_mean(T):
    a = T([1.0, 2.0, 3.0, 4.0])
    m = a.mean()
    assert abs(m.item() - 2.5) < 1e-6


def test_transpose(T):
    a = T([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = a.T
    assert b.shape == (3, 2)
    assert approx(b.to_vector(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])


# ── Autograd ───────────────────────────────────────────

def test_mul_backward(T):
    a = T([1.0, 2.0, 3.0], requires_grad=True)
    b = T([4.0, 5.0, 6.0], requires_grad=True)
    c = (a * b).sum()
    c.backward()
    assert approx(a.grad.to_vector(), [4.0, 5.0, 6.0])
    assert approx(b.grad.to_vector(), [1.0, 2.0, 3.0])


def test_chain_rule(T):
    a = T([1.0, 2.0], requires_grad=True)
    b = T([3.0, 4.0], requires_grad=True)
    c = (a * b + a).sum()
    c.backward()
    # d(a*b+a)/da = b+1, d/da = b+1 = [4, 5]
    assert approx(a.grad.to_vector(), [4.0, 5.0])
    # d(a*b+a)/db = a = [1, 2]
    assert approx(b.grad.to_vector(), [1.0, 2.0])


def test_quadratic(T):
    x = T([3.0], requires_grad=True)
    y = x * x
    y.backward()
    assert approx(x.grad.to_vector(), [6.0])


def test_matmul_backward(T):
    W = T([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x = T([[1.0], [1.0]], requires_grad=True)
    y = (W @ x).sum()
    y.backward()
    # dW = ones @ x^T = [[1,1],[1,1]]
    assert approx(W.grad.to_vector(), [1.0, 1.0, 1.0, 1.0])
    # dx = W^T @ ones = [4, 6]
    assert approx(x.grad.to_vector(), [4.0, 6.0])


def test_relu_backward(T):
    a = T([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
    a.relu().sum().backward()
    assert approx(a.grad.to_vector(), [0.0, 1.0, 0.0, 1.0])


# ── no_grad ────────────────────────────────────────────

def test_no_grad(T):
    from minitorch.cpp_tensor import NoGrad
    a = T([1.0, 2.0], requires_grad=True)
    b = T([3.0, 4.0], requires_grad=True)
    with NoGrad():
        c = a * b
        assert not c.requires_grad
        assert c.grad_fn is None


# ── 形状操作 ────────────────────────────────────────────

def test_reshape(T):
    a = T([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = a.reshape(2, 3)
    assert b.shape == (2, 3)


def test_numpy_roundtrip(T):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = T.from_numpy(arr)
    assert np.allclose(a.numpy(), arr)


def test_scalar_arithmetic(T):
    a = T([1.0, 2.0, 3.0])
    c = a + 10.0
    assert approx(c.to_vector(), [11.0, 12.0, 13.0])
    c2 = a * 2.0
    assert approx(c2.to_vector(), [2.0, 4.0, 6.0])