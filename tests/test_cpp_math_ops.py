"""test_cpp_math_ops：C++ 逐元素数学函数测试（Ch9 第一批）。

验证 exp/log/sqrt/abs/pow/clamp/sigmoid/tanh 的前向正确性和 backward 正确性。
对照 numpy 手算结果。
"""

import numpy as np

from minitorch import _cpp_ext as _C


def _close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


# ── exp ──────────────────────────────────────────────

def test_exp_forward():
    a = _C.TensorImpl([0.0, 1.0, 2.0], [3])
    r = _C.exp(a)
    assert _close(r.to_vector(), [1.0, np.e, np.e ** 2])


def test_exp_backward():
    a = _C.TensorImpl([0.0, 1.0, 2.0], [3], True)
    r = _C.autograd_exp(a)
    r.backward(_C.TensorImpl([1.0, 1.0, 1.0], [3]))
    # d/dx exp(x) = exp(x)
    assert _close(a.grad.to_vector(), [1.0, np.e, np.e ** 2])


# ── log ──────────────────────────────────────────────

def test_log_forward():
    a = _C.TensorImpl([1.0, np.e, np.e ** 2], [3])
    r = _C.log(a)
    assert _close(r.to_vector(), [0.0, 1.0, 2.0])


def test_log_backward():
    a = _C.TensorImpl([1.0, 2.0, 4.0], [3], True)
    r = _C.autograd_log(a)
    r.backward(_C.TensorImpl([1.0, 1.0, 1.0], [3]))
    # d/dx log(x) = 1/x
    assert _close(a.grad.to_vector(), [1.0, 0.5, 0.25])


# ── sqrt ─────────────────────────────────────────────

def test_sqrt_forward():
    a = _C.TensorImpl([0.0, 1.0, 4.0, 9.0], [4])
    r = _C.sqrt(a)
    assert _close(r.to_vector(), [0.0, 1.0, 2.0, 3.0])


def test_sqrt_backward():
    a = _C.TensorImpl([1.0, 4.0, 9.0], [3], True)
    r = _C.autograd_sqrt(a)
    r.backward(_C.TensorImpl([1.0, 1.0, 1.0], [3]))
    # d/dx sqrt(x) = 1 / (2*sqrt(x))
    assert _close(a.grad.to_vector(), [0.5, 0.25, 1.0 / 6.0])


# ── abs ──────────────────────────────────────────────

def test_abs_forward():
    a = _C.TensorImpl([-3.0, -1.0, 0.0, 2.0], [4])
    r = _C.abs_val(a)
    assert _close(r.to_vector(), [3.0, 1.0, 0.0, 2.0])


def test_abs_backward():
    a = _C.TensorImpl([-2.0, 3.0], [2], True)
    r = _C.autograd_abs_val(a)
    r.backward(_C.TensorImpl([1.0, 1.0], [2]))
    # d/dx |x| = sign(x)
    assert _close(a.grad.to_vector(), [-1.0, 1.0])


# ── pow_scalar ───────────────────────────────────────

def test_pow_forward():
    a = _C.TensorImpl([1.0, 2.0, 3.0], [3])
    r = _C.pow_scalar(a, 3.0)
    assert _close(r.to_vector(), [1.0, 8.0, 27.0])


def test_pow_backward():
    a = _C.TensorImpl([1.0, 2.0, 3.0], [3], True)
    r = _C.autograd_pow_scalar(a, 3.0)
    r.backward(_C.TensorImpl([1.0, 1.0, 1.0], [3]))
    # d/dx x^3 = 3*x^2
    assert _close(a.grad.to_vector(), [3.0, 12.0, 27.0])


def test_pow_half():
    a = _C.TensorImpl([4.0, 9.0], [2], True)
    r = _C.autograd_pow_scalar(a, 0.5)
    assert _close(r.to_vector(), [2.0, 3.0])
    r.backward(_C.TensorImpl([1.0, 1.0], [2]))
    # d/dx x^0.5 = 0.5 * x^(-0.5)
    assert _close(a.grad.to_vector(), [0.25, 1.0 / 6.0])


# ── clamp ────────────────────────────────────────────

def test_clamp_forward():
    a = _C.TensorImpl([-3.0, -0.5, 0.5, 3.0], [4])
    r = _C.clamp(a, -1.0, 1.0)
    assert _close(r.to_vector(), [-1.0, -0.5, 0.5, 1.0])


def test_clamp_backward():
    a = _C.TensorImpl([-3.0, -0.5, 0.5, 3.0], [4], True)
    r = _C.autograd_clamp(a, -1.0, 1.0)
    r.backward(_C.TensorImpl([1.0, 1.0, 1.0, 1.0], [4]))
    # grad = 0 where clamped, 1 where not
    assert _close(a.grad.to_vector(), [0.0, 1.0, 1.0, 0.0])


# ── sigmoid ──────────────────────────────────────────

def test_sigmoid_forward():
    a = _C.TensorImpl([0.0, 100.0, -100.0], [3])
    r = _C.sigmoid(a)
    assert _close(r.to_vector(), [0.5, 1.0, 0.0])


def test_sigmoid_backward():
    a = _C.TensorImpl([0.0], [1], True)
    r = _C.autograd_sigmoid(a)
    r.backward(_C.TensorImpl([1.0], [1]))
    # sigmoid(0) = 0.5, grad = 0.5 * (1 - 0.5) = 0.25
    assert _close(a.grad.to_vector(), [0.25])


# ── tanh ─────────────────────────────────────────────

def test_tanh_forward():
    a = _C.TensorImpl([0.0, 100.0, -100.0], [3])
    r = _C.tanh(a)
    assert _close(r.to_vector(), [0.0, 1.0, -1.0])


def test_tanh_backward():
    a = _C.TensorImpl([0.0], [1], True)
    r = _C.autograd_tanh(a)
    r.backward(_C.TensorImpl([1.0], [1]))
    # tanh(0) = 0, grad = 1 - 0^2 = 1
    assert _close(a.grad.to_vector(), [1.0])


def test_tanh_backward_nonzero():
    x = 1.0
    a = _C.TensorImpl([x], [1], True)
    r = _C.autograd_tanh(a)
    r.backward(_C.TensorImpl([1.0], [1]))
    expected = 1.0 - np.tanh(x) ** 2
    assert _close(a.grad.to_vector(), [expected])


# ── 组合测试 ─────────────────────────────────────────

def test_composite_exp_log():
    a = _C.TensorImpl([2.0, 5.0], [2], True)
    r = _C.autograd_log(_C.autograd_exp(a))
    # log(exp(x)) = x
    assert _close(r.to_vector(), [2.0, 5.0])
    r.backward(_C.TensorImpl([1.0, 1.0], [2]))
    # d/dx log(exp(x)) = 1
    assert _close(a.grad.to_vector(), [1.0, 1.0])


def test_composite_sigmoid_tanh_relation():
    # tanh(x) = 2*sigmoid(2x) - 1
    x = 0.7
    a = _C.TensorImpl([x], [1])
    t = _C.tanh(a)
    s = _C.sigmoid(_C.mul(a, _C.TensorImpl([2.0], [1])))
    s2 = _C.sub(_C.mul(s, _C.TensorImpl([2.0], [1])), _C.TensorImpl([1.0], [1]))
    assert _close(t.to_vector(), s2.to_vector())