"""test_cpp_compare_reduce：C++ 比较算子和归约算子测试（Ch9 第四批）。

验证 gt/lt/eq/ge/le 和 max/min/argmax 的前向正确性和 backward。
"""

import numpy as np

from minitorch import _cpp_ext as _C


def _close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


# ── 比较算子 ─────────────────────────────────────────

def test_gt():
    a = _C.TensorImpl([1.0, 2.0, 3.0, 4.0], [4])
    b = _C.TensorImpl([3.0, 2.0, 1.0, 4.0], [4])
    r = _C.gt(a, b)
    assert _close(r.to_vector(), [0.0, 0.0, 1.0, 0.0])


def test_lt():
    a = _C.TensorImpl([1.0, 2.0, 3.0, 4.0], [4])
    b = _C.TensorImpl([3.0, 2.0, 1.0, 4.0], [4])
    r = _C.lt(a, b)
    assert _close(r.to_vector(), [1.0, 0.0, 0.0, 0.0])


def test_eq():
    a = _C.TensorImpl([1.0, 2.0, 3.0, 4.0], [4])
    b = _C.TensorImpl([3.0, 2.0, 1.0, 4.0], [4])
    r = _C.eq(a, b)
    assert _close(r.to_vector(), [0.0, 1.0, 0.0, 1.0])


def test_ge():
    a = _C.TensorImpl([1.0, 2.0, 3.0, 4.0], [4])
    b = _C.TensorImpl([3.0, 2.0, 1.0, 4.0], [4])
    r = _C.ge(a, b)
    assert _close(r.to_vector(), [0.0, 1.0, 1.0, 1.0])


def test_le():
    a = _C.TensorImpl([1.0, 2.0, 3.0, 4.0], [4])
    b = _C.TensorImpl([3.0, 2.0, 1.0, 4.0], [4])
    r = _C.le(a, b)
    assert _close(r.to_vector(), [1.0, 1.0, 0.0, 1.0])


def test_gt_broadcast():
    a = _C.TensorImpl([1.0, 2.0, 3.0], [3])
    b = _C.TensorImpl([2.0], [1])
    r = _C.gt(a, b)
    assert _close(r.to_vector(), [0.0, 0.0, 1.0])


# ── max ──────────────────────────────────────────────

def test_max_global():
    a = _C.TensorImpl([3.0, 1.0, 4.0, 1.0, 5.0], [5])
    r = _C.max(a)
    assert _close(r.item(), 5.0)


def test_max_along_dim():
    a = _C.TensorImpl([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    r = _C.max(a, 1, False)
    assert r.shape == [2]
    assert _close(r.to_vector(), [3.0, 6.0])


def test_max_backward():
    a = _C.TensorImpl([1.0, 3.0, 2.0], [3], True)
    r = _C.autograd_max(a, -1, False)
    assert _close(r.item(), 3.0)
    r.backward(_C.TensorImpl([1.0], []))
    # grad flows only to the max element (index 1)
    assert _close(a.grad.to_vector(), [0.0, 1.0, 0.0])


def test_max_backward_2d():
    a = _C.TensorImpl([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], [2, 3], True)
    r = _C.autograd_max(a, 1, False)
    # max of each row: [3, 6]
    assert _close(r.to_vector(), [3.0, 6.0])
    r.backward(_C.TensorImpl([1.0, 1.0], [2]))
    # grad flows to max positions: (0,1) and (1,2)
    assert _close(a.grad.numpy(), [[0, 1, 0], [0, 0, 1]])


# ── min ──────────────────────────────────────────────

def test_min_global():
    a = _C.TensorImpl([3.0, 1.0, 4.0, 1.0, 5.0], [5])
    r = _C.min(a)
    assert _close(r.item(), 1.0)


def test_min_along_dim():
    a = _C.TensorImpl([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    r = _C.min(a, 1, False)
    assert _close(r.to_vector(), [1.0, 4.0])


def test_min_backward():
    a = _C.TensorImpl([1.0, 3.0, 2.0], [3], True)
    r = _C.autograd_min(a, -1, False)
    assert _close(r.item(), 1.0)
    r.backward(_C.TensorImpl([1.0], []))
    # grad flows only to the min element (index 0)
    assert _close(a.grad.to_vector(), [1.0, 0.0, 0.0])


# ── argmax ───────────────────────────────────────────

def test_argmax_global():
    a = _C.TensorImpl([3.0, 1.0, 4.0, 1.0, 5.0], [5])
    r = _C.argmax(a)
    assert _close(r.item(), 4.0)


def test_argmax_along_dim():
    a = _C.TensorImpl([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], [2, 3])
    r = _C.argmax(a, 1, False)
    # argmax of each row: [1, 2]
    assert _close(r.to_vector(), [1.0, 2.0])


def test_argmax_2d():
    a = _C.TensorImpl([0.1, 0.9, 0.2, 0.3, 0.4, 0.8], [2, 3])
    r = _C.argmax(a, 1, False)
    assert _close(r.to_vector(), [1.0, 2.0])