"""test_cpp_tensor_ops：C++ Tensor 操作和 inplace 算子测试（Ch9 第三批）。

验证 clone/expand/fill_/zero_ 和 sub_inplace/mul_inplace/div_inplace。
"""

import numpy as np

from minitorch import _cpp_ext as _C


def _close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


# ── clone ────────────────────────────────────────────

def test_clone():
    a = _C.TensorImpl([1.0, 2.0, 3.0], [3])
    b = a.clone()
    assert _close(b.to_vector(), [1.0, 2.0, 3.0])
    # 修改 b 不影响 a
    b.from_vector([10.0, 20.0, 30.0])
    assert _close(a.to_vector(), [1.0, 2.0, 3.0])


def test_clone_2d():
    a = _C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = a.clone()
    assert _close(b.to_vector(), [1.0, 2.0, 3.0, 4.0])
    assert b.shape == [2, 2]


# ── expand ───────────────────────────────────────────

def test_expand():
    a = _C.TensorImpl([1.0, 2.0, 3.0], [1, 3])
    b = a.expand([4, 3])
    assert b.shape == [4, 3]
    # 每行都应该一样
    for row in range(4):
        assert _close(b.numpy()[row], [1.0, 2.0, 3.0])


def test_expand_with_1dim():
    a = _C.TensorImpl([5.0], [1])
    b = a.expand([5])
    assert b.shape == [5]
    assert _close(b.to_vector(), [5.0, 5.0, 5.0, 5.0, 5.0])


# ── fill_ / zero_ ────────────────────────────────────

def test_fill_():
    a = _C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    a.fill_(7.0)
    assert _close(a.to_vector(), [7.0, 7.0, 7.0, 7.0])


def test_zero_():
    a = _C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    a.zero_()
    assert _close(a.to_vector(), [0.0, 0.0, 0.0, 0.0])


# ── sub_inplace ──────────────────────────────────────

def test_sub_inplace():
    a = _C.TensorImpl([10.0, 20.0, 30.0], [3])
    b = _C.TensorImpl([1.0, 2.0, 3.0], [3])
    _C.sub_inplace(a, b)
    assert _close(a.to_vector(), [9.0, 18.0, 27.0])


def test_mul_inplace():
    a = _C.TensorImpl([10.0, 20.0, 30.0], [3])
    b = _C.TensorImpl([2.0, 3.0, 4.0], [3])
    _C.mul_inplace(a, b)
    assert _close(a.to_vector(), [20.0, 60.0, 120.0])


def test_div_inplace():
    a = _C.TensorImpl([10.0, 20.0, 30.0], [3])
    b = _C.TensorImpl([2.0, 5.0, 10.0], [3])
    _C.div_inplace(a, b)
    assert _close(a.to_vector(), [5.0, 4.0, 3.0])


# ── CppTensor 包装测试 ───────────────────────────────

def test_cpptensor_clone():
    from minitorch.cpp_tensor import CppTensor
    a = CppTensor([1.0, 2.0, 3.0], requires_grad=True)
    b = a.clone()
    assert b.shape == (3,)
    assert _close(b.numpy(), [1.0, 2.0, 3.0])


def test_cpptensor_zero_():
    from minitorch.cpp_tensor import CppTensor
    a = CppTensor([1.0, 2.0, 3.0])
    a.zero_()
    assert _close(a.numpy(), [0.0, 0.0, 0.0])


def test_cpptensor_fill_():
    from minitorch.cpp_tensor import CppTensor
    a = CppTensor([1.0, 2.0, 3.0])
    a.fill_(9.0)
    assert _close(a.numpy(), [9.0, 9.0, 9.0])