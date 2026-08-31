"""Ch8 C++ 核心重写测试。

验证 C++ 扩展能正确加载，且基本算子行为与 Python 实现等价。
"""

import numpy as np
import pytest

from minitorch import _has_cpp

pytestmark = pytest.mark.skipif(not _has_cpp, reason="Ch8: C++ 扩展未编译")


def test_cpp_extension_loaded():
    from minitorch import _cpp_ext
    assert _cpp_ext.__version__ == "0.5.0"


def test_tensor_creation():
    from minitorch import _cpp_ext
    t = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert t.shape == [2, 2]
    assert t.numel == 4
    assert t.ndim == 2


def test_add():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = _cpp_ext.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2])
    c = _cpp_ext.add(a, b)
    assert c.to_vector() == [6.0, 8.0, 10.0, 12.0]


def test_sub():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2])
    b = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    c = _cpp_ext.sub(a, b)
    assert c.to_vector() == [4.0, 4.0, 4.0, 4.0]


def test_mul():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = _cpp_ext.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2])
    c = _cpp_ext.mul(a, b)
    assert c.to_vector() == [5.0, 12.0, 21.0, 32.0]


def test_matmul():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = _cpp_ext.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2])
    c = _cpp_ext.matmul(a, b)
    assert c.to_vector() == [19.0, 22.0, 43.0, 50.0]


def test_transpose():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = a.transpose()
    assert b.to_vector() == [1.0, 3.0, 2.0, 4.0]
    assert not b.is_contiguous()


def test_contiguous():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = a.transpose()
    c = b.contiguous()
    assert c.is_contiguous()
    assert c.to_vector() == [1.0, 3.0, 2.0, 4.0]


def test_neg():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, -2.0, 3.0, -4.0], [2, 2])
    b = _cpp_ext.neg(a)
    assert b.to_vector() == [-1.0, 2.0, -3.0, 4.0]


def test_relu():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([-1.0, 2.0, -3.0, 4.0], [2, 2])
    b = _cpp_ext.relu(a)
    assert b.to_vector() == [0.0, 2.0, 0.0, 4.0]


def test_numpy_roundtrip():
    from minitorch import _cpp_ext
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = _cpp_ext.TensorImpl.from_numpy(arr)
    assert t.shape == [2, 2]
    result = t.numpy()
    assert np.allclose(result, arr)


def test_view():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    b = a.view([3, 2])
    assert b.shape == [3, 2]
    assert b.to_vector() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_reshape():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    b = a.transpose().reshape([6])
    assert b.shape == [6]
    assert b.to_vector() == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]


# ── matmul 批量化 ──────────────────────────────────────

def test_matmul_1d_1d_dot():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0], [3])
    b = _cpp_ext.TensorImpl([4.0, 5.0, 6.0], [3])
    c = _cpp_ext.matmul(a, b)
    assert c.shape == []
    assert abs(c.item() - 32.0) < 1e-6  # 1*4 + 2*5 + 3*6 = 32


def test_matmul_1d_2d():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0], [2])  # [2]
    b = _cpp_ext.TensorImpl([3.0, 4.0, 5.0, 6.0], [2, 2])  # [2, 2]
    c = _cpp_ext.matmul(a, b)
    assert c.shape == [2]
    # [1,2] @ [[3,4],[5,6]] = [13, 16]
    assert c.to_vector() == [13.0, 16.0]


def test_matmul_2d_1d():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])  # [2, 2]
    b = _cpp_ext.TensorImpl([5.0, 6.0], [2])  # [2]
    c = _cpp_ext.matmul(a, b)
    assert c.shape == [2]
    # [[1,2],[3,4]] @ [5,6] = [17, 39]
    assert c.to_vector() == [17.0, 39.0]


def test_matmul_3d_batched():
    from minitorch import _cpp_ext
    # 2 个 batch，每个 [2, 2]
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])
    b = _cpp_ext.TensorImpl([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0], [2, 2, 2])
    c = _cpp_ext.matmul(a, b)
    assert c.shape == [2, 2, 2]
    # batch 0: [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
    # batch 1: [[5,6],[7,8]] @ [[1,1],[0,1]] = [[5,11],[7,15]]
    assert c.to_vector() == [1.0, 2.0, 3.0, 4.0, 5.0, 11.0, 7.0, 15.0]


def test_matmul_3d_2d_broadcast():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])
    b = _cpp_ext.TensorImpl([1.0, 0.0, 0.0, 1.0], [2, 2])  # 单位矩阵
    c = _cpp_ext.matmul(a, b)
    assert c.shape == [2, 2, 2]
    # 每个batch @ I = 原矩阵
    assert c.to_vector() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]