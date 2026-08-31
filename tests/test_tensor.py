"""Ch1 张量与存储测试。"""

import numpy as np

from minitorch import Storage, Tensor


def test_storage_basic():
    s = Storage(data=[1, 2, 3, 4])
    assert len(s) == 4
    assert s[2] == 3
    s[2] = 30
    assert s[2] == 30
    s.resize(8)
    assert len(s) == 8
    assert s[0] == 1


def test_tensor_construct_and_attrs():
    t = Tensor.from_numpy(np.arange(6).reshape(2, 3))
    assert t.shape == (2, 3)
    assert t.strides == (3, 1)
    assert t.ndim == 2
    assert t.size == 6
    assert t.is_contiguous()
    assert t.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_view_shares_storage():
    t = Tensor.from_numpy(np.arange(6).reshape(2, 3))
    v = t.view(3, 2)
    assert v.storage is t.storage
    v[0, 0] = 999
    assert t[0, 0].item() == 999


def test_transpose_stride_and_contiguous():
    t = Tensor.from_numpy(np.arange(6).reshape(2, 3))
    tt = t.transpose()
    assert tt.shape == (3, 2)
    assert tt.strides == (1, 3)
    assert not tt.is_contiguous()
    assert tt.tolist() == [[0, 3], [1, 4], [2, 5]]


def test_contiguous_materialize():
    t = Tensor.from_numpy(np.arange(6).reshape(2, 3))
    tt = t.transpose()
    c = tt.contiguous()
    assert c.is_contiguous()
    assert c.strides == (2, 1)
    assert c.tolist() == [[0, 3], [1, 4], [2, 5]]
    assert c.storage is not tt.storage


def test_reshape_copy_when_non_contiguous():
    t = Tensor.from_numpy(np.arange(6).reshape(2, 3))
    tt = t.transpose()
    r = tt.reshape(6)
    assert r.tolist() == [0, 3, 1, 4, 2, 5]
    assert r.storage is not tt.storage


def test_reshape_view_when_contiguous():
    t = Tensor.from_numpy(np.arange(6).reshape(2, 3))
    r = t.reshape(6)
    assert r.storage is t.storage


def test_unsqueeze_squeeze():
    t = Tensor.from_numpy(np.arange(3))
    u = t.unsqueeze(0)
    assert u.shape == (1, 3)
    assert u.strides == (0, 1)
    s = u.squeeze(0)
    assert s.shape == (3,)


def test_permute():
    t = Tensor.from_numpy(np.arange(24).reshape(2, 3, 4))
    p = t.permute(2, 0, 1)
    assert p.shape == (4, 2, 3)
    assert p[0, 0, 0].item() == t[0, 0, 0].item()
    assert p[3, 1, 2].item() == t[1, 2, 3].item()


def test_broadcast_tensors():
    a = Tensor.from_numpy(np.ones((3, 4)))
    b = Tensor.from_numpy(np.arange(4))
    ba, bb = Tensor.broadcast_tensors(a, b)
    assert ba.shape == (3, 4)
    assert bb.shape == (3, 4)
    assert bb.strides == (0, 1)


def test_broadcast_stride_zero():
    a = Tensor.from_numpy(np.arange(3))
    b = a.broadcast_to((2, 3))
    assert b.shape == (2, 3)
    assert b.strides == (0, 1)
    assert b[0].tolist() == b[1].tolist()


def test_arithmetic_with_broadcast():
    a = Tensor.from_numpy(np.ones((3, 4)))
    b = Tensor.from_numpy(np.arange(4))
    c = a + b
    assert c.shape == (3, 4)
    assert c[0].tolist() == [1, 2, 3, 4]
    assert c[1].tolist() == [1, 2, 3, 4]


def test_matmul():
    a = Tensor.from_numpy(np.arange(6).reshape(2, 3))
    b = Tensor.from_numpy(np.arange(12).reshape(3, 4))
    c = a @ b
    assert c.shape == (2, 4)
    np.testing.assert_allclose(c.numpy(), np.arange(6).reshape(2, 3) @ np.arange(12).reshape(3, 4))


def test_sum_mean():
    t = Tensor.from_numpy(np.arange(6).reshape(2, 3).astype(float))
    assert t.sum().item() == 15
    assert t.mean().item() == 2.5
    s = t.sum(dim=1)
    assert s.shape == (2,)
    assert s.tolist() == [3, 12]


def test_indexing():
    t = Tensor.from_numpy(np.arange(6).reshape(2, 3))
    assert t[0].tolist() == [0, 1, 2]
    assert t[1, 2].item() == 5
    assert t[:, 1].tolist() == [1, 4]


def test_neg_and_sub():
    t = Tensor.from_numpy(np.arange(3).astype(float))
    assert (-t).tolist() == [0, -1, -2]
    assert (t - 1).tolist() == [-1, 0, 1]


def test_factory_methods():
    assert Tensor.zeros(2, 3).tolist() == [[0, 0, 0], [0, 0, 0]]
    assert Tensor.ones(2).tolist() == [1, 1]
    assert Tensor.arange(5).tolist() == [0, 1, 2, 3, 4]


def test_T_property():
    t = Tensor.from_numpy(np.arange(6).reshape(2, 3))
    assert t.T.shape == (3, 2)
    assert t.T.tolist() == [[0, 3], [1, 4], [2, 5]]
    one_d = Tensor.from_numpy(np.arange(3))
    assert one_d.T.shape == (3,)


def test_allclose():
    a = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    b = Tensor.from_numpy(np.array([1.0 + 1e-10, 2.0, 3.0]))
    assert a.allclose(b)
