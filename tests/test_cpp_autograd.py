"""Ch8 C++ autograd 测试。

验证 C++ autograd（Node/Engine/AccumulateGrad）行为与 Python 实现等价。
覆盖：add/sub/mul/div/neg/relu/matmul/sum/mean/transpose 的前向+反向，
链式求导、广播反向、no_grad、retain_graph。
"""


import pytest

from minitorch import _has_cpp

pytestmark = pytest.mark.skipif(not _has_cpp, reason="Ch8: C++ 扩展未编译")


@pytest.fixture
def C():
    from minitorch import _cpp_ext
    return _cpp_ext


def approx(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b, strict=True))


def scalar_backward(C, t):
    """对非标量张量：sum 后 backward。"""
    s = C.autograd_sum(t)
    s.backward()
    return s


# ── 基本建图 ──────────────────────────────────────────

def test_add_backward(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    b = C.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2], requires_grad=True)
    c = C.autograd_add(a, b)
    assert c.requires_grad
    assert c.grad_fn is not None
    assert c.grad_fn.name == "Add"
    scalar_backward(C, c)
    assert approx(a.grad.to_vector(), [1.0, 1.0, 1.0, 1.0])
    assert approx(b.grad.to_vector(), [1.0, 1.0, 1.0, 1.0])


def test_sub_backward(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    b = C.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2], requires_grad=True)
    c = C.autograd_sub(a, b)
    scalar_backward(C, c)
    assert approx(a.grad.to_vector(), [1.0, 1.0, 1.0, 1.0])
    assert approx(b.grad.to_vector(), [-1.0, -1.0, -1.0, -1.0])


def test_mul_backward(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    b = C.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2], requires_grad=True)
    c = C.autograd_mul(a, b)
    scalar_backward(C, c)
    # da = b, db = a
    assert approx(a.grad.to_vector(), [5.0, 6.0, 7.0, 8.0])
    assert approx(b.grad.to_vector(), [1.0, 2.0, 3.0, 4.0])


def test_div_backward(C):
    a = C.TensorImpl([4.0, 6.0], [2], requires_grad=True)
    b = C.TensorImpl([2.0, 3.0], [2], requires_grad=True)
    c = C.autograd_div(a, b)
    scalar_backward(C, c)
    # da = 1/b, db = -a/b^2
    assert approx(a.grad.to_vector(), [0.5, 1.0 / 3.0])
    assert approx(b.grad.to_vector(), [-1.0, -2.0 / 3.0])


def test_neg_backward(C):
    a = C.TensorImpl([1.0, -2.0, 3.0], [3], requires_grad=True)
    c = C.autograd_neg(a)
    scalar_backward(C, c)
    assert approx(a.grad.to_vector(), [-1.0, -1.0, -1.0])


def test_relu_backward(C):
    a = C.TensorImpl([-1.0, 2.0, -3.0, 4.0], [4], requires_grad=True)
    c = C.autograd_relu(a)
    assert approx(c.to_vector(), [0.0, 2.0, 0.0, 4.0])
    scalar_backward(C, c)
    # relu'(x) = 1 if x > 0 else 0
    assert approx(a.grad.to_vector(), [0.0, 1.0, 0.0, 1.0])


# ── 矩阵乘法反向 ──────────────────────────────────────

def test_matmul_backward(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    b = C.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2], requires_grad=True)
    c = C.autograd_matmul(a, b)
    # c = [[19, 22], [43, 50]]
    assert approx(c.to_vector(), [19.0, 22.0, 43.0, 50.0])
    s = C.autograd_sum(c)
    s.backward()
    # da = ones @ b^T, db = a^T @ ones
    # b^T = [[5,7],[6,8]], ones @ b^T = [[11,15],[11,15]]
    assert approx(a.grad.to_vector(), [11.0, 15.0, 11.0, 15.0])
    # a^T = [[1,3],[2,4]], a^T @ ones = [[4,4],[6,6]]
    assert approx(b.grad.to_vector(), [4.0, 4.0, 6.0, 6.0])


# ── 归约反向 ──────────────────────────────────────────

def test_sum_backward(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    s = C.autograd_sum(a)
    assert approx(s.to_vector(), [10.0])
    s.backward()
    assert approx(a.grad.to_vector(), [1.0, 1.0, 1.0, 1.0])


def test_mean_backward(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    m = C.autograd_mean(a)
    assert approx(m.to_vector(), [2.5])
    m.backward()
    # d mean / d a_i = 1/n = 0.25
    assert approx(a.grad.to_vector(), [0.25, 0.25, 0.25, 0.25])


# ── 转置反向 ──────────────────────────────────────────

def test_transpose_backward(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    t = C.autograd_transpose(a)
    s = C.autograd_sum(t)
    s.backward()
    assert approx(a.grad.to_vector(), [1.0, 1.0, 1.0, 1.0])


# ── 链式求导 ──────────────────────────────────────────

def test_chain_rule(C):
    a = C.TensorImpl([1.0, 2.0], [2], requires_grad=True)
    b = C.TensorImpl([3.0, 4.0], [2], requires_grad=True)
    # c = a * b = [3, 8]
    c = C.autograd_mul(a, b)
    # d = c + a = [4, 10]
    d = C.autograd_add(c, a)
    # s = sum(d) = 14
    s = C.autograd_sum(d)
    assert approx(s.to_vector(), [14.0])
    s.backward()
    # ds/dd = [1, 1]
    # dd/da = b + 1 = [4, 5], dd/db = a = [1, 2]
    assert approx(a.grad.to_vector(), [4.0, 5.0])
    assert approx(b.grad.to_vector(), [1.0, 2.0])


def test_quadratic_gradient(C):
    x = C.TensorImpl([3.0], [1], requires_grad=True)
    # y = x^2 = x * x, dy/dx = 2x = 6
    y = C.autograd_mul(x, x)
    y.backward()
    assert approx(x.grad.to_vector(), [6.0])


def test_nested_computation(C):
    a = C.TensorImpl([1.0, 2.0, 3.0], [3], requires_grad=True)
    # b = relu(a) = [1, 2, 3]
    b = C.autograd_relu(a)
    # c = b * b = [1, 4, 9]
    c = C.autograd_mul(b, b)
    # s = sum(c) = 14
    s = C.autograd_sum(c)
    assert approx(s.to_vector(), [14.0])
    s.backward()
    # ds/da = 2 * relu(a) * relu'(a) = 2 * a (since a > 0)
    assert approx(a.grad.to_vector(), [2.0, 4.0, 6.0])


# ── 广播反向 ──────────────────────────────────────────

def test_broadcast_add_backward(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    b = C.TensorImpl([10.0, 20.0], [2], requires_grad=True)  # 广播
    c = C.autograd_add(a, b)
    s = C.autograd_sum(c)
    s.backward()
    # da = ones
    assert approx(a.grad.to_vector(), [1.0, 1.0, 1.0, 1.0])
    # db = sum over rows = [2, 2]
    assert approx(b.grad.to_vector(), [2.0, 2.0])


def test_broadcast_mul_backward(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    b = C.TensorImpl([10.0, 20.0], [2], requires_grad=True)  # 广播
    c = C.autograd_mul(a, b)
    s = C.autograd_sum(c)
    s.backward()
    # da = broadcast(b) = [[10,20],[10,20]]
    assert approx(a.grad.to_vector(), [10.0, 20.0, 10.0, 20.0])
    # db = sum over rows of a = [1+3, 2+4] = [4, 6]
    assert approx(b.grad.to_vector(), [4.0, 6.0])


# ── no_grad ───────────────────────────────────────────

def test_no_grad_mode(C):
    a = C.TensorImpl([1.0, 2.0], [2], requires_grad=True)
    b = C.TensorImpl([3.0, 4.0], [2], requires_grad=True)
    assert C.is_grad_enabled()
    prev = C.set_grad_enabled(False)
    try:
        c = C.autograd_add(a, b)
        assert not c.requires_grad
        assert c.grad_fn is None
    finally:
        C.set_grad_enabled(prev)


def test_no_grad_does_not_build_graph(C):
    a = C.TensorImpl([1.0, 2.0], [2], requires_grad=True)
    b = C.TensorImpl([3.0, 4.0], [2], requires_grad=True)
    prev = C.set_grad_enabled(False)
    try:
        c = C.autograd_mul(a, b)
        assert c.grad_fn is None
    finally:
        C.set_grad_enabled(prev)


# ── retain_graph ──────────────────────────────────────

def test_retain_graph(C):
    a = C.TensorImpl([1.0, 2.0], [2], requires_grad=True)
    b = C.TensorImpl([3.0, 4.0], [2], requires_grad=True)
    c = C.autograd_mul(a, b)
    s = C.autograd_sum(c)
    s.backward(retain_graph=True)
    assert approx(a.grad.to_vector(), [3.0, 4.0])
    assert approx(b.grad.to_vector(), [1.0, 2.0])
    # 可以再次 backward
    a.set_grad(None)
    b.set_grad(None)
    s.backward(retain_graph=True)
    assert approx(a.grad.to_vector(), [3.0, 4.0])


# ── 梯度累加 ──────────────────────────────────────────

def test_gradient_accumulation(C):
    a = C.TensorImpl([1.0, 2.0], [2], requires_grad=True)
    b = C.TensorImpl([3.0, 4.0], [2], requires_grad=True)
    c = C.autograd_mul(a, b)
    s1 = C.autograd_sum(c)
    s1.backward(retain_graph=True)
    # 第二次使用 a（模拟多次出现）
    d = C.autograd_add(a, b)
    s2 = C.autograd_sum(d)
    s2.backward()
    # a.grad 应累加：第一次 da=[3,4]，第二次 da=[1,1]，总计 [4,5]
    assert approx(a.grad.to_vector(), [4.0, 5.0])


# ── 线性层模拟 ────────────────────────────────────────

def test_linear_layer_gradient(C):
    W = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    x = C.TensorImpl([1.0, 1.0, 1.0, 1.0], [2, 2], requires_grad=True)
    # y = W @ x = [[3, 3], [7, 7]]
    y = C.autograd_matmul(W, x)
    assert approx(y.to_vector(), [3.0, 3.0, 7.0, 7.0])
    # loss = sum(y) = 20
    loss = C.autograd_sum(y)
    assert approx(loss.to_vector(), [20.0])
    loss.backward()
    # dW = grad_y @ x^T = ones @ ones^T = [[2,2],[2,2]]
    assert approx(W.grad.to_vector(), [2.0, 2.0, 2.0, 2.0])
    # dx = W^T @ grad_y = W^T @ ones_2x2 = [[4,4],[6,6]]
    assert approx(x.grad.to_vector(), [4.0, 4.0, 6.0, 6.0])


# ── matmul 批量化反向 ──────────────────────────────────

def test_matmul_1d_1d_backward(C):
    a = C.TensorImpl([1.0, 2.0, 3.0], [3], requires_grad=True)
    b = C.TensorImpl([4.0, 5.0, 6.0], [3], requires_grad=True)
    c = C.autograd_matmul(a, b)  # dot product = 32
    assert abs(c.item() - 32.0) < 1e-6
    c.backward()
    # da = b, db = a
    assert approx(a.grad.to_vector(), [4.0, 5.0, 6.0])
    assert approx(b.grad.to_vector(), [1.0, 2.0, 3.0])


def test_matmul_1d_2d_backward(C):
    a = C.TensorImpl([1.0, 2.0], [2], requires_grad=True)
    b = C.TensorImpl([3.0, 4.0, 5.0, 6.0], [2, 2], requires_grad=True)
    c = C.autograd_matmul(a, b)  # [13, 16]
    s = C.autograd_sum(c)
    s.backward()
    # da = b @ ones = [7, 11]
    assert approx(a.grad.to_vector(), [7.0, 11.0])
    # db = outer(a, ones) = [[1,1],[2,2]]
    assert approx(b.grad.to_vector(), [1.0, 1.0, 2.0, 2.0])


def test_matmul_2d_1d_backward(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    b = C.TensorImpl([5.0, 6.0], [2], requires_grad=True)
    c = C.autograd_matmul(a, b)  # [17, 39]
    s = C.autograd_sum(c)
    s.backward()
    # da = outer(ones, b) = [[5,6],[5,6]]
    assert approx(a.grad.to_vector(), [5.0, 6.0, 5.0, 6.0])
    # db = a^T @ ones = [4, 6]
    assert approx(b.grad.to_vector(), [4.0, 6.0])


def test_matmul_3d_batched_backward(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2], requires_grad=True)
    b = C.TensorImpl([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0], [2, 2, 2], requires_grad=True)
    c = C.autograd_matmul(a, b)  # [2,2,2]
    s = C.autograd_sum(c)
    s.backward()
    # da = ones @ b^T per batch
    # batch 0: b^T = [[1,0],[0,1]], ones @ b^T = [[1,1],[1,1]]
    # batch 1: b^T = [[1,0],[1,1]], ones @ b^T = [[2,1],[2,1]]
    assert approx(a.grad.to_vector(), [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0])
    # db = a^T @ ones per batch
    # batch 0: a^T = [[1,3],[2,4]], a^T @ ones = [[4,4],[6,6]]
    # batch 1: a^T = [[5,7],[6,8]], a^T @ ones = [[12,12],[14,14]]
    assert approx(b.grad.to_vector(), [4.0, 4.0, 6.0, 6.0, 12.0, 12.0, 14.0, 14.0])


# ── 多线程 Engine ──────────────────────────────────────

def test_backward_mt_basic(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    b = C.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2], requires_grad=True)
    c = C.autograd_mul(a, b)
    s = C.autograd_sum(c)
    s.backward_mt()
    assert approx(a.grad.to_vector(), [5.0, 6.0, 7.0, 8.0])
    assert approx(b.grad.to_vector(), [1.0, 2.0, 3.0, 4.0])


def test_backward_mt_chain(C):
    a = C.TensorImpl([1.0, 2.0, 3.0], [3], requires_grad=True)
    b = C.autograd_mul(a, a)
    c = C.autograd_sum(b)
    c.backward_mt()
    assert approx(a.grad.to_vector(), [2.0, 4.0, 6.0])


def test_backward_mt_matmul(C):
    a = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
    b = C.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2], requires_grad=True)
    c = C.autograd_matmul(a, b)
    s = C.autograd_sum(c)
    s.backward_mt()
    assert approx(a.grad.to_vector(), [11.0, 15.0, 11.0, 15.0])
    assert approx(b.grad.to_vector(), [4.0, 4.0, 6.0, 6.0])


def test_backward_mt_diamond(C):
    a = C.TensorImpl([2.0], [1], requires_grad=True)
    b = C.autograd_mul(a, a)
    c = C.autograd_mul(a, a)
    d = C.autograd_add(b, c)
    s = C.autograd_sum(d)
    s.backward_mt()
    assert approx(a.grad.to_vector(), [8.0])


def test_backward_mt_matches_sequential(C):
    import random
    random.seed(42)
    for _ in range(10):
        data_a = [random.uniform(-3, 3) for _ in range(6)]
        data_b = [random.uniform(-3, 3) for _ in range(6)]
        a1 = C.TensorImpl(data_a, [2, 3], requires_grad=True)
        b1 = C.TensorImpl(data_b, [3, 2], requires_grad=True)
        c1 = C.autograd_matmul(a1, b1)
        s1 = C.autograd_sum(c1)
        s1.backward()

        a2 = C.TensorImpl(data_a, [2, 3], requires_grad=True)
        b2 = C.TensorImpl(data_b, [3, 2], requires_grad=True)
        c2 = C.autograd_matmul(a2, b2)
        s2 = C.autograd_sum(c2)
        s2.backward_mt()

        assert approx(a1.grad.to_vector(), a2.grad.to_vector())
        assert approx(b1.grad.to_vector(), b2.grad.to_vector())


# ── Double Backward ────────────────────────────────────

def test_double_backward_mul(C):
    x = C.TensorImpl([1.0, 2.0, 3.0], [3], requires_grad=True)
    y = C.autograd_mul(x, x)  # y = x^2
    s = C.autograd_sum(y)
    s.backward(create_graph=True)
    assert approx(x.grad.to_vector(), [2.0, 4.0, 6.0])
    assert x.grad.grad_fn is not None

    g1 = x.grad.to_vector()
    g = x.grad
    g_sum = C.autograd_sum(g)
    g_sum.backward()
    g2 = x.grad.to_vector()
    second_deriv = [g2[i] - g1[i] for i in range(3)]
    assert approx(second_deriv, [2.0, 2.0, 2.0])


def test_double_backward_add(C):
    x = C.TensorImpl([3.0, 5.0], [2], requires_grad=True)
    y = C.autograd_add(x, x)  # y = 2x
    s = C.autograd_sum(y)
    s.backward(create_graph=True)
    assert approx(x.grad.to_vector(), [2.0, 2.0])
    # add is linear: second derivative = 0, x.grad has no grad_fn
