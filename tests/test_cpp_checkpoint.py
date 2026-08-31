"""Ch9 C++ 高级特性测试：gradient checkpointing。

验证 checkpoint 机制：前向不保存中间激活，backward 时重计算。
对应 torch.utils.checkpoint.checkpoint。
"""

import pytest

from minitorch import _has_cpp

pytestmark = pytest.mark.skipif(not _has_cpp, reason="Ch9: C++ 扩展未编译")


def test_checkpoint_basic_grad():
    """checkpoint 基本梯度正确：y = x^2, dy/dx = 2x。"""
    from minitorch import _cpp_ext

    x = _cpp_ext.TensorImpl([1.0, 2.0, 3.0], [3], True)

    def fn(inputs):
        v = inputs[0]
        return _cpp_ext.autograd_mul(v, v)

    y = _cpp_ext.checkpoint(fn, [x])
    y.backward(_cpp_ext.TensorImpl([1.0, 1.0, 1.0], [3], False))

    assert x.grad.to_vector() == pytest.approx([2.0, 4.0, 6.0])


def test_checkpoint_vs_no_checkpoint():
    """checkpoint 与普通 autograd 结果一致。"""
    from minitorch import _cpp_ext

    # 普通前向
    x1 = _cpp_ext.TensorImpl([0.5, 1.5, 2.0], [3], True)
    y1 = _cpp_ext.autograd_sum(
        _cpp_ext.autograd_mul(x1, x1), -1, False
    )  # y = sum(x^2)
    y1.backward(_cpp_ext.TensorImpl([1.0], [1], False))
    expected = x1.grad.to_vector()

    # checkpoint 前向
    x2 = _cpp_ext.TensorImpl([0.5, 1.5, 2.0], [3], True)

    def fn(inputs):
        v = inputs[0]
        return _cpp_ext.autograd_sum(_cpp_ext.autograd_mul(v, v), -1, False)

    y2 = _cpp_ext.checkpoint(fn, [x2])
    y2.backward(_cpp_ext.TensorImpl([1.0], [1], False))
    actual = x2.grad.to_vector()

    assert actual == pytest.approx(expected)


def test_checkpoint_multi_layer():
    """多层 checkpoint：y = relu(x^2 + x), dy/dx = (2x + 1) * (x^2+x > 0)。"""
    from minitorch import _cpp_ext

    # 普通前向
    x1 = _cpp_ext.TensorImpl([0.3, -0.5, 1.2], [3], True)
    a1 = _cpp_ext.autograd_mul(x1, x1)
    b1 = _cpp_ext.autograd_add(a1, x1)
    y1 = _cpp_ext.autograd_relu(b1)
    y1.backward(_cpp_ext.TensorImpl([1.0, 1.0, 1.0], [3], False))
    expected = x1.grad.to_vector()

    # checkpoint 前向
    x2 = _cpp_ext.TensorImpl([0.3, -0.5, 1.2], [3], True)

    def fn(inputs):
        v = inputs[0]
        a = _cpp_ext.autograd_mul(v, v)
        b = _cpp_ext.autograd_add(a, v)
        return _cpp_ext.autograd_relu(b)

    y2 = _cpp_ext.checkpoint(fn, [x2])
    y2.backward(_cpp_ext.TensorImpl([1.0, 1.0, 1.0], [3], False))
    actual = x2.grad.to_vector()

    assert actual == pytest.approx(expected)


def test_checkpoint_no_grad_input():
    """输入不需要梯度时，checkpoint 退化为普通前向。"""
    from minitorch import _cpp_ext

    x = _cpp_ext.TensorImpl([1.0, 2.0], [2], False)  # requires_grad=False

    def fn(inputs):
        return _cpp_ext.autograd_mul(inputs[0], inputs[0])

    y = _cpp_ext.checkpoint(fn, [x])
    assert y.grad_fn is None
    assert y.to_vector() == pytest.approx([1.0, 4.0])


def test_checkpoint_chained():
    """两个 checkpoint 串联。"""
    from minitorch import _cpp_ext

    # 普通前向: y = (x^2)^2 = x^4, dy/dx = 4x^3
    x1 = _cpp_ext.TensorImpl([1.0, 2.0], [2], True)
    a1 = _cpp_ext.autograd_mul(x1, x1)
    y1 = _cpp_ext.autograd_mul(a1, a1)
    y1.backward(_cpp_ext.TensorImpl([1.0, 1.0], [2], False))
    expected = x1.grad.to_vector()

    # 两个 checkpoint 串联
    x2 = _cpp_ext.TensorImpl([1.0, 2.0], [2], True)

    def fn1(inputs):
        return _cpp_ext.autograd_mul(inputs[0], inputs[0])

    a2 = _cpp_ext.checkpoint(fn1, [x2])

    def fn2(inputs):
        return _cpp_ext.autograd_mul(inputs[0], inputs[0])

    y2 = _cpp_ext.checkpoint(fn2, [a2])
    y2.backward(_cpp_ext.TensorImpl([1.0, 1.0], [2], False))
    actual = x2.grad.to_vector()

    assert actual == pytest.approx(expected)


def test_checkpoint_matmul():
    """checkpoint 用于矩阵乘法：y = sum(A @ B)。"""
    from minitorch import _cpp_ext

    A_data = [1.0, 2.0, 3.0, 4.0]  # 2x2
    B_data = [0.5, 1.0, 1.5, 2.0]  # 2x2

    # 普通前向
    A1 = _cpp_ext.TensorImpl(A_data, [2, 2], True)
    B1 = _cpp_ext.TensorImpl(B_data, [2, 2], True)
    C1 = _cpp_ext.autograd_matmul(A1, B1)
    y1 = _cpp_ext.autograd_sum(C1, -1, False)
    y1.backward(_cpp_ext.TensorImpl([1.0], [1], False))
    expected_A = A1.grad.to_vector()
    expected_B = B1.grad.to_vector()

    # checkpoint 前向
    A2 = _cpp_ext.TensorImpl(A_data, [2, 2], True)
    B2 = _cpp_ext.TensorImpl(B_data, [2, 2], True)

    def fn(inputs):
        C = _cpp_ext.autograd_matmul(inputs[0], inputs[1])
        return _cpp_ext.autograd_sum(C, -1, False)

    y2 = _cpp_ext.checkpoint(fn, [A2, B2])
    y2.backward(_cpp_ext.TensorImpl([1.0], [1], False))

    assert A2.grad.to_vector() == pytest.approx(expected_A)
    assert B2.grad.to_vector() == pytest.approx(expected_B)