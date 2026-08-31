"""Ch9 C++ 高级特性测试：autograd hooks + anomaly detection。

验证梯度钩子（register_hook/clear_hook）和异常检测（anomaly check）。
对应 torch.autograd 中 tensor.register_hook() 和 torch.autograd.detect_anomaly()。
"""

import pytest

from minitorch import _has_cpp

pytestmark = pytest.mark.skipif(not _has_cpp, reason="Ch9: C++ 扩展未编译")


# ── Hooks ──────────────────────────────────────────────


def test_hook_modify_grad():
    """钩子可以修改梯度：register_hook(lambda g: g * 2) 使梯度翻倍。"""
    from minitorch import _cpp_ext

    x = _cpp_ext.TensorImpl([1.0, 2.0, 3.0], [3], True)
    y = _cpp_ext.autograd_mul(x, x)  # y = x^2, dy/dx = 2x
    y.backward(_cpp_ext.TensorImpl([1.0, 1.0, 1.0], [3], False))

    grad_before = x.grad.to_vector()
    assert grad_before == pytest.approx([2.0, 4.0, 6.0])

    # 重新建图，注册钩子使梯度翻倍
    x2 = _cpp_ext.TensorImpl([1.0, 2.0, 3.0], [3], True)
    x2.register_hook(lambda g: _cpp_ext.autograd_mul(g, _cpp_ext.TensorImpl([2.0], [1], False).expand([3])))
    y2 = _cpp_ext.autograd_mul(x2, x2)
    y2.backward(_cpp_ext.TensorImpl([1.0, 1.0, 1.0], [3], False))

    grad_after = x2.grad.to_vector()
    assert grad_after == pytest.approx([4.0, 8.0, 12.0])


def test_hook_return_none():
    """钩子返回 None 时梯度不变。"""
    from minitorch import _cpp_ext

    x = _cpp_ext.TensorImpl([1.0, 2.0], [2], True)
    x.register_hook(lambda g: None)  # no-op hook
    y = _cpp_ext.autograd_mul(x, x)
    y.backward(_cpp_ext.TensorImpl([1.0, 1.0], [2], False))

    assert x.grad.to_vector() == pytest.approx([2.0, 4.0])



def test_clear_hook():
    """clear_hook 后梯度恢复正常。"""
    from minitorch import _cpp_ext

    x = _cpp_ext.TensorImpl([1.0, 2.0], [2], True)
    x.register_hook(lambda g: _cpp_ext.autograd_neg(g))
    x.clear_hook()
    y = _cpp_ext.autograd_mul(x, x)
    y.backward(_cpp_ext.TensorImpl([1.0, 1.0], [2], False))

    assert x.grad.to_vector() == pytest.approx([2.0, 4.0])


def test_hook_negate_grad():
    """钩子取反梯度。"""
    from minitorch import _cpp_ext

    x = _cpp_ext.TensorImpl([3.0, 5.0], [2], True)
    x.register_hook(lambda g: _cpp_ext.autograd_neg(g))
    y = _cpp_ext.autograd_sum(_cpp_ext.autograd_mul(x, x), -1, False)  # y = sum(x^2)
    y.backward(_cpp_ext.TensorImpl([1.0], [1], False))

    # dy/dx = 2x, hook 取反 → -2x
    assert x.grad.to_vector() == pytest.approx([-6.0, -10.0])


# ── Anomaly Detection ──────────────────────────────────


def test_anomaly_normal_backward():
    """正常 backward 不触发 anomaly。"""
    from minitorch import _cpp_ext

    _cpp_ext.set_anomaly_check_enabled(True)
    try:
        x = _cpp_ext.TensorImpl([1.0, 2.0, 3.0], [3], True)
        y = _cpp_ext.autograd_sum(_cpp_ext.autograd_mul(x, x), -1, False)
        y.backward(_cpp_ext.TensorImpl([1.0], [1], False))
        assert x.grad.to_vector() == pytest.approx([2.0, 4.0, 6.0])
    finally:
        _cpp_ext.set_anomaly_check_enabled(False)


def test_anomaly_detect_inf_grad():
    """sqrt(0) 的梯度为 inf，anomaly check 应抛异常。"""
    from minitorch import _cpp_ext

    _cpp_ext.set_anomaly_check_enabled(True)
    try:
        x = _cpp_ext.TensorImpl([0.0], [1], True)
        y = _cpp_ext.autograd_sqrt(x)  # dy/dx = 1/(2*sqrt(x)) → inf at x=0
        with pytest.raises(RuntimeError, match="Anomaly detected"):
            y.backward(_cpp_ext.TensorImpl([1.0], [1], False))
    finally:
        _cpp_ext.set_anomaly_check_enabled(False)


def test_anomaly_detect_nan_grad():
    """log(0) 的梯度为 1/0 = inf，进一步操作可产生 NaN。"""
    from minitorch import _cpp_ext

    _cpp_ext.set_anomaly_check_enabled(True)
    try:
        x = _cpp_ext.TensorImpl([0.0], [1], True)
        y = _cpp_ext.autograd_log(x)  # dy/dx = 1/x → inf at x=0
        with pytest.raises(RuntimeError, match="Anomaly detected"):
            y.backward(_cpp_ext.TensorImpl([1.0], [1], False))
    finally:
        _cpp_ext.set_anomaly_check_enabled(False)


def test_anomaly_disabled_no_raise():
    """关闭 anomaly check 时，即使梯度含 inf 也不抛异常。"""
    from minitorch import _cpp_ext

    _cpp_ext.set_anomaly_check_enabled(False)
    x = _cpp_ext.TensorImpl([0.0], [1], True)
    y = _cpp_ext.autograd_sqrt(x)
    y.backward(_cpp_ext.TensorImpl([1.0], [1], False))
    # grad 应为 inf，但不抛异常
    grad_val = x.grad.to_vector()[0]
    assert grad_val == pytest.approx(float("inf")) or grad_val != grad_val  # inf or nan


def test_anomaly_query():
    """查询 anomaly check 状态。"""
    from minitorch import _cpp_ext

    assert _cpp_ext.is_anomaly_check_enabled() is False
    _cpp_ext.set_anomaly_check_enabled(True)
    assert _cpp_ext.is_anomaly_check_enabled() is True
    _cpp_ext.set_anomaly_check_enabled(False)
    assert _cpp_ext.is_anomaly_check_enabled() is False