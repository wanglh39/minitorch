"""test_cpp_profiler：autograd profiler 测试（Ch9 深入）。"""

from minitorch import _cpp_ext as _C


def test_profiler_basic():
    a = _C.TensorImpl([1.0, 2.0, 3.0], [3], True)
    b = _C.TensorImpl([4.0, 5.0, 6.0], [3], True)
    c = _C.autograd_mul(a, b)
    s = _C.autograd_sum(c, -1, False)

    _C.profiler_start()
    s.backward(_C.TensorImpl([1.0], []))
    _C.profiler_stop()

    events = _C.profiler_events()
    assert len(events) > 0
    names = [e[0] for e in events]
    assert "Sum" in names
    assert "Mul" in names
    assert "AccumulateGrad" in names


def test_profiler_timing_positive():
    a = _C.TensorImpl([1.0, 2.0, 3.0], [3], True)
    b = _C.TensorImpl([4.0, 5.0, 6.0], [3], True)
    c = _C.autograd_matmul(a.reshape([1, 3]), b.reshape([3, 1]))
    s = _C.autograd_sum(c, -1, False)

    _C.profiler_start()
    s.backward(_C.TensorImpl([1.0], []))
    _C.profiler_stop()

    events = _C.profiler_events()
    for _name, duration, _mem_before, _mem_after, _tid in events:
        assert duration >= 0.0


def test_profiler_disabled_by_default():
    assert not _C.profiler_enabled()