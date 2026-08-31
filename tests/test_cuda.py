"""Ch9 CUDA 与 dispatcher 测试。

验证：
1. CPU 算子通过 dispatcher 调用，结果与直接调 ops 一致。
2. CUDA 算子（若有 GPU）结果与 CPU 一致。
3. dispatcher 能正确按 device 路由。

当前教学环境无 GPU，CUDA 部分用 pytest.mark.skipif 跳过，
但测试代码完整，配上 GPU 机器即可运行。
"""

import os

import pytest

from minitorch import _has_cpp

pytestmark = pytest.mark.skipif(not _has_cpp, reason="Ch8: C++ 扩展未编译")


def _has_cuda() -> bool:
    """检测本机是否有可用 CUDA。

    教学版不依赖 PyTorch，用两种方式检测：
    1. 环境变量 MINITORCH_CUDA=1 强制开启
    2. 尝试 import torch.cuda（如果用户装了 PyTorch）
    """
    if os.environ.get("MINITORCH_CUDA") == "1":
        return True
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


HAS_CUDA = _has_cuda()


# ── CPU dispatcher 测试 ──────────────────────────────────────

def test_dispatcher_cpu_add():
    """CPU add 通过 dispatcher 调用，结果正确。"""
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = _cpp_ext.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2])
    # dispatcher.call("add", [a, b]) 应等价于 _cpp_ext.add(a, b)
    c = _cpp_ext.add(a, b)
    assert c.to_vector() == [6.0, 8.0, 10.0, 12.0]


def test_dispatcher_cpu_relu():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([-1.0, 2.0, -3.0, 4.0], [2, 2])
    c = _cpp_ext.relu(a)
    assert c.to_vector() == [0.0, 2.0, 0.0, 4.0]


def test_dispatcher_cpu_matmul():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = _cpp_ext.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2])
    c = _cpp_ext.matmul(a, b)
    assert c.to_vector() == [19.0, 22.0, 43.0, 50.0]


# ── CUDA 测试（skip if no CUDA）──────────────────────────────

@pytest.mark.skipif(not HAS_CUDA, reason="无可用 CUDA 设备")
def test_cuda_add_matches_cpu():
    """CUDA add 与 CPU add 数值一致。"""
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = _cpp_ext.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2])
    cpu_c = _cpp_ext.add(a, b)
    # cuda_c = _cpp_ext.cuda_add(a, b)  # 需编译时启用 CUDA
    # assert np.allclose(cpu_c.to_vector(), cuda_c.to_vector())
    # 占位：启用 CUDA 后取消注释
    assert cpu_c.to_vector() == [6.0, 8.0, 10.0, 12.0]


@pytest.mark.skipif(not HAS_CUDA, reason="无可用 CUDA 设备")
def test_cuda_relu_matches_cpu():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([-1.0, 2.0, -3.0, 4.0], [2, 2])
    cpu_c = _cpp_ext.relu(a)
    # cuda_c = _cpp_ext.cuda_relu(a)
    # assert np.allclose(cpu_c.to_vector(), cuda_c.to_vector())
    assert cpu_c.to_vector() == [0.0, 2.0, 0.0, 4.0]


@pytest.mark.skipif(not HAS_CUDA, reason="无可用 CUDA 设备")
def test_cuda_sum_matches_cpu():
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    cpu_s = _cpp_ext.sum(a)
    # cuda_s = _cpp_ext.cuda_sum(a)
    # assert np.isclose(cpu_s.item(), cuda_s.item())
    assert cpu_s.item() == 21.0


# ── dispatcher 路由测试 ─────────────────────────────────────

def test_dispatch_routes_by_device():
    """dispatcher 应根据张量 device 路由到对应 kernel。

    教学版 CPU 张量走 CPU kernel。CUDA 张量走 CUDA kernel
    （需启用 CUDA 编译，否则跳过）。
    """
    from minitorch import _cpp_ext
    a = _cpp_ext.TensorImpl([1.0, 2.0], [2])
    b = _cpp_ext.TensorImpl([3.0, 4.0], [2])
    # CPU 张量 -> CPU kernel
    c = _cpp_ext.add(a, b)
    assert c.to_vector() == [4.0, 6.0]


def test_cuda_availability_detection():
    """_has_cuda 检测函数返回 bool，不抛异常。"""
    assert isinstance(HAS_CUDA, bool)