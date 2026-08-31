"""Ch8 C++ Allocator 测试。

验证自定义 Allocator（DefaultAllocator/PoolAllocator）的统计和内存池功能。
"""


import pytest

from minitorch import _has_cpp

pytestmark = pytest.mark.skipif(not _has_cpp, reason="Ch8: C++ 扩展未编译")


@pytest.fixture
def C():
    from minitorch import _cpp_ext
    return _cpp_ext


def test_default_allocator_stats(C):
    alloc = C.DefaultAllocator()
    assert alloc.name == "DefaultAllocator"
    assert alloc.total_allocated == 0
    assert alloc.num_allocations == 0


def test_pool_allocator_basic(C):
    alloc = C.PoolAllocator(pool_threshold=1024)
    assert alloc.name == "PoolAllocator"
    assert alloc.pool_hits == 0
    assert alloc.pool_misses == 0
    assert alloc.pool_size == 0


def test_allocator_with_tensors(C):
    default_alloc = C.DefaultAllocator()
    C.set_global_allocator(default_alloc)

    t = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert t.to_vector() == [1.0, 2.0, 3.0, 4.0]
    assert default_alloc.num_allocations > 0
    assert default_alloc.total_allocated > 0


def test_pool_allocator_reuse(C):
    pool_alloc = C.PoolAllocator(pool_threshold=1024 * 1024)
    C.set_global_allocator(pool_alloc)

    t1 = C.TensorImpl([1.0, 2.0, 3.0, 4.0], [4])
    del t1
    misses_after_first = pool_alloc.pool_misses

    t2 = C.TensorImpl([5.0, 6.0, 7.0, 8.0], [4])
    del t2

    assert pool_alloc.pool_misses == misses_after_first
    assert pool_alloc.pool_hits > 0


def test_restore_default_allocator(C):
    default_alloc = C.DefaultAllocator()
    C.set_global_allocator(default_alloc)

    t = C.TensorImpl([1.0, 2.0], [2])
    assert t.to_vector() == [1.0, 2.0]