# Ch1 张量与存�?

## 本章目标

读完本章后，你应当能够：

1. 理解为什�?PyTorch 要把 Tensor �?Storage **分离**，而不是用一个对象包揽数据和元信�?
2. 手动推导任意 shape/stride 下元素的物理偏移公式
3. 解释 view、reshape、transpose 三者的区别和各自的适用场景
4. 理解广播（broadcasting）的 stride=0 技巧，以及它为什么能做到零拷�?
5. 对照真实 PyTorch C++ 源码�?`TensorImpl` 的设计，理解 Python 原型�?C++ 工程实现之间的映射关�?

---

## 1. 为什么需�?Tensor/Storage 分离�?

### 1.1 朴素方案的问�?

假设我们设计一个张量类，最直觉的做法是把数据和形状绑在一起：

```python
class NaiveTensor:
    def __init__(self, data: list, shape: tuple):
        self.data = data        # 一维列�?
        self.shape = shape      # 多维形状
```

看起来够用了。但考虑一个常见操作—�?*转置**�?

```python
a = NaiveTensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
# a = [[1, 2, 3],
#      [4, 5, 6]]

b = a.transpose()  # 期望 shape=(3, 2)
# b = [[1, 4],
#      [2, 5],
#      [3, 6]]
```

朴素方案要么 **拷贝数据**（把 `[1,2,3,4,5,6]` 重排�?`[1,4,2,5,3,6]`），要么维护一个额外的索引映射。前者慢（大张量拷贝昂贵），后者复杂�?

### 1.2 stride 的洞�?

关键观察：转置只�?**改变了访问顺�?*，数据本身没变。如果我们记�?沿每个维度前进一步对�?buffer 中跳几个元素"，就能在不拷贝数据的情况下表达转置�?

这就�?**stride**（步幅）的概念：

```
原始 a: shape=(2, 3), stride=(3, 1)
buffer = [1, 2, 3, 4, 5, 6]

a[0, 0] = buffer[0*3 + 0*1] = buffer[0] = 1
a[0, 1] = buffer[0*3 + 1*1] = buffer[1] = 2
a[1, 0] = buffer[1*3 + 0*1] = buffer[3] = 4

转置 b: shape=(3, 2), stride=(1, 3)  �?交换�?stride
buffer = [1, 2, 3, 4, 5, 6]          �?同一�?buffer�?

b[0, 0] = buffer[0*1 + 0*3] = buffer[0] = 1
b[0, 1] = buffer[0*1 + 1*3] = buffer[3] = 4
b[1, 0] = buffer[1*1 + 0*3] = buffer[1] = 2
```

转置只需交换 shape 的两个维度和 stride 的两个维度—�?*零拷�?*�?

### 1.3 Storage 的角�?

Storage 就是那个"被多�?Tensor 共享�?buffer"。Tensor �?Storage �?**视图**（view），持有�?

- `storage`：指�?Storage 的引�?
- `shape`：逻辑形状
- `strides`：每个维度的步幅
- `storage_offset`：在 Storage 中的起始偏移

```
┌───────────────────────────────────�?
�? Storage                          �?
�? data = [1, 2, 3, 4, 5, 6]       �?
�? dtype = float64                  �?
└───────▲───────────────────▲──────�?
        �?                  �?
┌───────┴───────�?  ┌───────┴───────�?
�? Tensor a      �?  �? Tensor b      �?
�? shape=(2,3)   �?  �? shape=(3,2)   �?
�? stride=(3,1)  �?  �? stride=(1,3)  �?
�? offset=0      �?  �? offset=0      �?
└────────────────�?  └────────────────�?
   原始矩阵             转置矩阵
```

### 1.4 storage_offset 的用�?

`storage_offset` �?**切片** 也零拷贝�?

```python
a = Tensor([1, 2, 3, 4, 5, 6], shape=(6,), stride=(1,), offset=0)
b = a[2:]  # 取索�?2 开始的子张�?
# b = [3, 4, 5, 6]
# b.shape = (4,), b.stride = (1,), b.storage_offset = 2
# b �?a 共享同一�?Storage�?
```

切片只改 `storage_offset` �?`shape`，不拷贝数据�?

---

## 2. stride 语义详解

### 2.1 偏移公式

给定 Tensor �?`strides = (s₀, s�? ..., sₙ₋�?` �?`storage_offset = o`�?
元素 `(i₀, i�? ..., iₙ₋�?` �?Storage buffer 中的位置为：

```
position = o + Σ�?i�?· s�?
```

### 2.2 contiguous stride 的推�?

一�?**连续**（contiguous）张量，�?stride 从右向左递推�?

```
sₙ₋�?= 1
s�?= sₖ₊�?· shapeₖ₊�?   (k from n-2 down to 0)
```

示例：shape = (2, 3, 4)

```
s�?= 1
s�?= s�?· shape�?= 1 · 4 = 4
s₀ = s�?· shape�?= 4 · 3 = 12
stride = (12, 4, 1)
```

验证：元�?(1, 2, 3) 的位�?= 1×12 + 2×4 + 3×1 = 23�?
buffer 大小 = 2×3×4 = 24，索�?23 是最后一个元素。✓

### 2.3 非连续的例子

转置后的张量不是连续的：

```python
a: shape=(2, 3), stride=(3, 1)  �?contiguous
b = a.transpose()
b: shape=(3, 2), stride=(1, 3)  �?NOT contiguous
```

检�?contiguous：stride 必须等于 `_compute_contiguous_strides(shape)`�?

```python
def is_contiguous(self):
    if self.ndim <= 1:
        return True
    return self._strides == _compute_contiguous_strides(self._shape)
```

### 2.4 为什�?contiguous 重要�?

很多操作要求连续内存�?

- **view**：改�?shape 需要重新计�?stride，只有连续时才能保证元素物理排列正确
- **GPU kernel**：很�?CUDA kernel 假设连续布局，简化索引计�?
- **序列�?*：保存到磁盘时需要连续的 buffer

非连续张量需要先 `.contiguous()`（触发拷贝）才能做这些操作�?

---

## 3. 代码逐行实现

### 3.1 Storage �?

```python
class Storage:
    """一维数据缓冲区，可被一个或多个 Tensor 共享�?""

    __slots__ = ("_data",)

    def __init__(self, data=None, size: int = 0, dtype=np.float64):
        if data is not None:
            arr = np.asarray(data, dtype=dtype).ravel()
        else:
            arr = np.zeros(size, dtype=dtype)
        self._data = arr
```

**逐行解释**�?

- `__slots__ = ("_data",)`：禁止动态添加属性，减少每个 Storage 对象的内存开销（约 40 字节 �?16 字节）。PyTorch C++ �?Storage �?`c10::StorageImpl`，同样紧凑�?

- `np.asarray(data, dtype=dtype).ravel()`：把输入转为一�?numpy array。`ravel()` 返回视图（不拷贝）如果可能，否则拷贝�?

- `dtype=np.float64`：默�?float64（教学优先精度）。PyTorch 默认 float32（推理速度）�?

```python
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Storage:
        arr = np.asarray(arr)
        return cls(data=arr, dtype=arr.dtype)
```

`from_numpy` 保留输入 dtype。这�?Ch12 AMP 中很重要——float16 �?Storage 不能被强制转 float64�?

### 3.2 Tensor 类——构造与属�?

```python
class Tensor:
    def __init__(
        self,
        storage: Storage,
        shape: Sequence[int],
        strides: Sequence[int],
        storage_offset: int = 0,
        requires_grad: bool = False,
    ):
        self._storage = storage          # 共享�?Storage 引用
        self._shape = tuple(shape)       # 逻辑形状，如 (2, 3)
        self._strides = tuple(strides)   # 步幅，如 (3, 1)
        self._storage_offset = storage_offset  # buffer 起始偏移
        self.requires_grad = requires_grad     # 是否需要梯�?
        self.grad: Tensor | None = None         # 梯度（反向后填充�?
        self.grad_fn = None                     # 创建�?Tensor �?Node
```

**关键设计�?*�?

- `_storage` 是引用而非拷贝——多�?Tensor 可以共享同一 Storage，这�?view 零拷贝的基础
- `_shape` �?`_strides` �?`tuple` 而非 `list`——不可变，避免意外修�?
- `requires_grad` / `grad` / `grad_fn` �?Ch1 不使用，预留�?Ch2-Ch3 �?autograd

### 3.3 _numpy_view——stride 技巧的核心

这是整个 Tensor 类最关键的方法：

```python
    def _numpy_view(self) -> np.ndarray:
        base = self._storage.data
        byte_strides = tuple(s * base.itemsize for s in self._strides)
        return np.lib.stride_tricks.as_strided(
            base[self._storage_offset:],
            shape=self._shape if self._shape else (),
            strides=byte_strides if self._shape else (),
        )
```

**逐行解释**�?

1. `base = self._storage.data`：拿到底层一�?numpy array
2. `byte_strides = tuple(s * base.itemsize for s in self._strides)`�?
   - numpy �?stride 单位�?**字节**，我们的 stride 单位�?**元素**
   - float64 �?itemsize=8，所�?stride=(3,1) �?byte_stride=(24,8)
3. `np.lib.stride_tricks.as_strided(...)`�?
   - 这是 numpy �?黑魔�?——创建一个虚拟视图，按指定的 shape �?byte stride 读写底层 buffer
   - **不拷贝数�?*，只是改变访问方�?
   - 我们�?Tensor 的所有算子都通过这个方法拿到 numpy 视图做计�?

**示例**�?

```python
# Storage: data = [1, 2, 3, 4, 5, 6], itemsize=8
# Tensor: shape=(2, 3), stride=(3, 1), offset=0

_numpy_view() 返回:
  as_strided(data[0:], shape=(2,3), strides=(24,8))

  [[1, 2, 3],
   [4, 5, 6]]

# 转置�? shape=(3, 2), stride=(1, 3), offset=0
_numpy_view() 返回:
  as_strided(data[0:], shape=(3,2), strides=(8,24))

  [[1, 4],
   [2, 5],
   [3, 6]]

# 同一�?buffer，不同的访问方式�?
```

### 3.4 view vs reshape vs contiguous

```python
    def view(self, *shape: int) -> Tensor:
        shape = _infer_shape(shape, self.size)
        if not self.is_contiguous():
            raise RuntimeError("view 要求 contiguous，请�?.reshape() 或先 .contiguous()")
        if _prod(shape) != self.size:
            raise RuntimeError(f"view: 元素数不匹配 {shape} vs {self.size}")
        return Tensor(
            self._storage, shape, _compute_contiguous_strides(shape),
            self._storage_offset, self.requires_grad
        )
```

**view** 的语义：

- 只改 shape，重新计�?contiguous stride
- **不拷贝数�?*——返回的 Tensor 共享同一 Storage
- 要求输入 contiguous——否则物理排列不匹配�?shape �?stride

```python
    def reshape(self, *shape: int) -> Tensor:
        shape = _infer_shape(shape, self.size)
        if _prod(shape) != self.size:
            raise RuntimeError("reshape: 元素数不匹配")
        if self.is_contiguous():
            return self.view(*shape)      # 连续时等�?view
        return self.contiguous().view(*shape)  # 非连续时先拷�?
```

**reshape** 的语义：

- 如果 contiguous，等�?view（零拷贝�?
- 如果非连续，�?`.contiguous()`（触发拷贝），再 view
- 用户不需要关心是否连续——reshape 总是能工�?

```python
    def contiguous(self) -> Tensor:
        if self.is_contiguous():
            return self  # 已经连续，无需操作
        arr = self._numpy_view().copy()  # materialize：拷贝到连续内存
        storage = Storage.from_numpy(arr)
        return Tensor(storage, arr.shape, _compute_contiguous_strides(arr.shape),
                      0, self.requires_grad)
```

**contiguous** 的语义：

- 如果已经连续，返�?self（零开销�?
- 否则，`_numpy_view().copy()` 把数据按当前 stride 读取并写入连续的�?buffer
- �?Tensor �?storage_offset=0，stride �?contiguous �?

**三者的关系**�?

| 操作 | 要求 contiguous | 拷贝 | 改变 shape |
|------|----------------|------|-----------|
| view | �?| �?| �?|
| reshape | �?| 仅非连续�?| �?|
| contiguous | �?| 仅非连续�?| �?|

### 3.5 transpose——零拷贝转置

```python
    def _transpose(self, dim0: int = 1, dim1: int = 0) -> Tensor:
        if dim0 < 0:
            dim0 += self.ndim
        if dim1 < 0:
            dim1 += self.ndim
        shape = list(self._shape)
        strides = list(self._strides)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
        return Tensor(self._storage, shape, strides,
                      self._storage_offset, self.requires_grad)
```

**逐行解释**�?

1. 处理负索引：`dim0 += self.ndim`（如 -1 �?ndim-1�?
2. 交换 shape 的两个维度：`shape[dim0], shape[dim1] = shape[dim1], shape[dim0]`
3. 交换 stride 的两个维�?
4. 创建�?Tensor�?*共享同一 Storage**

注意有两个版本：

- `_transpose`：纯计算，不建图（Ch1�?
- `transpose`：建图版本（Ch2），�?`Transpose.apply`

```python
    def transpose(self, dim0: int = 1, dim1: int = 0) -> Tensor:
        from .ops.arithmetic import Transpose
        return Transpose.apply(self, dim0=dim0, dim1=dim1)
```

### 3.6 广播——stride=0 技�?

广播规则：两�?shape 从右对齐，每个维度要么相同、要么其中之一�?1�?

```python
def _broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    ndim = max(len(s) for s in shapes)
    aligned = [(1,) * (ndim - len(s)) + s for s in shapes]  # 左填�?1
    result = []
    for dims in zip(*aligned, strict=True):
        non_one = [d for d in dims if d != 1]
        if not non_one:
            result.append(1)
        elif len(set(non_one)) == 1:
            result.append(non_one[0])  # 所有非 1 维度相同
        else:
            raise ValueError(f"shape 无法广播: {shapes}")
    return tuple(result)
```

广播的实现用 **stride=0** 技巧：

```python
    def broadcast_to(self, shape: Sequence[int]) -> Tensor:
        shape = tuple(shape)
        pad = len(shape) - self.ndim
        self_shape = (1,) * pad + self._shape
        self_strides = (0,) * pad + self._strides  # 新增维度 stride=0
        new_strides = []
        for i, (s, t) in enumerate(zip(self_shape, shape, strict=True)):
            if s == t:
                new_strides.append(self_strides[i])
            elif s == 1:
                new_strides.append(0)  # 广播维度 stride=0
            else:
                raise RuntimeError(f"broadcast_to: {self._shape} 不能广播�?{shape}")
        return Tensor(self._storage, shape, tuple(new_strides),
                      self._storage_offset, self.requires_grad)
```

**stride=0 的含�?*�?

```python
# a: shape=(3,), stride=(1,), data=[1, 2, 3]
# 广播�?shape=(2, 3)

# 广播�? shape=(2, 3), stride=(0, 1)
# a[0, 0] = buffer[0*0 + 0*1] = buffer[0] = 1
# a[0, 1] = buffer[0*0 + 1*1] = buffer[1] = 2
# a[1, 0] = buffer[1*0 + 0*1] = buffer[0] = 1  �?stride=0 让第 0 �?原地踏步"
# a[1, 1] = buffer[1*0 + 1*1] = buffer[1] = 2

# 结果：[[1, 2, 3], [1, 2, 3]]  �?零拷贝广播！
```

### 3.7 算术运算——底层方�?

```python
    def _binary(self, other, op) -> Tensor:
        other = self._ensure_tensor(other)
        a, b = Tensor.broadcast_tensors(self, other)  # 广播到同一 shape
        return Tensor.from_numpy(op(a._numpy_view(), b._numpy_view()))

    def _add(self, other) -> Tensor:
        return self._binary(other, lambda a, b: a + b)
```

**逐行解释**�?

1. `self._ensure_tensor(other)`：把标量转为 Tensor（如 `5` �?`Tensor([5])`�?
2. `Tensor.broadcast_tensors(self, other)`：广播两者到同一 shape
3. `op(a._numpy_view(), b._numpy_view())`：numpy 运算（如 `a + b`�?
4. `Tensor.from_numpy(...)`：把结果包回 Tensor

这些�?**底层方法**（下划线前缀），纯计算不建图。Ch2 的公开方法（`add`、`mul`）会�?`Function.apply` 建图，backward 中用底层方法避免递归建图�?

### 3.8 索引——__getitem__ �?__setitem__

```python
    def __getitem__(self, key) -> Tensor:
        arr = self._numpy_view()[key]
        return Tensor.from_numpy(np.asarray(arr))

    def __setitem__(self, key, value):
        arr = self._numpy_view()
        if isinstance(value, Tensor):
            value = value._numpy_view()
        arr[key] = value
```

`__getitem__` 目前返回�?Tensor（拷贝）。PyTorch 的索引返回视图（共享 Storage），但实现更复杂（需计算�?shape/stride/offset）�?

`__setitem__` 通过 numpy 视图直接写入底层 buffer——原地修改，影响所有共享同一 Storage �?Tensor�?

---

## 4. 完整示例：从创建到运�?

```python
import numpy as np
from minitorch import Tensor

# 创建张量
a = Tensor.from_numpy(np.arange(12).reshape(3, 4))
print(a)
# Tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], shape=(3, 4))
print(f"shape={a.shape}, strides={a.strides}, contiguous={a.is_contiguous()}")
# shape=(3, 4), strides=(4, 1), contiguous=True

# 转置（零拷贝�?
b = a.transpose()
print(f"shape={b.shape}, strides={b.strides}, contiguous={b.is_contiguous()}")
# shape=(4, 3), strides=(1, 4), contiguous=False

# 转置后需�?contiguous 才能�?view
c = b.contiguous()
print(f"contiguous={c.is_contiguous()}")
# contiguous=True

# 切片（零拷贝，改 storage_offset�?
d = a[1:]  # �?1 行开�?
print(d)
# Tensor([[4, 5, 6, 7], [8, 9, 10, 11]], shape=(2, 4))

# 广播运算
e = Tensor.from_numpy(np.ones((4,)))  # shape=(4,)
f = a + e  # (3,4) + (4,) �?广播�?(3,4)
print(f)
# Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], shape=(3, 4))

# 修改切片影响原张量（共享 Storage�?
d[0, 0] = 99
print(a[1, 0].item())  # 99 �?a 也变�?
```

---

## 5. 常见陷阱

### 5.1 view 对非连续张量报错

```python
a = Tensor.from_numpy(np.arange(6).reshape(2, 3))
b = a.transpose()  # 非连�?
b.view(6)  # RuntimeError: view 要求 contiguous
b.reshape(6)  # OK——reshape 自动 contiguous
```

**解决**：用 `reshape` 代替 `view`，或�?`.contiguous()`�?

### 5.2 原地修改影响共享 Tensor

```python
a = Tensor.from_numpy(np.arange(6).reshape(2, 3))
b = a.view(6)  # 共享 Storage
b[0] = 99
print(a[0, 0].item())  # 99 �?a 也被修改
```

**解决**：如果需要独立副本，�?`a.numpy()` 拷贝后重新创�?Tensor�?

### 5.3 广播的梯度方�?

广播 `(3,) + (2,3) �?(2,3)` 的反向，梯度需�?`(2,3)` sum �?`(3,)`。这�?Ch2 �?`_reduce_grad` 中处理。如果自己写 backward 忘了 reduce，梯�?shape 不匹配会报错�?

---

## 6. 与真�?PyTorch 对照

### 6.1 对应关系

| minitorch | PyTorch C++ | 说明 |
|-----------|-------------|------|
| `Storage` | `c10::Storage` | 持有 `c10::Allocator` 分配的内�?|
| `Storage._data` | `StorageImpl::data_ptr_` | 裸指针，�?numpy array |
| `Tensor._shape` | `TensorImpl::sizes_` | `c10::SmallVector<int64_t>` |
| `Tensor._strides` | `TensorImpl::strides_` | 同上 |
| `Tensor._storage_offset` | `TensorImpl::storage_offset_` | `int64_t` |
| `Tensor._storage` | `TensorImpl::storage_` | `c10::intrusive_ptr<StorageImpl>` |
| `_numpy_view()` | `Tensor::data_ptr()` | PyTorch 直接用裸指针 |
| `is_contiguous()` | `TensorImpl::is_contiguous()` | 检�?strides_ == expected |

### 6.2 引用计数

PyTorch �?`intrusive_ptr`（侵入式引用计数）管�?Storage 生命周期�?

```cpp
// c10/Storage.h
struct Storage {
    c10::intrusive_ptr<StorageImpl> storage_impl_;
};
```

多个 Tensor 共享同一 Storage 时，引用计数自动增减。当最后一个引用消失，Storage 析构释放内存�?

本项目用 Python �?GC——`self._storage` �?Python 引用，引用计数由 CPython 管理。功能等价，�?Python GC �?STW 暂停的开销�?

### 6.3 dtype 系统

PyTorch 有完整的 dtype 系统：float16/32/64, int8/16/32/64, bool, bfloat16, complex64/128�?

教学版只�?numpy �?dtype，通过 `Storage.from_numpy` 保留输入 dtype。Ch12 �?AMP 依赖此特性实�?float16�?

### 6.4 设备抽象

PyTorch �?Tensor �?`device` 属性（CPU/CUDA/MPS/...），决定数据在哪个设备上。算子通过 dispatch 系统路由到对应设备的 kernel�?

教学版只�?CPU（numpy 后端）。Ch10 会讨�?CUDA 的设计�?

---

## 7. 设计的历史背�?

### 7.1 NumPy 的先�?

NumPy �?ndarray 早就用了 stride 机制。`numpy.lib.stride_tricks.as_strided` �?PyTorch stride 设计的灵感来源。PyTorch 的创新在于：

- �?Storage 独立出来（NumPy �?ndarray 直接持有 data�?
- 加入 autograd 元信息（requires_grad / grad_fn�?
- 加入设备抽象（device / layout�?

### 7.2 为什么不直接�?NumPy�?

1. **autograd**：NumPy 没有自动微分。PyTorch 需要在 Tensor 上挂 grad_fn�?
2. **GPU**：NumPy 只支�?CPU。PyTorch 需要数据能�?GPU 上�?
3. **dispatch**：PyTorch 需要根�?dtype/device/layout 路由到不�?kernel�?
4. **内存管理**：PyTorch 用自定义 Allocator（如 CUDACachingAllocator），而非 malloc�?

### 7.3 TensorImpl 的演�?

早期 PyTorch�?.x）的 Tensor 直接持有 TH 指针（如 `THFloatTensor`），每种 dtype 一�?C 类型�?.0 重构�?`TensorImpl` + `Storage` + `TensorOptions`，统一�?dtype/device 的处理。这�?ATen（A Tensor Library）的诞生�?

---

## 8. 练习

### 练习 1：手动推�?stride

给定 shape = (2, 3, 4) �?contiguous 张量�?

1. 计算 stride
2. 元素 (1, 2, 3) 的物理偏移是多少�?
3. 转置 dim0=0, dim1=2 后，新的 shape �?stride 是什么？

??? 解答
    1. stride = (12, 4, 1)
    2. 1×12 + 2×4 + 3×1 = 23
    3. shape = (4, 3, 2), stride = (1, 4, 12)
???

### 练习 2：广播的 stride

`a` �?shape = (3,), stride = (1,)。广播到 (2, 3) 后：

1. 新的 stride 是什么？
2. `a_broadcasted[1, 2]` 对应 buffer 的哪个位置？

??? 解答
    1. stride = (0, 1)——新增维�?stride=0
    2. 1×0 + 2×1 = 2，即 buffer[2]
???

### 练习 3：实�?unsqueeze

`unsqueeze(dim)` �?dim 位置插入大小�?1 的维度。实现它（提示：新维度的 stride 可以�?0 或任意值，因为大小�?1 的维度只取索�?0）�?

??? 解答
    ```python
    def unsqueeze(self, dim):
        if dim < 0:
            dim += self.ndim + 1
        shape = self._shape[:dim] + (1,) + self._shape[dim:]
        strides = self._strides[:dim] + (0,) + self._strides[dim:]
        return Tensor(self._storage, shape, strides,
                      self._storage_offset, self.requires_grad)
    ```
    新维度的 stride 设为 0——因为大小为 1，索引只能是 0，`0 * stride = 0` 对任�?stride 都成立�?
???

### 练习 4：验�?view 共享内存

写一个测试：创建 Tensor `a`，`b = a.view(...)`，修�?`b` 的元素，验证 `a` 也变了�?

??? 解答
    ```python
    def test_view_shares_memory():
        a = Tensor.from_numpy(np.arange(6).reshape(2, 3))
        b = a.view(6)
        b[0] = 99
        assert a[0, 0].item() == 99
    ```
???

---

## 9. 关键测试解读

### 9.1 view 共享内存

```python
def test_view_shares_memory():
    a = Tensor.from_numpy(np.arange(6).reshape(2, 3))
    b = a.view(3, 2)
    a[0, 0] = 99
    assert b[0, 0].item() == 99
```

这个测试验证 view 确实零拷贝——修�?a 的元素，b 也可见，因为它们共享同一 Storage�?

### 9.2 transpose 后非连续

```python
def test_transpose_not_contiguous():
    a = Tensor.from_numpy(np.arange(6).reshape(2, 3))
    b = a.transpose()
    assert not b.is_contiguous()
    c = b.contiguous()
    assert c.is_contiguous()
```

验证 transpose 产生非连续张量，contiguous() 能修复�?

### 9.3 广播形状推导

```python
def test_broadcast_shapes():
    assert _broadcast_shapes((2, 3), (3,)) == (2, 3)
    assert _broadcast_shapes((4, 1), (1, 3)) == (4, 3)
    with pytest.raises(ValueError):
        _broadcast_shapes((2, 3), (4, 3))  # 2 �?4，无法广�?
```

---

## 10. 优劣势总结

### 优势

1. **view 零拷�?*：transpose/slice/reshape（连续时）只�?~24 字节元信息，不复制数据。这是处理大张量（GB 级）的基石�?
2. **统一抽象**：所有算子通过 `_numpy_view()` 拿到 numpy 视图做计算，无需关心 stride 细节�?
3. **广播零拷�?*：stride=0 技巧让广播不复制数据，只是重复读同一位置�?

### 代价

1. **非连续开销**：非连续张量�?view/序列化时需�?contiguous()（拷贝）。PyTorch 的每�?CUDA kernel 都有 contiguous �?non-contiguous 两个版本，增加开发成本�?
2. **共享语义陷阱**：原位修改影响所有共�?Storage �?Tensor。用户需理解 view 的共享语义�?
3. **stride 复杂�?*：算子需处理任意 stride（广播、转置、切片产生各�?stride 模式）�?

### 适用场景

任何需要高效多维数组操作的框架。NumPy、JAX、MLIR tensor、TensorFlow 都采用类似设计。stride 是多维数组领�?40 年的成熟智慧�?

---

## 11. 本章文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `src/minitorch/storage.py` | 61 | Storage �?|
| `src/minitorch/tensor.py` | 391 | Tensor 类（含底层算�?公开算子+backward�?|
| `tests/test_tensor.py` | ~200 | 19 个测�?|
| `docs/docs/ch01-tensor-storage.md` | 本文�?| 本章文档 |

---

## 12. 深入理解 as_strided

### 12.1 什么是 as_strided�?

`numpy.lib.stride_tricks.as_strided` �?numpy 最强大也最危险的函数之一。它创建一�?**虚拟视图**——不拷贝数据，只是改�?shape �?stride 来重新解释同一段内存�?

```python
import numpy as np
from numpy.lib.stride_tricks import as_strided

buf = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
print(buf.strides)  # (8,)——float64 �?8 字节

# �?as_strided 把一维数组解释为 2×3 矩阵
view = as_strided(buf, shape=(2, 3), strides=(24, 8))
print(view)
# [[1. 2. 3.]
#  [4. 5. 6.]]

# �?as_strided 做转置（零拷贝）
transposed = as_strided(buf, shape=(3, 2), strides=(8, 24))
print(transposed)
# [[1. 4.]
#  [2. 5.]
#  [3. 6.]]
```

### 12.2 为什么危险？

as_strided 不检�?stride 是否越界。你可以创建一�?读越�?的视图：

```python
# 危险！stride 太大，会读到 buffer 之外的内�?
evil = as_strided(buf, shape=(2, 3), strides=(100, 8))
# 这不会报错，但读到的值是垃圾数据�?segfault
```

PyTorch �?Tensor 不暴�?as_strided 给用户（�?`as_strided` 方法但需显式指定 stride）。我们的 `_numpy_view` 是内部方法，stride 由我们的逻辑保证正确�?

### 12.3 我们的用�?

```python
def _numpy_view(self):
    base = self._storage.data              # 一�?numpy array
    byte_strides = tuple(s * base.itemsize for s in self._strides)
    return as_strided(
        base[self._storage_offset:],       # �?offset 开�?
        shape=self._shape,
        strides=byte_strides,
    )
```

我们�?stride 始终�?`_compute_contiguous_strides`（连续）或维度交换（转置）或 stride=0（广播）产生，保证不越界�?

---

## 13. 内存布局可视�?

### 13.1 C-contiguous（行优先�?

```
shape=(2, 3), stride=(3, 1)

逻辑:     物理 buffer:
[1, 2, 3]   [1, 2, 3, 4, 5, 6]
[4, 5, 6]   �?       �?
            row 0   row 1
```

行优先：同一行的元素在内存中相邻�?

### 13.2 Fortran-contiguous（列优先�?

```
shape=(2, 3), stride=(1, 2)

逻辑:     物理 buffer:
[1, 3, 5]   [1, 2, 3, 4, 5, 6]
[2, 4, 6]   �? �? �?
            c0 c1 c2
```

列优先：同一列的元素在内存中相邻。PyTorch 不常用，�?MATLAB �?Fortran 默认此布局�?

### 13.3 转置后的布局

```
原始 a: shape=(2, 3), stride=(3, 1)
buffer = [1, 2, 3, 4, 5, 6]

a.T: shape=(3, 2), stride=(1, 3)
buffer = [1, 2, 3, 4, 5, 6]  �?同一 buffer�?

逻辑 a.T:
[1, 4]   �?a.T[0] = buffer[0], buffer[3]
[2, 5]   �?a.T[1] = buffer[1], buffer[4]
[3, 6]   �?a.T[2] = buffer[2], buffer[5]
```

### 13.4 广播后的布局

```
原始 a: shape=(3,), stride=(1,)
buffer = [1, 2, 3]

广播�?(2, 3): stride=(0, 1)
buffer = [1, 2, 3]  �?同一 buffer�?

逻辑:
[1, 2, 3]   �?row 0: buffer[0], buffer[1], buffer[2]
[1, 2, 3]   �?row 1: buffer[0+0], buffer[1+0], buffer[2+0]
              �?stride[0]=0 �?row 1 读同一位置
```

---

## 14. 性能考量

### 14.1 contiguous vs non-contiguous 的性能差异

```python
import time
import numpy as np

# 大张�?
a = np.random.randn(10000, 10000)

# contiguous 访问
t0 = time.time()
_ = a.sum()
t_contiguous = time.time() - t0

# non-contiguous 访问（转置）
b = a.T
t0 = time.time()
_ = b.sum()
t_non_contiguous = time.time() - t0

print(f"contiguous: {t_contiguous:.4f}s")
print(f"non-contiguous: {t_non_contiguous:.4f}s")
# non-contiguous 通常�?2-5x（cache miss�?
```

**原因**：contiguous 访问是顺序内存，CPU cache line 利用率高。non-contiguous 访问跳跃，cache miss 频繁�?

### 14.2 何时�?contiguous()�?

- 频繁访问同一张量（如训练循环中的权重�?
- 需要传给要求连续的 API（如 view、某�?CUDA kernel�?
- 序列化前

**不该** 在每次运算前�?contiguous()——如果只是做一次加法，非连续的 as_strided 开销可接受�?

---

## 15. 调试技�?

### 15.1 打印张量的完整信�?

```python
def debug_tensor(name, t):
    print(f"{name}:")
    print(f"  shape={t.shape}, strides={t.strides}")
    print(f"  offset={t.storage_offset}, contiguous={t.is_contiguous()}")
    print(f"  data={t.numpy()}")
    print(f"  storage_id={id(t.storage)}")

a = Tensor.from_numpy(np.arange(6).reshape(2, 3))
debug_tensor("a", a)
b = a.transpose()
debug_tensor("b=a.T", b)
```

### 15.2 检查两�?Tensor 是否共享 Storage

```python
def shares_storage(a, b):
    return a.storage is b.storage

a = Tensor.from_numpy(np.arange(6).reshape(2, 3))
b = a.view(6)
c = a.contiguous()
print(shares_storage(a, b))  # True
print(shares_storage(a, c))  # True（a 已连续，contiguous() 返回 self�?
```

### 15.3 追踪内存共享�?

当出现意外的原位修改 bug 时，检查哪�?Tensor 共享同一 Storage�?

```python
# �?__init__ 中加日志
class Storage:
    _counter = 0
    def __init__(self, ...):
        Storage._counter += 1
        self._id = Storage._counter
        print(f"Storage #{self._id} created")
```

---

## 16. 下一章预�?

本章实现�?Tensor/Storage �?**数据结构**�?**纯计算算�?*（`_add`、`_mul` 等）。但这些算子不追踪梯度——没�?`requires_grad`、没�?`grad_fn`、没�?`backward`�?

Ch2 将引�?**Function/Node/AccumulateGrad** 体系，让算子自动构建计算图。Ch3 实现反向传播引擎，从输出沿图回溯，用链式法则计算梯度。这是自动微分（autograd）的核心�?
