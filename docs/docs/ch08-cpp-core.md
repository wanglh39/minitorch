# 第八章 C++ 核心重写：Storage / TensorImpl / Autograd / pybind11 绑定

> 前七章的 minitorch 全用 Python 写，跑得对但跑得慢。本章我们把"热路径"
> ——Storage、TensorImpl、基本算子——用 C++ 重写，再用 pybind11 绑定回 Python。
> 重写后 Python 前端 API 一行不改，行为测试套件全过，但逐元素算子快了一个数量级。
> 这一章讲清"为什么要分三层""C++ 层怎么设计""绑定层怎么接线"，并逐行解读实现，
> 最后对照真实 PyTorch 的 `c10::Storage` / `at::TensorImpl` / `torch::autograd`，
> 让你理解 PyTorch 工程分层的来龙去脉。

---

## 8.1 本章目标

读完本章后，你应当能够：

1. 说出 PyTorch 的三层分离——**前端 API（Python）/ 核心计算（C++）/ 绑定层（pybind11）**——各自职责，以及为什么这么分。
2. 解释为什么用 C++ 重写核心而不是用 NumPy/Cython/Numba：C++ 能精确控制内存布局、引用计数、与 CUDA 的衔接，且是 PyTorch 的真实做法。
3. 手写一个 `Storage` C++ 类：用 `std::vector<double>` 持有数据、`shared_ptr` 管理生命周期、`operator[]` 带边界检查。
4. 手写一个 `TensorImpl` C++ 类：持有 `shared_ptr<Storage>` + shape + strides + storage_offset，实现 `view`/`transpose`/`contiguous` 的"零拷贝"。
5. 写出逐元素算子的通用模板 `binary_op`，并用 lambda 注入 `+`/`-`/`*`/`/`，复用同一套广播与遍历代码。
6. 用 pybind11 把 C++ 类和函数暴露给 Python，处理 `shared_ptr` 引用计数、`std::vector` 转换、NumPy 互操作。
7. 写出 CMakeLists.txt，用 `pybind11_add_module` 编译出 `.pyd`/`.so`，并装到 Python 包里能 `import`。
8. 排查三类常见编译/运行陷阱：DLL 依赖缺失、`std::div`/`std::sum` 名字冲突、Python 版本 ABI 不兼容。
9. 对照真实 PyTorch：`c10::Storage` vs 我们的 `Storage`、`at::TensorImpl` vs 我们的 `TensorImpl`、pybind11 vs CPython C-API。
10. 讲清 ATen 的诞生背景（2016 年 Torch7 的 C 后端 + Chainer 的 Python 前端思路融合）和 PyTorch 1.0 的"Variable/Tensor 合并"重构。

---

## 8.2 原理铺垫：PyTorch 的三层分离

### 8.2.1 为什么要分三层

先看一个朴素问题：**为什么不全用 Python 写？**

前七章的 minitorch 就是全 Python，能跑通 autograd、能训练 MLP。但它有两个硬伤：

1. **慢**：逐元素算子的 Python 解释器开销远大于浮点计算本身。一个 `[1000000]` 的加法，Python 循环要几十毫秒，C++ 只要零点几毫秒。
2. **接不上 GPU**：CUDA kernel 只能用 C++/CUDA C 调用，Python 不能直接 `__global__`。

那为什么不全用 C++ 写？因为 C++ 写前端（定义模型、写训练循环、调试）体验极差：没有 REPL、编译慢、类型繁琐、动态形状痛苦。

**结论**：前端要 Python 的灵活，核心要 C++ 的性能和异构能力。这就天然分成两层。再加一层"把 C++ 暴露给 Python"的绑定代码，就是三层：

```
┌─────────────────────────────────────────────┐
│  前端 API（Python）                          │   ← 用户写 model、训练循环
│  minitorch.Tensor / nn.Module / optim       │     灵活、动态、易调试
├─────────────────────────────────────────────┤
│  绑定层（pybind11 / CPython C-API）          │   ← 把 C++ 类/函数包成 Python 对象
│  cpp/binding/module.cpp                     │     处理引用计数、类型转换、异常
├─────────────────────────────────────────────┤
│  核心计算（C++）                              │   ← 真正干活的
│  Storage / TensorImpl / ops / autograd      │     快、能接 CUDA、控制内存
└─────────────────────────────────────────────┘
```

!!! tip "三层各自的一句话职责"
- **前端**：定义"用户怎么用"——API 长什么样、Module 怎么组织、autograd 怎么建图。
- **核心**：定义"实际怎么算"——内存怎么布局、算子怎么遍历、CUDA 怎么 launch。
- **绑定**：定义"两层怎么对话"——C++ 对象怎么变成 Python 句柄、异常怎么翻译、引用计数怎么对齐。

### 8.2.2 为什么用 C++ 而不是别的

能加速 Python 的方案很多，逐个对比：

| 方案 | 优点 | 缺点 | 适合 minitorch 吗 |
|------|------|------|-------------------|
| **NumPy** | 现成、快、广播成熟 | 不能自定义算子的内存布局；autograd 要在外面包；接 CUDA 要靠 cupy | 核心 Storage 其实就是 NumPy 的简化版，但我们要"自己造轮子"才能讲清原理 |
| **Cython** | 渐进加速、可调 Python | 语法是 Python 超集，生成 C 中间层；引用计数易错；与 CUDA 衔接弱 | 不适合讲"PyTorch 怎么做" |
| **Numba** | JIT、零样板 | 只加速数值循环；难表达复杂类层次；不能写 TensorImpl | 不适合 |
| **C 扩展（CPython C-API）** | 极致控制、PyTorch 早期用过 | 样板爆炸、引用计数全手写、易内存泄漏 | 能讲清历史，但教学代价高 |
| **pybind11 + C++** | 样板少、自动引用计数、与 C++ 生态无缝、能接 CUDA | 编译链跨平台有门槛 | **就是 PyTorch 现在的做法** |

minitorch 选 pybind11 + C++，不是因为它最快，而是因为它**和真实 PyTorch 同构**——学会 minitorch 的 C++ 层，去看 `c10/`、`aten/`、`torch/csrc/` 几乎没有概念鸿沟。

### 8.2.3 重写后什么不变、什么变

**不变**（这是关键设计原则）：

- Python 前端的 `Tensor` API：`t.shape`、`t + s`、`t.relu()`、`t.matmul(s)` 全部不变。
- 测试套件：前七章的 `tests/test_*.py` 一行不改，全过。这验证"行为等价"。
- autograd、nn.Module、optim：这些是 Python 层的"用户态"代码，不重写。

**变**：

- `Tensor` 内部从"持有 Python list/numpy array"改成"持有 C++ `TensorImpl` 句柄"。
- 算子的实际计算从 Python 循环改成 C++ 循环。
- 新增 `_C_ext` 扩展模块（`.pyd`/`.so`），`import minitorch._C_ext` 拿到 C++ 核心。

!!! warning "别误解"重写""
我们重写的是**核心计算**，不是整个 minitorch。autograd 引擎、计算图、nn.Module 这些"框架逻辑"仍在 Python——因为它们不在热路径上，且 Python 写起来更清晰。真实 PyTorch 的 autograd 也在 C++，但那是工程优化，不是教学必需。

---

## 8.3 设计决策与权衡

### 8.3.1 决策表

| 决策点 | 选项 | 我们选 | 理由 |
|--------|------|--------|------|
| 核心语言 | C / C++ / Rust / Zig | **C++17** | PyTorch 同款；CUDA 原生支持；pybind11 生态成熟 |
| 绑定方式 | CPython C-API / pybind11 / nanobind / ctypes | **pybind11** | 样板少、自动引用计数、PyTorch 同款；nanobind 更新但生态年轻 |
| 内存管理 | 裸指针 / `unique_ptr` / `shared_ptr` | **`shared_ptr`** | view 操作要共享 Storage，必须引用计数；`unique_ptr` 不够 |
| 数据容器 | 裸数组 / `std::vector` / 自定义 allocator | **`double*` + 全局 `Allocator`** | Ch8 基础用 `vector`；Ch9 高级特性引入 `Allocator` 接口 + `PoolAllocator` 内存池，与 `c10::Allocator` 同构 |
| dtype | 只 double / 模板 / 多 dtype | **只 double** | 教学版简化；真实 PyTorch 支持 float16/bfloat16/int8/... |
| 构建 | setuptools / CMake / scikit-build | **CMake + pybind11_add_module** | PyTorch 同款；跨平台；能加 CUDA |
| 异常 | C++ 异常 / 返回码 / abort | **C++ 异常 → pybind11 翻译成 Python 异常** | 自动、干净 |
| 算子分发 | 直接调 / dispatch table | **Ch8 直接调，Ch10 引入 dispatch** | 渐进教学：先讲清 C++，再讲异构路由 |

### 8.3.2 为什么用 `shared_ptr` 而不是 `unique_ptr`

`TensorImpl` 持有 `shared_ptr<Storage>`，不是 `unique_ptr<Storage>`。原因是 **view 操作要共享 Storage**：

```cpp
auto a = make_tensor({1,2,3,4}, {2,2});
auto b = a->transpose();   // b 和 a 共享同一块 Storage！
```

`transpose` 只改 shape/strides，不拷贝数据。`a` 和 `b` 都指向同一个 `Storage`，谁先析构都不能释放它，得等最后一个引用消失。这就是引用计数，`shared_ptr` 正好。

`unique_ptr` 是独占所有权，不能共享，所以不行。真实 PyTorch 用自己的 `c10::intrusive_ptr`（比 `shared_ptr` 省一次原子操作，因为线程模型不同），原理一样。

### 8.3.3 为什么算子返回新张量而不是原地修改

我们的算子签名是 `TensorImplPtr add(a, b)`——返回新张量，不改输入。原因：

1. **autograd 需要**：反向传播要用前向的输入，输入不能被覆盖。
2. **语义清晰**：`c = a + b` 不该偷偷改 `a`。
3. **PyTorch 同款**：`torch.add` 也是返回新张量；原地版本是 `torch.add_`（带下划线），另有一套。

代价：多一次分配。真实 PyTorch 有"out-of-place → in-place → fused"的优化链，教学版不展开。

---

## 8.4 代码逐行实现：Storage C++ 类

`Storage` 是最底层——一块一维的 `double` 缓冲区。对应阶段一的 `src/minitorch/storage.py`，对应真实 PyTorch 的 `c10::Storage`。

### 8.4.1 头文件 `cpp/c10/storage.h`

```cpp
// Storage：一维数据缓冲区（Ch8 C++ 核心重写）
//
// 使用全局 Allocator 分配内存，shared_ptr 自动引用计数。
// 对应真实 PyTorch 的 c10::Storage / c10::StorageImpl。

#pragma once

#include "allocator.h"   // Ch9 高级特性引入的自定义分配器
#include <cstddef>
#include <vector>
#include <memory>
#include <string>

namespace minitorch {

class Storage : public std::enable_shared_from_this<Storage> {
public:
    explicit Storage(size_t size);
    Storage(const std::vector<double>& data);
    Storage(std::vector<double>&& data);

    Storage(const Storage& other);
    Storage& operator=(const Storage& other);
    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;
    ~Storage();   // 非默认：要调 allocator->deallocate

    double* data() { return data_; }
    const double* data() const { return data_; }
    size_t size() const { return size_; }

    double& operator[](size_t idx);
    const double& operator[](size_t idx) const;

    std::string repr() const;

private:
    double* data_;    // 裸指针，由 Allocator 分配/释放
    size_t size_;
};
} // namespace minitorch
```

与旧版的区别：`std::vector<double> data_` 改为 `double* data_` + `size_t size_`，内存通过全局 `Allocator` 分配。这使得 `PoolAllocator` 可以复用已释放的内存块，减少 malloc/free 调用。详见 Ch9 §9.4。

逐点解释：

- `#pragma once`：标准防重复包含，比 `#ifndef` 守卫简洁。
- `namespace minitorch`：所有 C++ 代码放 `minitorch` 命名空间，避免和 `std`、PyTorch 真实符号冲突。
- `enable_shared_from_this`：让 `Storage` 能安全地 `shared_from_this()` 拿到自己的 `shared_ptr`。本教学版没强用到，但加上以备 Ch10 dispatcher。
- `explicit Storage(size_t)`：`explicit` 防止隐式转换（`Storage s = 5;` 会被禁，必须 `Storage s(5);`）。
- `std::vector<double>&& data`：移动构造，传右值时零拷贝。
- `data()` 返回裸指针：算子层要用连续指针做遍历，这是性能关键路径，不能每次返回 `vector` 拷贝。
- `operator[]` 带边界检查：教学版要安全；真实 PyTorch 在 release 模式去掉检查换性能。

### 8.4.2 实现 `cpp/c10/storage.cpp`

```cpp
#include "storage.h"
#include <sstream>
#include <stdexcept>
#include <cstring>   // memcpy

namespace minitorch {

// 构造：通过全局 Allocator 分配
Storage::Storage(size_t size) : data_(nullptr), size_(size) {
    if (size > 0) {
        data_ = get_global_allocator().allocate(size);
        // allocate 返回的内存已零初始化（new double[size]()）
    }
}

// 从 vector 拷贝：先 allocate 再 memcpy
Storage::Storage(const std::vector<double>& data) : data_(nullptr), size_(data.size()) {
    if (size_ > 0) {
        data_ = get_global_allocator().allocate(size_);
        std::memcpy(data_, data.data(), size_ * sizeof(double));
    }
}

// 深拷贝：allocate + memcpy
Storage::Storage(const Storage& other) : data_(nullptr), size_(other.size_) {
    if (size_ > 0) {
        data_ = get_global_allocator().allocate(size_);
        std::memcpy(data_, other.data_, size_ * sizeof(double));
    }
}

// 移动：偷走指针，置空源对象
Storage::Storage(Storage&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

// 析构：归还给 Allocator（可能进内存池而非真正 free）
Storage::~Storage() {
    if (data_) {
        get_global_allocator().deallocate(data_, size_);
    }
}

double& Storage::operator[](size_t idx) {
    if (idx >= size_) {
        throw std::out_of_range("Storage index out of range: " + std::to_string(idx));
    }
    return data_[idx];
}
// 越界抛 std::out_of_range，pybind11 翻译成 Python IndexError

std::string Storage::repr() const {
    std::ostringstream oss;
    oss << "Storage([";
    for (size_t i = 0; i < size_; ++i) {
        if (i > 0) oss << ", ";
        oss << data_[i];
    }
    oss << "])";
    return oss.str();
}
} // namespace minitorch
```

关键变化：所有构造/析构都走 `get_global_allocator()`。默认是 `DefaultAllocator`（直接 `new/delete` + 统计），可切换为 `PoolAllocator`（维护空闲块列表，deallocate 时不真正释放而是放入池中，下次 allocate 同尺寸时复用）。详见 Ch9 §9.4。

!!! tip "为什么 `data()` 返回裸指针而不是 `vector&`"
算子的内层循环要 `double* p = storage->data();` 然后裸指针遍历 `p[i]`。如果返回 `vector&`，每次访问多一次 `vector::operator[]`（虽然有内联，但语义上多一层）。真实 PyTorch 的 `c10::Storage::data()` 也返回裸指针。

---

## 8.5 代码逐行实现：TensorImpl C++ 类

`TensorImpl` 是张量的"实现核心"——持有 Storage + shape + strides + offset。对应阶段一的 `src/minitorch/tensor.py`，对应真实 PyTorch 的 `at::TensorImpl`。

### 8.5.1 头文件 `cpp/c10/tensor.h`（关键部分）

```cpp
#pragma once
#include "storage.h"
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

namespace minitorch {

class TensorImpl;
using TensorImplPtr = std::shared_ptr<TensorImpl>;

// 计算 contiguous strides（行优先）
std::vector<int64_t> compute_contiguous_strides(const std::vector<int64_t>& shape);

// 推导 shape（支持 -1，如 view(-1, 2)）
std::vector<int64_t> infer_shape(std::vector<int64_t> shape, int64_t numel);

// 广播 shapes
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a,
                                       const std::vector<int64_t>& b);

// shape 乘积
int64_t numel(const std::vector<int64_t>& shape);

class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    // 从 Storage 创建（最通用构造）
    TensorImpl(std::shared_ptr<Storage> storage,
               std::vector<int64_t> shape,
               std::vector<int64_t> strides,
               int64_t storage_offset = 0,
               bool requires_grad = false);

    // 从数据数组创建（便捷构造）
    TensorImpl(const std::vector<double>& data,
               std::vector<int64_t> shape,
               bool requires_grad = false);

    // 拷贝构造（浅拷贝，共享 Storage）
    TensorImpl(const TensorImpl& other);
    TensorImpl& operator=(const TensorImpl& other);

    ~TensorImpl() = default;

    // ── 属性 ──
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t storage_offset() const { return storage_offset_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return ::minitorch::numel(shape_); }
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool v) { requires_grad_ = v; }

    std::shared_ptr<Storage> storage() { return storage_; }

    // ── 形状操作 ──
    bool is_contiguous() const;
    TensorImplPtr contiguous() const;       // 物理化成连续
    TensorImplPtr view(std::vector<int64_t> shape) const;
    TensorImplPtr reshape(std::vector<int64_t> shape) const;
    TensorImplPtr transpose(int64_t dim0 = 1, int64_t dim1 = 0) const;
    TensorImplPtr permute(std::vector<int64_t> dims) const;

    // ── 数据访问 ──
    int64_t linear_offset(const std::vector<int64_t>& indices) const;
    double item() const;
    std::vector<double> to_vector() const;
    void from_vector(const std::vector<double>& data);
    void set_item(const std::vector<int64_t>& indices, double value);

    std::string repr() const;

private:
    std::shared_ptr<Storage> storage_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t storage_offset_;
    bool requires_grad_;
};
} // namespace minitorch
```

几个关键设计：

1. **`TensorImplPtr = shared_ptr<TensorImpl>`**：所有算子收发 `TensorImplPtr`，不是裸 `TensorImpl*`。引用计数自动管生命周期。
2. **`shape_`/`strides_` 是 `vector<int64_t>`**：用 `int64_t` 而非 `int`，因为大张量元素数会超 32 位。真实 PyTorch 同款。
3. **`storage_offset_`**：view/slice 可能从 Storage 中间开始读，这个偏移记录起点。真实 PyTorch 同款。
4. **拷贝构造是浅拷贝**：`TensorImpl(const TensorImpl& other)` 共享 Storage（`shared_ptr` 拷贝即共享）。深拷贝要显式调 `contiguous()`。

### 8.5.2 strides 计算

```cpp
std::vector<int64_t> compute_contiguous_strides(const std::vector<int64_t>& shape) {
    int64_t n = static_cast<int64_t>(shape.size());
    std::vector<int64_t> strides(n);
    if (n == 0) return strides;
    strides[n - 1] = 1;                    // 最右维 stride=1（行优先）
    for (int64_t i = n - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}
```

对 shape `[2, 3, 4]`：`strides = [12, 4, 1]`。意思是"沿第 0 维走一步跳 12 个元素，沿第 1 维跳 4 个，沿第 2 维跳 1 个"。这是行优先（C order）布局，和 NumPy/PyTorch 默认一致。

### 8.5.3 广播 shapes

```cpp
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a,
                                       const std::vector<int64_t>& b) {
    int64_t na = static_cast<int64_t>(a.size());
    int64_t nb = static_cast<int64_t>(b.size());
    int64_t n = std::max(na, nb);
    std::vector<int64_t> result(n);
    for (int64_t i = 0; i < n; ++i) {
        int64_t da = (i < na) ? a[na - 1 - i] : 1;   // 从右往左对齐
        int64_t db = (i < nb) ? b[nb - 1 - i] : 1;
        if (da == 1) result[n - 1 - i] = db;          // 1 广播成对方
        else if (db == 1) result[n - 1 - i] = da;
        else if (da == db) result[n - 1 - i] = da;    // 相等直接取
        else throw std::runtime_error("shape 无法广播");
    }
    return result;
}
```

广播规则：从右往左对齐维度，每维要么相等、要么其中一个是 1。`[2,3]` 和 `[3]` 广播成 `[2,3]`；`[2,3]` 和 `[2,1]` 广播成 `[2,3]`；`[2,3]` 和 `[2,4]` 报错。

### 8.5.4 `linear_offset`：逻辑索引 → 物理偏移

```cpp
int64_t TensorImpl::linear_offset(const std::vector<int64_t>& indices) const {
    int64_t offset = storage_offset_;
    for (size_t i = 0; i < indices.size() && i < strides_.size(); ++i) {
        offset += indices[i] * strides_[i];
    }
    return offset;
}
```

这是张量访问的"万能公式"：给定逻辑索引 `(i, j, k)`，物理偏移 = `offset + i*stride[0] + j*stride[1] + k*stride[2]`。view/transpose 只改 strides，这个公式自动适应，零拷贝。

### 8.5.5 `contiguous`：把非连续张量物理化

```cpp
TensorImplPtr TensorImpl::contiguous() const {
    if (is_contiguous()) {
        return std::make_shared<TensorImpl>(*this);   // 已经连续，浅拷贝
    }
    auto n = numel();
    std::vector<double> flat(static_cast<size_t>(n));
    std::vector<int64_t> indices(ndim(), 0);
    for (int64_t i = 0; i < n; ++i) {
        flat[static_cast<size_t>(i)] =
            storage_->data()[static_cast<size_t>(linear_offset(indices))];
        // 递增索引（行优先）
        for (int64_t d = ndim() - 1; d >= 0; --d) {
            if (++indices[d] < shape_[d]) break;
            indices[d] = 0;
        }
    }
    auto new_storage = make_storage(flat);
    return std::make_shared<TensorImpl>(new_storage, shape_,
                                        compute_contiguous_strides(shape_), 0, requires_grad_);
}
```

`transpose` 后张量非连续（strides 不是 `[12,4,1]` 那种标准形）。`contiguous()` 按逻辑顺序读一遍数据，写到新 Storage，得到连续张量。这是 `view` 要求连续的原因——`view` 不拷贝，只能在连续 buffer 上重解 shape。

!!! warning "索引递增的小技巧"
内层 `for (int64_t d = ndim()-1; d >= 0; --d) { if (++indices[d] < shape_[d]) break; indices[d] = 0; }` 是"进位加法"——从最右维加 1，满了进位到左维。这等价于 `np.ndindex`，但不用乘法，每步 O(ndim) 最坏，常数极小。

### 8.5.6 `transpose`：零拷贝

```cpp
TensorImplPtr TensorImpl::transpose(int64_t dim0, int64_t dim1) const {
    if (dim0 < 0) dim0 += ndim();    // 支持负索引
    if (dim1 < 0) dim1 += ndim();
    auto shape = shape_;
    auto strides = strides_;
    std::swap(shape[dim0], shape[dim1]);
    std::swap(strides[dim0], strides[dim1]);   // 只 swap strides！
    return std::make_shared<TensorImpl>(storage_, shape, strides,
                                        storage_offset_, requires_grad_);
}
```

`transpose` 不碰数据，只 swap 两个维度的 shape 和 strides。新张量共享旧 Storage。这就是"view 操作零拷贝"的精髓。

---

## 8.6 代码逐行实现：算子

算子在 `cpp/aten/ops.cpp`。核心是一个通用模板 `binary_op`，所有逐元素二元算子复用它。

### 8.6.1 广播到目标 shape

```cpp
TensorImplPtr broadcast_to(const TensorImplPtr& a,
                            const std::vector<int64_t>& target_shape) {
    const auto& src_shape = a->shape();
    int64_t src_ndim = static_cast<int64_t>(src_shape.size());
    int64_t dst_ndim = static_cast<int64_t>(target_shape.size());

    if (dst_ndim < src_ndim) {
        throw std::runtime_error("broadcast_to: 目标维度不能少于自身");
    }

    // 左填充：把 src 维度右对齐到 dst
    int64_t pad = dst_ndim - src_ndim;
    std::vector<int64_t> padded_shape(dst_ndim, 1);
    std::vector<int64_t> padded_strides(dst_ndim, 0);
    for (int64_t i = 0; i < src_ndim; ++i) {
        padded_shape[pad + i] = src_shape[i];
        padded_strides[pad + i] = a->strides()[i];
    }

    // 计算新 stride：广播维度 stride=0（读同一元素）
    std::vector<int64_t> new_strides(dst_ndim);
    for (int64_t i = 0; i < dst_ndim; ++i) {
        if (padded_shape[i] == target_shape[i]) {
            new_strides[i] = padded_strides[i];
        } else if (padded_shape[i] == 1) {
            new_strides[i] = 0;   // 关键：stride=0 让遍历时反复读同一位置
        } else {
            throw std::runtime_error("broadcast_to: shape 不兼容");
        }
    }

    return std::make_shared<TensorImpl>(a->storage(), target_shape, new_strides,
                                        a->storage_offset(), a->requires_grad());
}
```

**广播的零拷贝技巧**：把 `[3]` 广播成 `[2,3]`，不拷贝数据，只把新张量的 strides 设成 `[0, 1]`——第 0 维 stride=0 意味着"沿第 0 维走时地址不变"，于是两个"行"实际读同一块数据。这是 PyTorch 广播的核心实现，和我们这里完全一致。

### 8.6.2 通用二元算子模板

```cpp
template <typename Op>
TensorImplPtr binary_op(const TensorImplPtr& a, const TensorImplPtr& b, Op op) {
    auto [ba, bb] = broadcast_tensors(a, b);   // 先广播到同一 shape
    const auto& shape = ba->shape();
    int64_t n = ba->numel();

    std::vector<double> result(static_cast<size_t>(n));
    std::vector<int64_t> indices(ba->ndim(), 0);

    for (int64_t i = 0; i < n; ++i) {
        int64_t off_a = ba->linear_offset(indices);
        int64_t off_b = bb->linear_offset(indices);
        result[static_cast<size_t>(i)] = op(
            ba->storage()->data()[static_cast<size_t>(off_a)],
            bb->storage()->data()[static_cast<size_t>(off_b)]
        );
        // 递增索引
        for (int64_t d = ba->ndim() - 1; d >= 0; --d) {
            if (++indices[d] < shape[d]) break;
            indices[d] = 0;
        }
    }

    return make_tensor(result, shape);
}
```

`Op` 是模板参数——一个可调用对象（lambda 或函数指针）。`op` 接收两个 `double` 返回一个 `double`。这样 `add`/`sub`/`mul`/`div` 只差一个 lambda：

```cpp
TensorImplPtr add(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x + y; });
}
TensorImplPtr sub(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x - y; });
}
TensorImplPtr mul(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x * y; });
}
TensorImplPtr div(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x / y; });
}
```

!!! tip "模板 vs 虚函数"
这里用编译期模板（`template <typename Op>`），`op` 调用在编译期就内联进循环。如果用虚函数（`std::function` 或继承），每次循环多一次间接调用。教学版用模板，性能最佳；真实 PyTorch 用代码生成批量产生特化代码，思路相同。

### 8.6.3 一元算子（neg / relu）

```cpp
TensorImplPtr relu(const TensorImplPtr& a) {
    int64_t n = a->numel();
    std::vector<double> result(static_cast<size_t>(n));
    std::vector<int64_t> indices(a->ndim(), 0);
    for (int64_t i = 0; i < n; ++i) {
        double val = a->storage()->data()[static_cast<size_t>(a->linear_offset(indices))];
        result[static_cast<size_t>(i)] = std::max(0.0, val);
        for (int64_t d = a->ndim() - 1; d >= 0; --d) {
            if (++indices[d] < a->shape()[d]) break;
            indices[d] = 0;
        }
    }
    return make_tensor(result, a->shape());
}
```

`relu` 在 0 处取 `max(0.0, val)`，即 `x > 0` 时返回 x，否则 0。导数在 0 处取 0（与 PyTorch 一致）。

### 8.6.4 矩阵乘法

```cpp
TensorImplPtr matmul(const TensorImplPtr& a, const TensorImplPtr& b) {
    const auto& sa = a->shape();
    const auto& sb = b->shape();
    if (sa.size() != 2 || sb.size() != 2) {
        throw std::runtime_error("matmul: 目前只支持 2D");
    }
    if (sa[1] != sb[0]) {
        throw std::runtime_error("matmul: shape 不兼容");
    }
    int64_t M = sa[0], K = sa[1], N = sb[1];
    std::vector<double> result(static_cast<size_t>(M * N), 0.0);

    auto a_data = a->to_vector();   // 物理化成连续，简化索引
    auto b_data = b->to_vector();

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            double s = 0;
            for (int64_t k = 0; k < K; ++k) {
                s += a_data[static_cast<size_t>(i * K + k)] *
                     b_data[static_cast<size_t>(k * N + j)];
            }
            result[static_cast<size_t>(i * N + j)] = s;
        }
    }
    return make_tensor(result, {M, N});
}
```

三重循环，O(M·K·N)。教学版朴素实现；真实 PyTorch 调 BLAS（cuBLAS/MKL），分块、向量化、多线程。但**语义和这里完全一致**——这就是 matmul 的定义。

---

## 8.7 代码逐行实现：pybind11 绑定

绑定层在 `cpp/binding/module.cpp`。它把 C++ 的 `Storage`、`TensorImpl`、算子暴露成 Python 的类和函数。

### 8.7.1 模块入口

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // 让 pybind11 认识 std::vector
#include <pybind11/numpy.h>     // NumPy 互操作

#include "c10/storage.h"
#include "c10/tensor.h"
#include "aten/ops.h"

namespace py = pybind11;
using namespace minitorch;
using namespace minitorch::ops;

PYBIND11_MODULE(_C_ext, m) {
    m.doc() = "minitorch C++ core (Ch8)";
    // ... 在这里绑定类和函数
}
```

`PYBIND11_MODULE(_C_ext, m)` 是宏，展开后等价于 CPython 的 `PyInit__C_ext`。`m` 是模块对象，往上面挂类和函数。

### 8.7.2 绑定 Storage

```cpp
py::class_<Storage, std::shared_ptr<Storage>>(m, "Storage")
    .def(py::init<size_t>(), py::arg("size"))
    .def(py::init<const std::vector<double>&>(), py::arg("data"))
    .def("__len__", &Storage::size)
    .def("__getitem__", [](const Storage& s, size_t i) { return s[i]; })
    .def("__repr__", &Storage::repr)
    .def("data", [](Storage& s) {
        return py::array_t<double>(
            static_cast<py::ssize_t>(s.size()),
            s.data(),
            py::cast(&s)   // 持有引用，防止 Storage 被回收
        );
    });
```

- `py::class_<Storage, std::shared_ptr<Storage>>`：第二个模板参数告诉 pybind11 用 `shared_ptr` 持有——这样 Python 端的 `Storage` 对象引用计数和 C++ 的 `shared_ptr` 对齐，不会提前回收。
- `.def(py::init<size_t>(), ...)`：暴露构造函数，`py::arg` 给参数名，Python 端能 `Storage(size=10)`。
- `__len__`/`__getitem__`/`__repr__`：Python 魔术方法，pybind11 直接映射。
- `data()` 返回 `py::array_t<double>`：零拷贝把 C++ 的 `double*` 包成 NumPy 数组。`py::cast(&s)` 让数组持有对 `Storage` 的引用，数组存活期间 Storage 不会被回收——这是避免"悬垂指针"的关键。

### 8.7.3 绑定 TensorImpl

```cpp
py::class_<TensorImpl, TensorImplPtr>(m, "TensorImpl")
    .def(py::init<const std::vector<double>&, std::vector<int64_t>, bool>(),
         py::arg("data"), py::arg("shape"), py::arg("requires_grad") = false)
    .def_property_readonly("shape", [](const TensorImpl& t) { return t.shape(); })
    .def_property_readonly("strides", [](const TensorImpl& t) { return t.strides(); })
    .def_property_readonly("ndim", &TensorImpl::ndim)
    .def_property_readonly("numel", &TensorImpl::numel)
    .def("is_contiguous", &TensorImpl::is_contiguous)
    .def("contiguous", &TensorImpl::contiguous)
    .def("view", [](const TensorImplPtr& t, std::vector<int64_t> shape) {
        return t->view(std::move(shape));
    })
    .def("transpose", &TensorImpl::transpose,
         py::arg("dim0") = 1, py::arg("dim1") = 0)
    .def("item", &TensorImpl::item)
    .def("to_vector", &TensorImpl::to_vector)
    .def("numpy", [](const TensorImplPtr& t) {
        auto data = t->to_vector();
        auto shape = t->shape();
        std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
        py::array_t<double> arr(py_shape);
        std::copy(data.begin(), data.end(), arr.mutable_data());
        return arr;
    })
    .def_static("zeros", [](std::vector<int64_t> shape) {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return make_tensor(std::vector<double>(static_cast<size_t>(n), 0.0), shape);
    })
    .def_static("from_numpy", [](py::array_t<double> arr) {
        py::buffer_info buf = arr.request();
        std::vector<int64_t> shape;
        for (auto s : buf.shape) shape.push_back(static_cast<int64_t>(s));
        std::vector<double> data(
            static_cast<double*>(buf.ptr),
            static_cast<double*>(buf.ptr) + buf.size
        );
        return make_tensor(data, shape);
    });
```

几个要点：

- `def_property_readonly`：只读属性，Python 端 `t.shape`（不是 `t.shape()`）。
- `def_static`：静态方法，`TensorImpl.zeros([2,3])`。
- `from_numpy`：通过 `py::buffer_info` 拿 NumPy 数组的裸指针和 shape，零拷贝构造。`py::array_t<double>` 是 pybind11 的类型化数组包装。
- lambda 包装：很多方法用 lambda 而非直接传函数指针，因为要 `std::move` 参数或做类型转换。

### 8.7.4 绑定算子（避开名字冲突）

```cpp
// 用 lambda 包装避免与 std::div/std::sum 等名字冲突
m.def("add", [](const TensorImplPtr& a, const TensorImplPtr& b) { return add(a, b); });
m.def("sub", [](const TensorImplPtr& a, const TensorImplPtr& b) { return sub(a, b); });
m.def("mul", [](const TensorImplPtr& a, const TensorImplPtr& b) { return mul(a, b); });
m.def("div", [](const TensorImplPtr& a, const TensorImplPtr& b) { return div(a, b); });
m.def("neg", [](const TensorImplPtr& a) { return neg(a); });
m.def("relu", [](const TensorImplPtr& a) { return relu(a); });
m.def("sum", [](const TensorImplPtr& a, int64_t dim, bool keepdim) { return sum(a, dim, keepdim); },
      py::arg("a"), py::arg("dim") = -1, py::arg("keepdim") = false);
m.def("matmul", [](const TensorImplPtr& a, const TensorImplPtr& b) { return matmul(a, b); });
```

!!! warning "为什么用 lambda 包一层"
`m.def("sum", &sum)` 看似更简洁，但 `sum` 会和 `std::sum`（`<numeric>` 里）歧义，`div` 会和 `std::div`（`<cstdlib>` 里）歧义。用 lambda 显式指明 `minitorch::ops::sum`，既避开冲突，又能加 `py::arg` 默认值。这是 C++ 算子绑定的常见痛点。

---

## 8.8 C++ Autograd 及高级特性

本章讲解了 C++ 核心的**基础层**：Storage、TensorImpl、ops、pybind11 绑定。

C++ 核心的**高级特性**——autograd 引擎、多线程并行、double backward、自定义 Allocator、Profiler、Hooks、Checkpointing——内容量大且涉及多个架构级设计决策，已独立成 **[Ch9 C++ 高级特性](ch09-cpp-advanced.md)**，涵盖：

- §9.2 C++ Autograd：Node 虚函数 / Engine 拓扑排序 / 建图算子
- §9.3 多线程 Engine：ThreadPool + 原子依赖计数 + 并行调度
- §9.4 double backward：`create_graph` + MulNode 用 autograd::mul 建新图
- §9.5 自定义 Allocator：`Allocator` 接口 + `PoolAllocator` 内存池
- §9.6 Autograd Profiler：记录每个 Node 耗时和内存分配
- §9.7 梯度钩子：`register_hook` 修改/监控梯度
- §9.8 Anomaly Detection：检测 NaN/Inf 梯度并抛异常
- §9.9 Gradient Checkpointing：用重计算换内存

这些特性对应真实 PyTorch 的 `torch/csrc/autograd/engine.cpp`（多线程引擎）、1.0 的 `create_graph` 机制、`c10::Allocator` 子系统、`torch.autograd.profiler`、`torch.utils.checkpoint` 等。

---

## 8.9 完整示例：从 CMake 到 import

### 8.9.1 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(minitorch_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)   # 用纯标准 C++，不依赖编译器扩展

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)   # 默认 Release，开优化
endif()

# ── pybind11 ──
find_package(pybind11 CONFIG REQUIRED)

# ── 源文件 ──
# 三层分离：c10（核心抽象）/ aten（算子）/ autograd（自动微分）
set(MINITORCH_SOURCES
    c10/allocator.cpp
    c10/storage.cpp
    c10/tensor.cpp
    aten/ops.cpp
    autograd/node.cpp
    autograd/engine.cpp
    autograd/ops.cpp
    binding/module.cpp
)

# ── 编译扩展模块 ──
pybind11_add_module(_C_ext ${MINITORCH_SOURCES})

# ── 包含目录 ──
# 以 cpp/ 为根，#include 使用 "c10/xxx.h" / "aten/xxx.h" / "autograd/xxx.h"
target_include_directories(_C_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ── 安装到 Python 包里 ──
install(TARGETS _C_ext
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/../src/minitorch
)
```

`pybind11_add_module(_C_ext ...)` 是 pybind11 提供的便利函数，等价于：

```cmake
add_library(_C_ext MODULE ${SOURCES})    # MODULE = 共享库，Python 可 import
target_link_libraries(_C_ext PRIVATE pybind11::module)
set_target_properties(_C_ext PROPERTIES PREFIX "" SUFFIX ".pyd"/".so")
```

它自动处理：链接 pybind11、设对后缀、加 Python 头文件路径。

### 8.9.2 编译命令（Windows / PowerShell）

```powershell
# 1. 安装 pybind11
pip install pybind11

# 2. 配置（让 CMake 找到 pybind11）
cd C:\Users\wlh19\Desktop\pytorch\cpp
cmake -B build -S . -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

# 3. 编译
cmake --build build --config Release

# 4. 产物 _C_ext.cp314-win_amd64.pyd 会被装到 src/minitorch/
```

### 8.9.3 Python 端 import

```python
# src/minitorch/__init__.py 里
try:
    from . import _C_ext as _cpp_ext
    _has_cpp = True
except ImportError:
    _has_cpp = False

# src/minitorch/tensor.py 里
class Tensor:
    def __init__(self, data, shape, requires_grad=False):
        if _has_cpp:
            self._impl = _cpp_ext.TensorImpl(data, shape, requires_grad)
        else:
            self._impl = PythonTensorImpl(data, shape)   # 阶段一的纯 Python 实现
```

前端 `Tensor` 自动用 C++ 核心或 Python 核心取决于是否编译了扩展。**用户代码不变**。

### 8.9.4 验证

```python
from minitorch import _cpp_ext
a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
b = _cpp_ext.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2])
c = _cpp_ext.add(a, b)
print(c.to_vector())   # [6.0, 8.0, 10.0, 12.0]
print(c.numpy())       # [[ 6.  8.]
                       #  [10. 12.]]
```

---

## 8.10 常见陷阱

### 8.10.1 DLL 依赖缺失

**症状**：`import minitorch._C_ext` 报 `ImportError: DLL load failed`。

**原因**：编译时链接了某个 DLL（如 `vcruntime140.dll`、`libstdc++-6.dll`），运行时找不到。

**排查**（Windows）：

```powershell
# 用 dumpbin 看依赖
dumpbin /dependents _C_ext.cp314-win_amd64.pyd
```

**解决**：

- 装上 Visual C++ Redistributable。
- 或静态链接 runtime：CMake 里 `set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")`。
- Linux 用 `ldd` 查依赖，`LD_LIBRARY_PATH` 加路径。

### 8.10.2 名字冲突：std::div / std::sum

**症状**：编译报 `reference to 'div' is ambiguous` 或 `call of overloaded 'sum(...)' is ambiguous`。

**原因**：`using namespace minitorch::ops;` 后，`div` 同时指 `minitorch::ops::div` 和 `std::div`（来自 `<cstdlib>`），编译器无法消歧。

**解决**：

- 用 lambda 显式包装（我们就是这么做的）。
- 或不用 `using namespace`，全限定 `minitorch::ops::div`。
- 或 `#undef div` / `#undef sum`（Windows 的 `<cstdlib>` 会把 `div` 定义成宏，最脏但有效）。

### 8.10.3 Python 版本 ABI 不兼容

**症状**：Python 3.13 编译的 `.pyd` 在 Python 3.14 import 报 `ImportError: Module use of python313.dll conflicts with this version of Python`。

**原因**：CPython 的 C-API 不保证跨版本 ABI 兼容（有限稳定 ABI 除外）。`.pyd` 绑死了编译时的 Python 版本。

**解决**：

- 每个要支持的 Python 版本各编译一次（CI 矩阵）。
- 文件名带版本标签：`_C_ext.cp314-win_amd64.pyd`，Python 只 import 匹配自己版本的。
- 或用 pybind11 的 `PYBIND11_MODULE(..., m, py::mod_gil_not_used())` 配合有限稳定 ABI（进阶）。

### 8.10.4 shared_ptr 循环引用

**症状**：内存泄漏，TensorImpl 永不析构。

**原因**：`TensorImpl` 继承 `enable_shared_from_this`，若内部又持 `shared_ptr` 回指自己或形成环，引用计数清不掉。

**解决**：用 `weak_ptr` 打破环。本教学版没这个问题（TensorImpl 不持回指），但 autograd 的计算图会遇——`Node` 的 `next_edges_` 用 `weak_ptr` 避免环。

### 8.10.5 异常穿越语言边界

**症状**：C++ 抛异常没被 catch，进程崩溃。

**原因**：C++ 异常不能直接抛进 Python 栈，必须经 pybind11 翻译。

**解决**：pybind11 自动把 `std::exception` 翻译成 Python 异常。`std::runtime_error` → `RuntimeError`，`std::out_of_range` → `IndexError`。**不要**在绑定层用 `try/catch` 吞掉异常——让 pybind11 翻译。

---

## 8.11 与真实 PyTorch 对照

### 8.11.1 Storage

| 我们的 `minitorch::Storage` | 真实 `c10::Storage` |
|----------------------------|---------------------|
| `double* data_` + `size_t size_` + `Allocator*` | `c10::DataPtr` + `c10::Allocator*` |
| 只支持 double | 支持 float16/bfloat16/int8/... |
| 只 host 内存（但可换 Allocator） | 支持 CUDA/pinned/managed 内存（靠 Allocator） |
| `shared_ptr` 引用计数 | `c10::intrusive_ptr`（省原子操作） |
| 无 resizable | 有 `set_nbytes`/`resize` |

我们的 Storage 是 c10::Storage 的"单 dtype、单 device"特化。**核心结构（一块 buffer + 大小 + allocator 指针 + 引用计数）完全一致**。自定义 Allocator 的深入讲解见第九章 §9.4。

### 8.11.2 TensorImpl

| 我们的 `minitorch::TensorImpl` | 真实 `at::TensorImpl` / `c10::TensorImpl` |
|--------------------------------|-------------------------------------------|
| `shared_ptr<Storage> storage_` | `c10::Storage storage_` |
| `vector<int64_t> shape_` | `c10::SymDimVector sizes_`（符号形状支持） |
| `vector<int64_t> strides_` | `c10::DimVector strides_` |
| `int64_t storage_offset_` | `int64_t storage_offset_` |
| `bool requires_grad_` | `bool requires_grad_`（实际由 autograd MetaData 管） |
| 无 dispatch key | `DispatchKeySet key_set_`（路由用） |
| 无 naming/tracing | 支持 `set_debug_name`、FX tracing 钩子 |

真实 TensorImpl 多了"dispatch key set"（Ch10 讲）和一堆元数据（命名、tracing、autograd 包裹），但**张量的本质——Storage + shape + strides + offset——和这里一字不差**。

### 8.11.3 绑定层

| 我们的 pybind11 绑定 | 真实 PyTorch 绑定 |
|----------------------|-------------------|
| `pybind11_add_module` | 手写 CPython C-API + 大量样板 |
| `py::class_<TensorImpl, shared_ptr<TensorImpl>>` | `THPVariable_Wrap` 手动管引用 |
| 自动异常翻译 | 手写 `catch (...)` → `PyErr_SetString` |
| ~125 行绑定代码 | `torch/csrc/autograd/python_autograd.cpp` 等数千行 |

PyTorch 早期（0.x）全用 CPython C-API，样板爆炸。后来部分迁到 pybind11（`torch::autograd` 的 C++ 前端），但核心 `torch/csrc/` 仍是 C-API（历史包袱 + 极致性能）。我们用 pybind11 是"如果今天重写会怎么做"的选择。

!!! tip "为什么 PyTorch 不全换 pybind11"
CPython C-API 比 pybind11 省一层间接调用，热路径上每个张量操作都过绑定层，省一点很重要。且 PyTorch 的绑定有大量定制（自定义 `__torch_function__`、autograd hooks、dispatcher 钩子），C-API 更灵活。这是"工程现实"和"教学清晰"的取舍。

---

## 8.12 历史背景

### 8.12.1 ATen 的诞生（2016）

PyTorch 的 C++ 核心叫 **ATen**（"A Tensor Library"），2016 年由 Soumith Chintala 等人创建。背景：

- **Torch7**（Lua 前端 + C 后端）性能强但 Lua 生态小，用户少。
- **Chainer**（Python 前端 + NumPy 后端）易用但慢，且 autograd 在 Python。
- **TensorFlow** 静态图难调试。

Soumith 的思路：**Torch7 的 C 后端 + Chainer 的 Python 前端 + 动态图**。C 后端就是 ATen——把 Torch7 的 `libtorch` 的张量算子抽出来，做成"纯张量库"，不带 autograd。autograd 在 Python 层（后来的 `torch::autograd` C++ 化是后迁的）。

ATen 的关键设计是 **"operator dispatch by device"**——同一个 `at::add` 在 CPU 走 `at::native::add_cpu`，在 CUDA 走 `at::native::add_cuda`。这就是 Ch10 要讲的 dispatcher。

### 8.12.2 PyTorch 1.0 重构（2018）

PyTorch 0.x 有两个张量类型：`Tensor`（纯张量）和 `Variable`（autograd 包裹）。用户要写 `Variable(t, requires_grad=True)`，烦。

2018 年 PyTorch 1.0 把两者合并：`Variable` 废弃，`Tensor` 直接带 `requires_grad`。内部实现是 `Tensor` 变成 `TensorImpl` 的句柄，`TensorImpl` 持有可选的 autograd 元数据。**这就是我们 `TensorImpl` 有 `requires_grad_` 字段的由来**——它在 C++ 层统一了"纯张量"和"autograd 张量"。

同时 `torch::autograd` 迁到 C++，`Function` 基类、`Node`、`Edge` 都在 C++，Python 端只是薄壳。minitorch 出于教学把 autograd 留在 Python，但**结构对照真实 PyTorch 的 C++ autograd**。

### 8.12.3 从 ATen 到 "core" 重命名（2020+）

2020 年后 PyTorch 把 `aten/` 和 `c10/`（C++ 核心库）整理成 `core/`，强调"c10 是无依赖的纯 C++ 库，aten 是算子层"。minitorch 的目录结构（`cpp/c10/` 放核心抽象，`cpp/aten/` 放算子，`cpp/autograd/` 放自动微分）就是这套分层的简化。

---

## 8.13 练习题

### 练习 1：给 Storage 加 `resize` 方法

要求：`void Storage::resize(size_t new_size, double fill = 0.0)`，把 buffer 改成 `new_size`，新增位置填 `fill`，已有位置保留。

??? 解答 ???

```cpp
// c10/storage.h 里加声明
void resize(size_t new_size, double fill = 0.0);

// c10/storage.cpp 里加实现
void Storage::resize(size_t new_size, double fill) {
    if (new_size > data_.size()) {
        data_.resize(new_size, fill);   // resize 会用 fill 填新增位置
    } else {
        data_.resize(new_size);          // 缩小，旧值丢弃
    }
}
```

注意 `std::vector::resize(n, val)` 只在扩大时用 `val` 填新位置；缩小时不碰保留部分。这正是我们要的语义。

### 练习 2：实现 `exp` 算子

要求：`TensorImplPtr exp(const TensorImplPtr& a)`，逐元素 `e^x`。用一元算子的模式，包含 `<cmath>` 的 `std::exp`。

??? 解答 ???

```cpp
// aten/ops.h 加声明
TensorImplPtr exp(const TensorImplPtr& a);

// aten/ops.cpp 加实现
TensorImplPtr exp(const TensorImplPtr& a) {
    int64_t n = a->numel();
    std::vector<double> result(static_cast<size_t>(n));
    std::vector<int64_t> indices(a->ndim(), 0);
    for (int64_t i = 0; i < n; ++i) {
        double val = a->storage()->data()[static_cast<size_t>(a->linear_offset(indices))];
        result[static_cast<size_t>(i)] = std::exp(val);
        for (int64_t d = a->ndim() - 1; d >= 0; --d) {
            if (++indices[d] < a->shape()[d]) break;
            indices[d] = 0;
        }
    }
    return make_tensor(result, a->shape());
}

// binding/module.cpp 里绑定
m.def("exp", [](const TensorImplPtr& a) { return exp(a); });
```

### 练习 3：解释为什么 `transpose` 后 `is_contiguous()` 返回 false

要求：对 `[2,3]` 的张量 `a`，`b = a.transpose()` 后 `b.is_contiguous()` 为什么是 false？写出 `b` 的 shape 和 strides。

??? 解答 ???

`a` 的 shape `[2,3]`，strides `[3,1]`（contiguous）。
`transpose()` swap 第 0、1 维：shape 变 `[3,2]`，strides 变 `[1,3]`。
`compute_contiguous_strides([3,2])` = `[2,1]`，而 `b` 的 strides 是 `[1,3]`，不等，所以 `is_contiguous()` 返回 false。

物理含义：`b` 沿第 0 维走一步跳 1 个元素（原 `a` 的行内），沿第 1 维走一步跳 3 个元素（原 `a` 的行间）——这是列优先访问行优先存储，不连续。

### 练习 4：`view` 为什么要求 contiguous

要求：解释 `view` 的实现里 `if (!is_contiguous()) throw` 的原因。给一个会触发该异常的例子。

??? 解答 ???

`view` 只改 shape 和 strides，不拷贝数据。它假设数据是按"新 shape 的行优先"连续排列的。如果原张量非连续（如 `transpose` 后），数据物理排列不对应任何简单的行优先 shape，强行 `view` 会读到错位数据。

会触发的例子：
```python
a = _cpp_ext.TensorImpl([1,2,3,4,5,6], [2,3])
b = a.transpose()          # shape [3,2], 非连续
c = b.view([6])            # 抛异常！
```

正确做法：`c = b.contiguous().view([6])` 或直接 `c = b.reshape([6])`（`reshape` 会自动 `contiguous`）。

### 练习 5：pybind11 的 `py::cast(&s)` 是干嘛的

要求：解释绑定 `Storage::data` 时 `py::array_t<double>(size, ptr, py::cast(&s))` 第三个参数的作用。去掉会怎样？

??? 解答 ???

`py::cast(&s)` 把 `Storage*` 转成 `py::object`，作为返回的 NumPy 数组的"base object"。NumPy 数组持有这个 base 的引用，只要数组还活着，base（Storage）就不会被 Python 回收。

去掉的话，NumPy 数组只持裸指针 `ptr`，不持 Storage 引用。如果 Python 端 `s` 被回收，Storage 析构，`ptr` 悬垂，再访问数组就 use-after-free。这是 pybind11 + NumPy 零拷贝的**必经安全措施**。

---

## 8.14 关键测试解读

C++ 核心共有 155 个测试（`test_cpp_ext.py` 18 + `test_cpp_autograd.py` 31 + `test_cpp_tensor.py` 23 + `test_cpp_allocator.py` 5 + `test_cpp_math_ops.py` 20 + `test_cpp_loss_ops.py` 12 + `test_cpp_tensor_ops.py` 12 + `test_cpp_compare_reduce.py` 16 + `test_cpp_profiler.py` 3 + `test_cpp_hooks_anomaly.py` 9 + `test_cpp_checkpoint.py` 6），全过。本节挑 `test_cpp_ext.py` 中的代表性测试逐个看它们验证什么：

### 8.14.1 加载与版本

```python
def test_cpp_extension_loaded():
    from minitorch import _cpp_ext
    assert _cpp_ext.__version__ == "0.2.0"
```

验证 `.pyd` 能 import、`__version__` 属性挂上了。这是"编译链通了"的烟雾测试。

### 8.14.2 创建与属性

```python
def test_tensor_creation():
    t = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert t.shape == [2, 2]
    assert t.numel == 4
    assert t.ndim == 2
```

验证构造函数、`shape`/`numel`/`ndim` 属性。`t.shape` 是属性（`def_property_readonly`）不是方法。

### 8.14.3 算子正确性

```python
def test_add():
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = _cpp_ext.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2])
    c = _cpp_ext.add(a, b)
    assert c.to_vector() == [6.0, 8.0, 10.0, 12.0]
```

`add`/`sub`/`mul`/`matmul`/`neg`/`relu` 各一个，验证数值正确。`to_vector()` 把数据拉平成 Python list 做断言。

### 8.14.4 view 语义

```python
def test_transpose():
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = a.transpose()
    assert b.to_vector() == [1.0, 3.0, 2.0, 4.0]   # 转置后的逻辑顺序
    assert not b.is_contiguous()                    # 但物理上非连续

def test_contiguous():
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = a.transpose()
    c = b.contiguous()
    assert c.is_contiguous()
    assert c.to_vector() == [1.0, 3.0, 2.0, 4.0]   # 数据一样，但连续了
```

这两个测试是 view 操作的精髓：`transpose` 不拷贝（`to_vector` 按逻辑顺序读出转置结果），`contiguous` 物理化（数据一样但 strides 变标准形）。

### 8.14.5 NumPy 互操作

```python
def test_numpy_roundtrip():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = _cpp_ext.TensorImpl.from_numpy(arr)
    assert t.shape == [2, 2]
    result = t.numpy()
    assert np.allclose(result, arr)
```

`from_numpy` → `numpy()` 往返不丢数据。这是绑定层与 NumPy 生态衔接的验证。

### 8.14.6 reshape 跨 contiguous

```python
def test_reshape():
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    b = a.transpose().reshape([6])
    assert b.shape == [6]
    assert b.to_vector() == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
```

`a.transpose()` 非连续，`reshape` 自动先 `contiguous` 再 `view`。结果 `[1,4,2,5,3,6]` 是转置后逻辑顺序的拉平。

---

## 8.15 优劣势总结

### 8.15.1 优势

1. **性能**：逐元素算子比 Python 快一个数量级（C++ 循环 + 编译器优化 + 无解释器开销）。
2. **接 CUDA 的前置**：CUDA kernel 只能从 C++ 调，有了 C++ 核心才能做 Ch10。
3. **与真实 PyTorch 同构**：学完 minitorch 的 C++ 层，看 `c10/`、`aten/` 没有概念鸿沟。
4. **行为等价可验证**：复用前七章的 Python 测试套件，一行不改全过，证明重写没引入 bug。
5. **前端零改动**：用户的 `Tensor`、`nn.Module`、训练循环代码不变，平滑升级。

### 8.15.2 劣势

1. **构建链复杂**：CMake + pybind11 + 编译器，跨平台有门槛（尤其 Windows 的 MSVC）。
2. **调试难**：C++ 层的 bug 要用 gdb/LLDB/Visual Studio 调，不如 Python 的 pdb 顺手。
3. **迭代慢**：改一行 C++ 要重新编译（几秒到几十秒），Python 改完即跑。
4. **教学版简化多**：只 double、只 host、无 BLAS、无并行，性能离真实 PyTorch 还很远。
5. **绑定样板**：每个类/算子要写绑定代码，量大（pybind11 已是最省的方案，但仍 125 行）。

### 8.15.3 什么时候值得用 C++

- **热路径**：被调几百万次的算子，Python 开销不可接受。
- **要接 GPU/异构**：CUDA/ROCm/SYCL 只能从 C++ 进。
- **要发 wheel 给别人用**：C++ 编译成二进制，用户不用看 Python 源码也能用。
- **要精确控制内存**：自定义 allocator、内存池、零拷贝。

反过来，**autograd 引擎、nn.Module、训练循环**这些"框架逻辑"留在 Python 更好——它们不在热路径，且 Python 写起来清晰 10 倍。minitorch 的分层正是这个原则的体现。

---

## 8.16 下一章预告

本章搭好了 C++ 核心的"地基"——Storage、TensorImpl、基本算子、pybind11 绑定。但 autograd 引擎、多线程并行、double backward、自定义 Allocator、Profiler、Hooks、Checkpointing 这些**高级特性**还没有展开。下一章（Ch9）深入讲解这些内容：

- **C++ Autograd 引擎**：Node 虚函数设计、AccumulateGrad、grad_mode RAII、与 Python autograd 对照。
- **多线程 Engine**：ThreadPool + 原子依赖计数 + 并行执行无依赖 Node，对照 PyTorch 的 `Engine::execute_with_thread_pool`。
- **double backward**：`create_graph=true` 时建二阶图，MulNode 用 `autograd::mul` 而非 `ops::mul`，broadcast_to 保留 grad_fn 的 bug 修复。
- **自定义 Allocator**：Allocator 接口、DefaultAllocator 统计、PoolAllocator 内存池，对照 `c10::Allocator`。
- **Profiler**：记录每个 Node 执行耗时和内存分配，对照 `torch.autograd.profiler`。
- **梯度钩子**：`register_hook` 在 AccumulateGrad 中调用，可修改梯度。
- **Anomaly Detection**：检测 NaN/Inf 梯度并抛异常，对照 `torch.autograd.detect_anomaly()`。
- **Gradient Checkpointing**：用重计算换内存，前向 NoGrad、backward 重执行前向，对照 `torch.utils.checkpoint.checkpoint()`。

这些特性让 minitorch 的 C++ 核心从"能跑"升级到"能和真实 PyTorch 对照工程细节"。
