# A5 完整 dispatch key 链

> 本附录对应主线 Ch2/Ch10。Ch10 实现了简单的 device 路由 dispatcher（CPU/CUDA），本附录讲解 PyTorch 完整的 dispatch key 体系——它是 PyTorch 能同时支持多设备、autograd、autocast 的架构基石。

---

## A5.1 为什么需要 dispatch key

同一个算子名 `"add"` 在不同场景下需要不同实现：

| 场景 | 需要的实现 |
|------|-----------|
| CPU 张量 | `cpu_add` |
| CUDA 张量 | `cuda_add` |
| 需要梯度 | `autograd_add`（先建图再调底层） |
| 混合精度 | `autocast_add`（先转 fp16 再调） |
| 量化推理 | `quantized_add`（int8 kernel） |
| 元数据推导 | `meta_add`（只算 shape/dtype，不算数据） |

如果用 if/else 硬编码，每个算子都要写 N 个分支，组合爆炸。**dispatch key** 把这些维度正交化，每个维度一个 key，按优先级查表。

### A5.1.1 组合爆炸问题

不用 dispatch key 的伪代码：

```cpp
// 反模式：if/else 硬编码
Tensor add(const Tensor& a, const Tensor& b) {
    if (a.requires_grad() && a.is_cuda() && is_autocast()) {
        return autograd_autocast_cuda_add(a, b);
    } else if (a.requires_grad() && a.is_cpu() && is_autocast()) {
        return autograd_autocast_cpu_add(a, b);
    } else if (a.is_cuda() && a.is_quantized()) {
        return quantized_cuda_add(a, b);
    } else if (a.is_cpu() && a.is_quantized()) {
        return quantized_cpu_add(a, b);
    }
    // ... 几十个分支 ...
}
```

**问题**：
- 每个算子都要写 N×M 个分支（N 设备 × M 功能）
- 新增一个维度（如新设备 XLA）要改所有算子
- 代码重复、易错

**dispatch key 的解法**：每个维度独立，查表组合。

---

## A5.2 minitorch 的简单 dispatcher（Ch10 回顾）

```cpp
// aten/dispatcher.h
enum class Device { CPU, CUDA };

class Dispatcher {
    // op_name → (device → kernel)
    std::unordered_map<std::string,
        std::unordered_map<Device, KernelFn>> table_;
};

// 注册
REGISTER_CPU_OP("add", cpu_add);
REGISTER_CUDA_OP("add", cuda_add);

// 调用：按张量的 device 自动路由
TensorImplPtr result = dispatcher.call("add", {a, b});
// a.device == CPU → 调 cpu_add
// a.device == CUDA → 调 cuda_add
```

**只有一层 key**（device），够用于教学。PyTorch 的 dispatcher 有**十几层 key**。

### A5.2.1 minitorch dispatcher 的实现

```cpp
class Dispatcher {
    std::unordered_map<std::string,
        std::unordered_map<Device, KernelFn>> table_;

public:
    void register_op(const std::string& name, Device dev, KernelFn fn) {
        table_[name][dev] = fn;
    }

    TensorImplPtr call(const std::string& name,
                       const std::vector<TensorImplPtr>& args) {
        // 从第一个张量获取 device
        Device dev = args[0]->device();
        auto it = table_.find(name);
        if (it == table_.end()) throw std::runtime_error("op not found");
        auto& dev_table = it->second;
        auto jt = dev_table.find(dev);
        if (jt == dev_table.end()) throw std::runtime_error("device not supported");
        return jt->second(args);
    }
};
```

**核心结构**：`op_name → (device → kernel)` 的嵌套查表。PyTorch 的结构相同，只是 key 更多。

---

## A5.3 PyTorch 的 dispatch key 体系

### A5.3.1 key 的分类

PyTorch 的 dispatch key 分为几大类，按**优先级从高到低**排列：

```
┌─ 功能 key（Functionality Keys）─────────────────────────┐
│  AutocastCUDA    ← 混合精度自动转换                     │
│  AutocastCPU     ← CPU 混合精度                         │
│  PythonTLSSnapshot ← Python 线程局部状态                │
│  Python          ← Python fallback                      │
│  Functionalize   ← 函数化变换（vmap、反向模式 AD）       │
│  Autograd        ← 自动微分（建图 + 反向）               │
│  Negative        ← 负数处理                              │
├─ 后端 key（Backend Keys）──────────────────────────────┤
│  CPU             ← CPU 实现                             │
│  CUDA            ← CUDA 实现                            │
│  MPS             ← Apple Metal                         │
│  XLA             ← Google TPU                           │
│  Meta            ← 只推导 shape/dtype                   │
│  QuantizedCPU    ← int8 CPU                             │
│  QuantizedCUDA   ← int8 CUDA                           │
│  ...                                                     │
├─ 复合 key（Composite Keys）────────────────────────────┤
│  CompositeImplicitAutograd ← 用 autograd 公式组合       │
│  CompositeExplicitAutograd ← 需手动注册 backward        │
└─────────────────────────────────────────────────────────┘
```

### A5.3.2 key 的优先级

当调用 `torch.add(a, b)` 时，dispatcher 收集输入张量的所有 key，按优先级从高到低查找：

```
a = torch.randn(3, requires_grad=True).cuda()

a 的 key 集合:
  Autograd  (requires_grad=True)
  CUDA      (device=cuda)

查找顺序:
  1. Autograd → 找到 autograd_add → 调用
  2. autograd_add 内部去掉 Autograd key，调底层
  3. CUDA → 找到 cuda_add → 调用 GPU kernel
```

**关键**：Autograd key 的优先级高于 CUDA，所以 `torch.add` 先走 autograd（建图），再走 CUDA（计算）。

### A5.3.3 DispatchKeySet

每个张量有一个 key 集合（`DispatchKeySet`），是 bitmask：

```cpp
// c10/core/DispatchKey.h
enum class DispatchKey {
    CPU = 0,
    CUDA = 1,
    Autograd = 2,
    AutocastCUDA = 3,
    // ...
};

// DispatchKeySet 是 bitmask
struct DispatchKeySet {
    uint64_t bits_;
    bool has(DispatchKey k) const { return bits_ & (1ULL << int(k)); }
    DispatchKeySet remove(DispatchKey k) const {
        return {bits_ & ~(1ULL << int(k))};
    }
};

// 张量的 key 集合
DispatchKeySet Tensor::key_set() const {
    DispatchKeySet keys;
    if (requires_grad_) keys = keys | DispatchKey::Autograd;
    keys = keys | device_to_key(device_);
    return keys;
}
```

**bitmask 的优势**：O(1) 检查 key 是否存在，O(1) 添加/删除 key。

---

## A5.4 Autograd dispatch key

### A5.4.1 自动包反向

Autograd key 的魔法：**用户调用 `torch.add(a, b)`，dispatcher 自动先建图再计算**。

```cpp
// PyTorch 内部伪代码
Tensor autograd_add(const Tensor& a, const Tensor& b) {
    // 1. 去掉 Autograd key，调底层
    auto result = dispatcher.call("add", {a, b},
                                  /*skip_key=*/Autograd);

    // 2. 如果需要梯度，建图
    if (a.requires_grad() || b.requires_grad()) {
        auto grad_fn = std::make_shared<AddBackward>();
        grad_fn->next_edges = {a.grad_fn(), b.grad_fn()};
        result.set_grad_fn(grad_fn);
    }

    return result;
}
```

**对比 minitorch**：我们手动分了 `ops::add`（纯计算）和 `autograd::add`（建图），用户要选调哪个。PyTorch 的 dispatch key 让用户只调 `torch.add`，autograd 自动包装。

### A5.4.2 不同算子的 autograd 策略

| 策略 | 含义 | 示例 |
|------|------|------|
| **AutogradImpl** | 手动注册 backward | `add` → `AddBackward` |
| **CompositeImplicitAutograd** | 用已有算子组合 | `softplus = log(1 + exp(x))`，自动用 `log`/`exp` 的 backward |
| **CompositeExplicitAutograd** | 需手动注册但非自动 | `dropout`（有随机性，不能自动组合） |

```python
# CompositeImplicitAutograd 示例
# softplus 只需注册前向，backward 自动由 log/exp 的 backward 组合
@torch.ops.CompositeImplicitAutograd
def softplus(x):
    return torch.log(1 + torch.exp(x))
# → backward 自动: d/dx log(1+exp(x)) = sigmoid(x)
```

### A5.4.3 Autograd 的 key 剥离

autograd_add 调用底层时，必须**去掉 Autograd key**，否则会无限递归：

```cpp
Tensor autograd_add(const Tensor& a, const Tensor& b) {
    // 去掉 Autograd key，只保留后端 key (CPU/CUDA)
    auto result = dispatcher.call("add", {a, b},
                                  /*skip_keys=*/DispatchKeySet(DispatchKey::Autograd));
    //                         ↑ 去掉 Autograd
    // → 现在只查 CPU/CUDA key，不会再次进入 autograd_add

    if (a.requires_grad()) {
        // 建图...
    }
    return result;
}
```

**如果不去掉**：`autograd_add` 调底层 → 底层又查到 Autograd key → 又调 `autograd_add` → 无限递归。

---

## A5.5 Meta key：shape 推导

Meta key 的实现**不计算数据**，只推导 shape/dtype/strides：

```cpp
Tensor meta_add(const Tensor& a, const Tensor& b) {
    auto result_shape = broadcast_shapes(a.shape(), b.shape());
    return Tensor(/*shape=*/result_shape, /*dtype=*/a.dtype(),
                  /*data=*/nullptr);  // 无数据
}
```

**用途**：
- `torch.empty_like(x)`：先走 Meta 推导 shape，再分配内存
- `torch.compile`：编译时用 Meta 推导所有中间 shape，不跑真数据
- 异构计算：先检查 shape 兼容性，再分派到 CPU/CUDA

### A5.5.1 Meta 的实际应用

```python
# torch.compile 用 Meta 推导 shape
@torch.compile
def fn(x):
    y = x @ x.T       # shape: [N, M] @ [M, N] = [N, N]
    z = y.sum(dim=1)  # shape: [N]
    return z

# 编译时:
#   1. 创建 Meta 张量（只有 shape，无数据）
#   2. 跑前向 → 推导所有中间 shape
#   3. 用 shape 信息生成优化代码（Triton kernel）
#   4. 真数据用生成的代码执行
```

### A5.5.2 Meta 的注册

```cpp
// aten/src/ATen/native/Meta.cpp
TORCH_LIBRARY_IMPL(aten, Meta, m) {
    m.impl("add",      &meta_add);
    m.impl("mul",      &meta_mul);
    m.impl("matmul",   &meta_matmul);
    m.impl("conv2d",   &meta_conv2d);
    // → 所有算子的 shape 推导规则
}
```

---

## A5.6 Functionalize key

Functionalize 是 PyTorch 最复杂的 dispatch key，用于：

- **vmap**：自动批量变换
- **反向模式 AD**（functorch）：高阶自动微分
- **view 消除**：把 view 操作变成 clone + 计算，简化语义

```python
# functorch 示例
from functorch import vmap, grad

# vmap: 自动把单样本函数变成批量函数
batched_fn = vmap(lambda x: x ** 2)  # 自动批量
result = batched_fn(torch.randn(1000))  # 一次调用处理 1000 个

# grad: 自动求导
dfdx = grad(lambda x: x ** 2)  # → 2x
print(dfdx(torch.tensor(3.0)))  # → 6.0
```

Functionalize 通过**重写算子语义**实现这些变换，是 PyTorch 2.0 的 functorch 基础。

### A5.6.1 View 消除

```python
# view 操作（共享存储）
x = torch.randn(4, 4)
y = x.view(16)      # y 和 x 共享存储
y[0] = 999          # 修改 y 也修改 x

# Functionalize 把 view 变成 clone:
# y = x.view(16) → y = x.clone().reshape(16)
# → y 和 x 独立，修改 y 不影响 x
# → 简化了 autograd 的语义（不用处理 view 的梯度）
```

**为什么需要 view 消除**：view 操作的梯度很复杂（需要处理别名关系），Functionalize 把 view 变成 clone 后，autograd 只需处理简单的不共享情况。

---

## A5.7 Autocast key

混合精度训练时，Autocast key 自动把输入转成 fp16：

```python
with torch.autocast(device_type="cuda"):
    # 在这个上下文里，算子自动走 AutocastCUDA key
    x = torch.randn(3, 3)  # fp32
    y = torch.matmul(x, x)  # ← Autocast 把 x 转 fp16，调 cuda_matmul_fp16
    # 某些算子（如 softmax）保持 fp32
    z = torch.softmax(y, dim=-1)  # ← Autocast 保持 fp32
```

```cpp
Tensor autocast_matmul(const Tensor& a, const Tensor& b) {
    // 自动转 fp16
    auto a_fp16 = a.to(kHalf);
    auto b_fp16 = b.to(kHalf);
    return dispatcher.call("matmul", {a_fp16, b_fp16},
                           /*skip_key=*/AutocastCUDA);
}
```

**哪些算子转 fp16**：matmul、conv、linear（计算密集型）
**哪些保持 fp32**：softmax、layer_norm、loss（数值敏感型）

### A5.7.1 Autocast 的算子分类

```cpp
// aten/src/ATen/autocast_mode.cpp
// 1. 降精度算子（compute-intensive）
TORCH_LIBRARY_IMPL(aten, AutocastCUDA, m) {
    m.impl("matmul",   &autocast_to_fp16<matmul>);
    m.impl("conv2d",   &autocast_to_fp16<conv2d>);
    m.impl("linear",   &autocast_to_fp16<linear>);
}

// 2. 保持精度算子（precision-sensitive）
TORCH_LIBRARY_IMPL(aten, AutocastCUDA, m) {
    m.impl("softmax",    &autocast_passthrough<softmax>);     // 保持 fp32
    m.impl("layer_norm", &autocast_passthrough<layer_norm>);
    m.impl("log_softmax", &autocast_passthrough<log_softmax>);
}
```

**分类原则**：
- **计算密集型**（matmul, conv）：fp16 加速明显，精度损失可接受
- **数值敏感型**（softmax, layer_norm）：fp32 保持数值稳定
- **降精度型**（sum, mean）：在 fp16 下累加可能溢出，保持 fp32

---

## A5.8 dispatch 的查找流程

完整调用 `torch.matmul(a, b)` 的查找流程：

```
1. 收集 key: a.keys = {Autograd, CUDA, AutocastCUDA}
2. 按优先级排序: AutocastCUDA > Autograd > CUDA

3. 查 AutocastCUDA:
   → 找到 autocast_matmul
   → 内部转 fp16，去掉 AutocastCUDA key
   → 递归调 dispatcher

4. 查 Autograd:
   → 找到 autograd_matmul
   → 内部建图，去掉 Autograd key
   → 递归调 dispatcher

5. 查 CUDA:
   → 找到 cuda_matmul_fp16
   → 调 GPU kernel
   → 返回结果

6. 结果沿调用栈返回:
   CUDA result → Autograd 加 grad_fn → Autocast 返回
```

### A5.8.1 查找的代码实现

```cpp
// c10/core/Dispatcher.h
class Dispatcher {
    // op_name → DispatchTable
    std::unordered_map<std::string, DispatchTable> ops_;

public:
    Tensor call(const std::string& op_name,
                const std::vector<Tensor>& args) {
        // 1. 收集所有输入的 key
        DispatchKeySet keys;
        for (auto& t : args) keys = keys | t.key_set();

        // 2. 查表
        auto& table = ops_[op_name];
        auto kernel = table.lookup(keys);
        // → 按 keys 的优先级，找到第一个注册的 kernel

        // 3. 调用
        return kernel(args);
    }
};

// DispatchTable::lookup
KernelFn DispatchTable::lookup(DispatchKeySet keys) const {
    // 按优先级遍历 key
    for (auto key : keys.ordered_keys()) {  // 从高优先级到低
        if (kernels_[int(key)]) return kernels_[int(key)];
    }
    // 查 fallback
    return fallback_;
}
```

---

## A5.9 fallback 机制

不是每个算子都注册了所有 key。**fallback** 是"没找到时的默认行为"：

```cpp
// 查找流程
auto kernel = table.lookup(op_name, key);
if (kernel == nullptr) {
    // 查 fallback
    kernel = table.lookup_fallback(key);
    if (kernel == nullptr) {
        // 查下一优先级 key
        kernel = table.lookup(op_name, next_key);
    }
}
```

常见 fallback：

| key | fallback | 含义 |
|-----|---------|------|
| CUDA | CPU | 没 CUDA 实现就跑 CPU（慢但能出结果） |
| Autograd | 无 | 没 autograd 实现则不建图 |
| Meta | 无 | 没 Meta 实现则不推导 shape |
| CompositeImplicitAutograd | 用前向组合 | 自动用已有算子的 backward |

### A5.9.1 CPU → CUDA fallback

```cpp
// 没注册 CUDA 实现的算子，fallback 到 CPU
Tensor cuda_op_fallback(const Tensor& a) {
    // 把数据搬到 CPU
    auto a_cpu = a.to(CPU);
    // 调 CPU 实现
    auto result_cpu = cpu_op(a_cpu);
    // 搬回 CUDA
    return result_cpu.to(CUDA);
}
// → 功能正确但慢（数据来回搬运）
```

---

## A5.10 注册宏详解

### A5.10.1 TORCH_LIBRARY

```cpp
// 定义算子签名
TORCH_LIBRARY(aten, m) {
    // 定义算子 "add" 的签名
    m.def("add(Tensor self, Tensor other) -> Tensor");
    // → 注册到 dispatcher，但还没有实现

    // 定义带 overload 的算子
    m.def("add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor");
}
```

### A5.10.2 TORCH_LIBRARY_IMPL

```cpp
// 为特定 key 注册实现
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("add", &cpu_add);        // CPU 实现
}

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    m.impl("add", &cuda_add);       // CUDA 实现
}

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
    m.impl("add", &autograd_add);   // Autograd 实现
}

TORCH_LIBRARY_IMPL(aten, Meta, m) {
    m.impl("add", &meta_add);       // Meta 实现
}
```

**对比 minitorch 的注册宏**：

```cpp
// minitorch (Ch10)
REGISTER_CPU_OP("add", cpu_add);
REGISTER_CUDA_OP("add", cuda_add);
// → 只有 device 一个维度

// PyTorch
TORCH_LIBRARY_IMPL(aten, CPU, m)     { m.impl("add", &cpu_add); }
TORCH_LIBRARY_IMPL(aten, CUDA, m)    { m.impl("add", &cuda_add); }
TORCH_LIBRARY_IMPL(aten, Autograd, m){ m.impl("add", &autograd_add); }
TORCH_LIBRARY_IMPL(aten, Meta, m)    { m.impl("add", &meta_add); }
// → 多个维度，每个维度独立注册
```

### A5.10.3 注册的完整示例

```cpp
// 1. 定义算子（一次）
TORCH_LIBRARY(aten, m) {
    m.def("matmul(Tensor self, Tensor other) -> Tensor");
}

// 2. 各 key 注册实现（多次，分散在不同文件）
// aten/src/ATen/native/CPU.cpp
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("matmul", &cpu_matmul);
}

// aten/src/ATen/native/cuda/Blas.cpp
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    m.impl("matmul", &cuda_matmul);
}

// torch/csrc/autograd/FunctionsManual.cpp
TORCH_LIBRARY_IMPL(aten, Autograd, m) {
    m.impl("matmul", &autograd_matmul);
}

// aten/src/ATen/native/Meta.cpp
TORCH_LIBRARY_IMPL(aten, Meta, m) {
    m.impl("matmul", &meta_matmul);
}
```

**关键**：定义一次，实现分散。新增设备只需新增 `TORCH_LIBRARY_IMPL`，不改已有代码。

---

## A5.11 添加新后端

### A5.11.1 添加新设备

以添加 NPU（华为昇腾）为例：

```cpp
// 1. 添加 dispatch key
enum class DispatchKey {
    // ...
    NPU,  // 新增
};

// 2. 注册算子实现
TORCH_LIBRARY_IMPL(aten, NPU, m) {
    m.impl("add",    &npu_add);
    m.impl("matmul", &npu_matmul);
    m.impl("conv2d", &npu_conv2d);
    // ... 为所有算子注册 NPU 实现
}

// 3. 张量支持 NPU device
class Tensor {
    Device device_;  // 包含 NPU
    // ...
};

// 用户使用
x = torch.randn(3, device="npu")
y = torch.matmul(x, x)  # → 自动走 NPU key
```

**工作量**：需要为所有算子实现 NPU 版本（调用 NPU 驱动 API）。这是 PyTorch 支持新硬件的标准方式。

### A5.11.2 添加新功能

以添加 "DebugTrace"（记录所有算子调用）为例：

```cpp
// 1. 添加 dispatch key
enum class DispatchKey {
    // ...
    DebugTrace,  // 新功能 key
};

// 2. 注册实现（包装所有算子）
TORCH_LIBRARY_IMPL(aten, DebugTrace, m) {
    m.impl("add", [](const Tensor& a, const Tensor& b) {
        log_call("add", a, b);  // 记录
        return dispatcher.call("add", {a, b},
                               /*skip=*/DebugTrace);  // 调底层
    });
}

// 3. 启用
// 张量的 key_set 包含 DebugTrace → 自动记录所有调用
```

**关键**：新功能不需要改任何已有算子，只需注册新 key 的实现。这是 dispatch key 的核心价值——**开放扩展**。

### A5.11.3 实际案例: PyTorch 的 PrivateUse1

PyTorch 预留了 `PrivateUse1` / `PrivateUse2` / `PrivateUse3` 三个 key 供第三方扩展：

```cpp
// 第三方后端（如华为 NPU）用 PrivateUse1
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add", &my_npu_add);
    m.impl("matmul", &my_npu_matmul);
    // ...
}

// 注册 device 名
torch::register_device("npu", /*key=*/c10::DispatchKey::PrivateUse1);

// 用户使用
x = torch.randn(3, device="npu")  # → PrivateUse1 key
```

**这就是 torch_npu（华为昇腾）、torch_mlu（寒武纪）等第三方后端的实现方式**——不需要改 PyTorch 源码，只用 `PrivateUse1` key 注册。

---

## A5.12 minitorch vs PyTorch dispatcher

| 维度 | minitorch | PyTorch |
|------|-----------|---------|
| key 数量 | 1（device） | ~20 |
| 查找 | `dict[op_name][device]` | `DispatchTable` 按 key 优先级 |
| autograd | 手动调 `autograd::add` | 自动（Autograd key） |
| autocast | 未实现 | AutocastCUDA key |
| meta | 未实现 | Meta key |
| fallback | 未实现 | 多级 fallback |
| 注册 | `REGISTER_CPU_OP` 宏 | `TORCH_LIBRARY_IMPL` 宏 |
| key 表示 | `enum Device` | `DispatchKeySet` (bitmask) |

### A5.12.1 为什么 minitorch 只做 device key

教学目标下，device key 足以讲清 dispatcher 的核心思想：
- **op_name → (key → kernel)** 的查表结构
- 新增 device 只需注册新 kernel，不改调用点
- 解耦"算子名"和"实现"

其他 key（Autograd、Autocast 等）是工程扩展，原理相同——只是 key 更多、优先级更复杂。

### A5.12.2 从 minitorch 到 PyTorch 的扩展路径

```
minitorch (Ch10):
  key = {CPU, CUDA}
  → 用户手动选 ops::add 或 autograd::add

+ Autograd key:
  key = {Autograd, CPU, CUDA}
  → 用户只调 add，自动建图

+ Autocast key:
  key = {AutocastCUDA, Autograd, CPU, CUDA}
  → 自动混合精度

+ Meta key:
  key = {Meta, AutocastCUDA, Autograd, CPU, CUDA}
  → shape 推导

+ Functionalize key:
  → vmap, 高阶 AD
```

每一步都是**添加新 key**，不改已有代码。这就是 dispatch key 架构的扩展性。

---

## A5.13 与真实 PyTorch 对照

| 概念 | PyTorch | 文件 |
|------|---------|------|
| DispatchKey | `c10::DispatchKey` | `aten/src/ATen/core/DispatchKey.h` |
| DispatchTable | `c10::DispatchTable` | `c10/core/DispatchTable.h` |
| Dispatcher | `c10::Dispatcher` | `c10/core/Dispatcher.h` |
| DispatchKeySet | `c10::DispatchKeySet` | `c10/core/DispatchKey.h` |
| 注册宏 | `TORCH_LIBRARY_IMPL` | `aten/src/ATen/library.h` |
| 算子定义 | `TORCH_LIBRARY` | `aten/src/ATen/library.h` |
| Autograd key | `Autograd` dispatch key | `torch/csrc/autograd/autograd.cpp` |
| Meta key | `Meta` dispatch key | `aten/src/ATen/native/Meta.cpp` |
| Autocast | `AutocastCUDA` dispatch key | `aten/src/ATen/autocast_mode.cpp` |
| Functionalize | `Functionalize` dispatch key | `aten/src/ATen/Functionalize.cpp` |
| Python fallback | `Python` dispatch key | `torch/csrc/autograd/python_variable.cpp` |

---

## A5.14 Python key 与 Negative key

### A5.14.1 Python dispatch key

Python key 是最后的 fallback——当没有 C++ 实现时，调用 Python 实现：

```python
# 自定义算子用 Python 实现
@torch.library.impl("mylib::my_op", "Python")
def my_op_python(x):
    # 纯 Python 实现
    result = x.clone()
    for i in range(x.shape[0]):
        result[i] = complex_python_logic(x[i])
    return result

# 调用时走 Python key
y = torch.ops.mylib.my_op(x)
# → 查 C++ keys: 未找到
# → 查 Python key: 找到 my_op_python
```

**用途**：
- 快速原型（先 Python 实现，验证正确性，再写 C++）
- 不需要性能的算子
- 与 Python 库交互（如 numpy）

### A5.14.2 Negative key

Negative key 处理负数索引的特殊语义：

```python
x = torch.randn(10)
y = x[-1]  # 负索引: -1 表示最后一个

# 内部:
# 1. 查 Negative key → 找到 negative_index_select
# 2. negative_index_select 把 -1 转成 9
# 3. 调底层 index_select(x, 9)
```

**为什么需要单独的 key**：负索引的转换逻辑对所有算子相同，用一个 key 统一处理，避免每个算子都写负索引检查。

---

## A5.15 复合 key 详解

### A5.15.1 CompositeImplicitAutograd

```cpp
// 只注册前向，backward 自动组合
TORCH_LIBRARY_IMPL(aten, CompositeImplicitAutograd, m) {
    m.impl("softplus", [](const Tensor& x) {
        return torch::log(1 + torch::exp(x));
        // → backward 自动: d/dx log(1+exp(x)) = sigmoid(x)
        //   用 log 和 exp 的已注册 backward 组合
    });
}
```

**工作原理**：
1. 用户调 `softplus(x)`
2. 查到 CompositeImplicitAutograd 实现
3. 展开成 `log(1 + exp(x))`——调用 `log` 和 `exp`
4. `log` 和 `exp` 各自走 Autograd key，建图
5. 最终的图是 `log` 和 `exp` 的组合，backward 自动链式法则

### A5.15.2 CompositeExplicitAutograd

```cpp
// 需要手动注册 backward（不能自动组合）
TORCH_LIBRARY_IMPL(aten, CompositeExplicitAutograd, m) {
    m.impl("dropout", &dropout_forward);
}
m.def("dropout.backward", &dropout_backward);  // 手动注册
```

**为什么 dropout 不能自动组合**：dropout 有随机性，`dropout(x) = x * mask / p`，mask 是随机的。backward 需要**同一个 mask**，不能重新随机。所以必须手动注册 backward，保存 mask。

---

## A5.16 dispatch key 的历史演进

```
PyTorch 0.x:  无 dispatcher，每个算子手动 if/else 判断 device
PyTorch 1.0:  引入 dispatcher，最初只有 CPU/CUDA key
PyTorch 1.5:  加入 Autograd key（之前 autograd 是单独的层）
PyTorch 1.8:  加入 Meta key（用于 torch.compile 预研）
PyTorch 1.10: 加入 Functionalize key（functorch）
PyTorch 1.13: 加入 Autocast key（之前 autocast 是 Python 层）
PyTorch 2.0:  dispatcher 成熟，支持 torch.compile
```

**演进动机**：
- 早期：快速迭代，if/else 够用
- 中期：设备/功能增多，组合爆炸 → 需要 dispatcher
- 后期：torch.compile 需要精确控制算子分发 → dispatcher 成为核心

---

## A5.17 小结

PyTorch 的 dispatch key 体系是一个**多维分派表**：

1. **功能 key**（Autograd、Autocast、Functionalize）：包装/变换算子语义
2. **后端 key**（CPU、CUDA、Meta）：不同设备的实际实现
3. **复合 key**（CompositeImplicitAutograd）：用已有算子自动组合

调用 `torch.add(a, b)` 时，dispatcher 收集 a 的所有 key，按优先级从高到低查找。Autograd 先于 CUDA，所以自动建图；Autocast 先于 Autograd，所以自动转精度。

minitorch 只实现了 device key（CPU/CUDA），但核心结构（op_name → key → kernel）与 PyTorch 一致。理解了这个结构，就理解了 PyTorch 如何用一套 API 支持多设备、autograd、量化、混合精度。

**关键设计思想**：
- **正交化**：每个维度一个 key，避免组合爆炸
- **优先级**：功能 key > 后端 key，先变换再计算
- **开放扩展**：新增设备/功能只需注册新 key，不改已有代码
- **fallback**：未注册的 key 有默认行为，保证可用性
- **bitmask**：key 集合用 bitmask，O(1) 操作
- **PrivateUse1-3**：预留 key 供第三方后端扩展（torch_npu 等）
