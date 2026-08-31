# 第十�?CUDA �?dispatcher：异构计算与 device 路由

> 上一章我们把核心计算迁到 C++，但还只跑在 CPU 上。本章给 minitorch 接上 GPU�?
> 引入 **dispatcher**——一个按张量 device 把算子路由到 CPU kernel �?CUDA kernel �?
> 调度表——并写出真正�?CUDA kernel（`__global__` 函数、thread/block/grid、shared memory reduction）�?
> 你会看到"同一�?`add` �?CPU �?GPU 上是两份代码，但调用点只有一�?是怎么做到的，
> 这正�?PyTorch 能同时跑�?CPU/CUDA/MPS/XLA/... 的核心机制。本章代码在�?GPU 环境也完整可读�?

---

## 10.1 本章目标

读完本章后，你应当能够：

1. 解释**异构计算**的动机：CPU 擅长串行复杂逻辑，GPU 擅长大规模数据并行，要把算子分派到对的设备�?
2. 说出 **dispatcher �?device 路由**思想：`op_name �?(device �?kernel)` 的查表分发，为什么比 `if (device == CPU) ... else ...` 硬编码好�?
3. 讲清 **CUDA 编程模型**的三级层次：thread（线程）/ block（线程块�? grid（网格），以�?`threadIdx`/`blockIdx`/`blockDim`/`gridDim` 内置变量�?
4. 写一个最简单的 CUDA kernel：`__global__ void add_kernel(...)`，配边界守卫 `if (i < n)`，并�?`<<<blocks, threads>>>` 语法 launch�?
5. 解释 **kernel launch 是异步的**：`<<<>>>` 立即返回，要 `cudaStreamSynchronize` 或隐式同步（�?`cudaMemcpy` Device→Host）才等结果�?
6. �?**shared memory + `__syncthreads`** 写一�?block �?reduction（sum kernel），说出为什�?shared memory �?global memory 快、为什么必须同步�?
7. 实现 dispatch table：`unordered_map<op_name, unordered_map<device, KernelFn>>`，把 CPU �?CUDA 算子注册进去�?
8. 排查三类 CUDA 常见陷阱：异步执行导致的"读到旧数�?、host↔device 内存拷贝开销、shared memory bank conflict�?
9. 对照真实 PyTorch：`c10::DispatchTable` / `c10::DispatchKey` / `aten/native/cuda/`，说出我们的简化在哪�?
10. 讲清 CUDA 的诞生（2006 NVIDIA G80）和 PyTorch dispatcher 的演化（�?`TH` 函数指针表到 `c10::Dispatcher`）�?

---

## 10.2 原理铺垫：异构计算与 dispatcher

### 10.2.1 为什么要异构

CPU �?GPU 是两种截然不同的处理器：

| 维度 | CPU | GPU |
|------|-----|-----|
| 核心�?| 4�?4 个复杂核�?| 数千个简单核�?|
| 单核性能 | 强（深流水、分支预测、大缓存�?| 弱（顺序执行、简单控制） |
| 并行粒度 | 线程级（几线程到几十线程�?| 数据级（SIMT，一�?warp 32 线程同步执行�?|
| 内存 | DDR，几�?GB，延迟低 | 显存（HBM/GDDR），几十 GB，带宽高但延迟高 |
| 擅长 | 串行逻辑、复杂控制流、小数据 | 大规模数据并行（矩阵乘、逐元素、conv�?|

深度学习的核心计算（矩阵乘、卷积、逐元素算子）正好�?大规模数据并�?——一�?`[10000, 10000]` 的矩阵乘�?10^12 次乘加，GPU 的几千核心并行能�?CPU �?10�?00 倍�?

但训练循环的"框架逻辑"（建图、反向、optimizer step 的控制流）是串行的，GPU 跑反而慢。所�?*同一个程序里，框架逻辑�?CPU，热路径算子�?GPU**——这就是异构计算�?

### 10.2.2 dispatcher：按 device 路由

现在问题来了：`c = a + b` 这个加法，如�?`a`、`b` �?CPU 上，要走 CPU 加法；如果在 GPU 上，要走 GPU 加法。怎么写？

**朴素方案**：在每个算子�?`if`�?

```cpp
TensorImplPtr add(a, b) {
    if (a.device() == CPU)  return cpu_add(a, b);
    if (a.device() == CUDA) return cuda_add(a, b);
    throw ...;
}
```

这能跑，但有几个问题�?

1. **算子数量爆炸**：几十个算子，每个都要写这个 `if`，重复且易漏�?
2. **新增 device 要改所有算�?*：想�?MPS 支持？去每个算子里加一�?`if` 分支。灾难�?
3. **算子�?device 耦合**：`add` 的代码里出现 `cuda_add`，编译时就算�?CUDA 也要能过（要 `#ifdef`）�?
4. **难扩展其他维�?*：除�?device，还想按 dtype（float32 vs float16）、按 autograd（要不要建图）路由，`if` 嵌套会爆炸�?

**dispatcher 方案**：把"算子�?�?实现"解耦，用一张表�?

```
dispatch_table:
    "add"   �?{ CPU: cpu_add,   CUDA: cuda_add   }
    "mul"   �?{ CPU: cpu_mul,   CUDA: cuda_mul   }
    "relu"  �?{ CPU: cpu_relu,  CUDA: cuda_relu  }
    "matmul"�?{ CPU: cpu_matmul,CUDA: cuda_matmul}
    ...
```

调用时：`dispatcher.call("add", {a, b})`——查表，�?`a.device()` �?kernel，调用�?*调用点不知道也不关心 device**�?

新增 device（如 MPS）：只写 `mps_add` �?`register_kernel("add", MPS, mps_add)`�?*任何调用点不�?*。这�?dispatcher 的核心价值�?

!!! tip "心智模型"
dispatcher 就像餐厅�?菜单—厨�?映射：菜单上�?宫保鸡丁"（op name），后厨有川菜厨师（CPU kernel）和粤菜师傅（CUDA kernel）。服务员（调用点）只报菜名，餐厅经理（dispatcher）看今天哪个厨师在岗（device）派单。新来个鲁菜师傅（新 device），菜单不用改，只需登记"鲁菜师傅也会做宫保鸡�?�?

### 10.2.3 真实 PyTorch �?dispatch key

真实 PyTorch �?dispatcher 比这复杂——除�?device，还要按 **dtype**�?*layout**（dense/sparse）�?*autograd**（要不要建图）�?*autocast**（要不要自动类型转换）路由。这些统�?**dispatch key**，组合起来有几百种�?

我们的教学版只按 device（CPU/CUDA）路由，是真�?dispatcher �?投影"——足以讲�?查表分发"的思想，又不被几百�?key 淹没�?

---

## 10.3 原理铺垫：CUDA 编程模型

### 10.3.1 三级层次：thread / block / grid

一�?kernel launch 会启动大量线程，它们被组织成三级层次�?

```
grid
├── block (0,0)
�?  ├── thread (0,0)   �?threadIdx = (0,0), blockIdx = (0,0)
�?  ├── thread (1,0)
�?  └── ... (blockDim.x × blockDim.y �?thread)
├── block (1,0)
�?  └── ...
└── ... (gridDim.x × gridDim.y �?block)
```

- **thread**：最小执行单元，跑一�?kernel 代码，有私有寄存器�?
- **block**：若�?thread 组成。block �?thread 可通过 **shared memory** 协作，可�?`__syncthreads()` 同步。block 之间**不能**直接通信或同步�?
- **grid**：一�?launch 的所�?block。所�?block 跑同一�?kernel 代码�?

每个 thread 用内置变量定位自己：

```cuda
int tid  = threadIdx.x;                    // block �?x 方向索引
int bid  = blockIdx.x;                     // grid �?block x 方向索引
int bdim = blockDim.x;                     // block �?x 方向 thread �?
int gdim = gridDim.x;                      // grid �?x 方向 block �?
int i    = bid * bdim + tid;               // 全局一维索引（最常用�?
```

### 10.3.2 SIMT �?warp

GPU 执行模型�?**SIMT**（Single Instruction, Multiple Threads）：�?32 �?thread 组成一�?**warp**，warp 内所�?thread **同一时刻执行同一条指�?*（不同数据）�?

如果 kernel 里有 `if (cond) A else B`，warp 内有�?thread �?A、有的走 B，GPU �?*串行执行两个分支**（warp divergence），性能下降。所以写 kernel 要尽量让 warp 内分支一致�?

### 10.3.3 内存层次

| 层次 | 范围 | 大小 | 速度 | 谁能访问 |
|------|------|------|------|----------|
| 寄存�?| �?thread | �?KB/线程 | 最�?| 私有 |
| shared memory | �?block | ~48 KB/block | 快（~20×寄存器延迟） | block 内所�?thread，可同步 |
| global memory（显存） | �?grid | �?GB | 慢（~100×shared�?| 所�?thread + host |
| host memory | CPU 内存 | 几十 GB | 最慢（�?PCIe 传） | host |

�?kernel 的核心优化思路�?*把数据从 global 搬到 shared，在 shared 里算，结果写�?global**。因�?shared �?global 快几十倍�?

### 10.3.4 异步执行与同�?

kernel launch �?*异步**的：

```cuda
add_kernel<<<blocks, threads>>>(da, db, dc, n);   // 立即返回�?
// 这里 dc 可能还没算完
cudaMemcpy(hc.data(), dc, ..., cudaMemcpyDeviceToHost);  // 这行会等 kernel
// 这里 hc 才有�?
```

`cudaMemcpy` Device→Host �?*隐式同步**（等之前所�?kernel 完成）。但 Device→Device 的拷贝是异步的。要显式等：

```cuda
cudaDeviceSynchronize();   // 等所�?stream 所�?kernel
cudaStreamSynchronize(stream);  // 等特�?stream
```

异步的好处：CPU 可以�?GPU 算的时候干别的（比如准备下一批数据）。坏处：容易写出"读到未算完数�?�?bug�?

### 10.3.5 launch 配置 `<<<blocks, threads, shared_mem, stream>>>`

完整�?launch 语法有四个参数：

```cuda
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args);
```

- `gridDim`：grid 的形状（多少�?block）。可�?`int`�?D）或 `dim3`�?D）�?
- `blockDim`：每�?block �?thread 数。同样可 `int` �?`dim3`�?
- `sharedMemBytes`：动�?shared memory 大小（字节）。对�?kernel 里的 `extern __shared__` 数组。默�?0�?
- `stream`：在哪个 stream �?launch。默�?0（默�?stream）。不�?stream 上的 kernel 可并发执行�?

教学版只用了前两个：`add_kernel<<<blocks, threads>>>(...)`。sum kernel 用了第三个（动�?shared mem）：`sum_kernel<<<blocks, threads, threads*sizeof(double)>>>`�?

**约束**�?

- `blockDim.x * blockDim.y * blockDim.z �?1024`（多�?GPU 一�?block 最�?1024 thread）�?
- `gridDim.x/y/z �?2^31 - 1`（grid 维度上限很大，但 block 总数�?SM 数量限制，多了排队）�?
- shared memory �?block 上限 ~48KB（可配置�?96KB，但会减少可同驻�?block 数）�?

### 10.3.6 SM 与占用率

GPU 由若�?**SM**（Streaming Multiprocessor）组成，每个 SM 能同时跑多个 block（co-residency）。一�?block 跑在哪个 SM 上由硬件决定，软件不可控�?

**占用�?*（occupancy�? 实际活跃 warp �?/ SM 最�?warp 数。影响占用率的因素：

- block �?thread 数（太少占用低）
- block 用的寄存器数（太多限�?co-residency�?
- block 用的 shared memory（太多限�?co-residency�?

教学�?block=256、不用多少寄存器�?shared mem，占用率高。真�?PyTorch 的复�?kernel（如 conv）会 profile �?block size 最大化占用率�?

!!! tip "占用率不是越高越�?"
100% 占用率不一定最快。如�?kernel 是访存受限（memory-bound），�?warp 也喂不饱显存带宽，反而增加调度开销。要 profile 看是 compute-bound 还是 memory-bound，对症下药�?

---

## 10.4 设计决策与权�?

| 决策�?| 选项 | 我们�?| 理由 |
|--------|------|--------|------|
| dispatch key | �?device / device+dtype / 全组�?| **�?device** | 教学简化；讲清路由思想即可 |
| 表结�?| 嵌套 hash map / 扁平数组 | **嵌套 `unordered_map`** | 简单直观；真实 PyTorch 用扁平数组（DispatchKey 下标）更高效 |
| kernel 签名 | 固定元数 / 变参 vector | **统一 `vector<TensorImplPtr> �?TensorImplPtr`** | 能放进同一张表；真�?PyTorch �?stackbased 变参更高�?|
| CUDA 内存 | 张量常驻 GPU / 每次来回�?| **教学版来回拷** | 复用 Ch8 �?host-only TensorImpl；真�?PyTorch 张量常驻 GPU |
| 何时同步 | 每步同步 / 流式异步 | **每步隐式同步** | 教学清晰；真�?PyTorch 全异�?+ caching allocator |
| reduction 算法 | 树形归约 / warp shuffle | **树形归约 + shared mem** | 讲清 `__syncthreads`；真�?PyTorch �?warp shuffle 更快 |
| 是否实现全部算子�?CUDA �?| 全做 / 只做几个演示 | **只做 add/mul/relu/sum** | 讲清模式即可；其余算子同�?|

### 10.4.1 为什么教学版"来回�?

真实 PyTorch 的张量可以常�?GPU：`a = a.cuda()` �?`a` �?Storage 在显存，后续算子都在 GPU 跑，不来回拷�?

但我们的 Ch8 `TensorImpl` �?`Storage` �?`std::vector<double>`（host 内存），没有"GPU Storage"概念。要支持常驻 GPU 得给 TensorImpl �?device 字段、给 Storage �?allocator——改动大�?

教学版取巧：每次�?CUDA 算子，把 host 数据拷到 GPU、算完、拷�?host、包成原 TensorImpl�?*语义正确，性能不如常驻**，但代码简单、能复用 Ch8 全部基础设施。文档里会反复强调这是教学简化�?

### 10.4.2 为什么用 `unordered_map` 而非数组

真实 PyTorch �?dispatch table �?*扁平数组**：`KernelFn table[NUM_DISPATCH_KEYS]`，用 `DispatchKey`（int）当下标，O(1) 且缓存友好�?

我们�?`unordered_map<op_name, unordered_map<device, KernelFn>>`，O(1) 平均但常数大（两�?hash）。教学�?hash map 是因为它直观—�?op �?�?device �?kernel"的映射一目了然。性能差距在教学场景可忽略�?

---

## 10.5 代码逐行实现：dispatch table

`cpp/aten/dispatcher.h` �?dispatcher 的全部。逐段看：

### 10.5.1 Device 类型

```cpp
enum class DeviceType : int8_t {
    CPU = 0,
    CUDA = 1,
};

inline std::string device_name(DeviceType d) {
    switch (d) {
        case DeviceType::CPU:  return "cpu";
        case DeviceType::CUDA: return "cuda";
    }
    return "unknown";
}

using DispatchKey = DeviceType;   // 教学版直接用 device �?key
```

`enum class` 强类型枚举，不会隐式�?int。真�?PyTorch �?`DispatchKey` 是一个稠�?int 枚举，有几十个值（CPU/CUDA/MPS/XLA/AutogradCPU/AutocastCPU/...），我们只两个�?

### 10.5.2 Kernel 函数签名

```cpp
using KernelFn = std::function<TensorImplPtr(const std::vector<TensorImplPtr>&)>;
```

所有算子统一签名�?收一个张�?vector，返回一个张�?。这样不同元数的算子（一�?relu、二�?add、归�?sum）能放进同一张表�?

代价：每次调用要打包/解包 vector。真�?PyTorch �?stackbased 变参（直接从栈上读参数），省这步。教学版可接受�?

### 10.5.3 DispatchTable

```cpp
class DispatchTable {
public:
    void register_kernel(const std::string& op_name,
                         DispatchKey device,
                         KernelFn fn) {
        table_[op_name][static_cast<int>(device)] = std::move(fn);
    }

    bool has_kernel(const std::string& op_name, DispatchKey device) const {
        auto it = table_.find(op_name);
        if (it == table_.end()) return false;
        return it->second.find(static_cast<int>(device)) != it->second.end();
    }

    const KernelFn& lookup(const std::string& op_name, DispatchKey device) const {
        auto it = table_.find(op_name);
        if (it == table_.end()) {
            throw std::runtime_error("dispatch: unknown op '" + op_name + "'");
        }
        auto kit = it->second.find(static_cast<int>(device));
        if (kit == it->second.end()) {
            throw std::runtime_error(
                "dispatch: op '" + op_name + "' has no kernel for device " +
                device_name(device));
        }
        return kit->second;
    }

private:
    std::unordered_map<
        std::string,
        std::unordered_map<int, KernelFn>> table_;
};
```

嵌套 map：外层按 op 名查，内层按 device 查。`lookup` 在两处可能抛异常——op 没注册、或 op 在该 device 没注册。错误信息带 op 名和 device 名，方便调试�?

### 10.5.4 全局 Dispatcher 单例

```cpp
class Dispatcher {
public:
    static Dispatcher& instance() {
        static Dispatcher inst;   // Meyers 单例，线程安全（C++11+�?
        return inst;
    }

    void register_kernel(const std::string& op_name,
                         DispatchKey device,
                         KernelFn fn) {
        table_.register_kernel(op_name, device, std::move(fn));
    }

    TensorImplPtr call(const std::string& op_name,
                       const std::vector<TensorImplPtr>& args) const {
        if (args.empty()) {
            throw std::runtime_error("dispatch: call with no args");
        }
        DispatchKey device = device_of(args[0]);   // 按第一个张量定 device
        const KernelFn& fn = table_.lookup(op_name, device);
        return fn(args);
    }

private:
    DispatchTable table_;
};
```

`call` �?dispatcher 的入口：取第一个张量的 device，查表，�?kernel�?*调用点只�?`Dispatcher::instance().call("add", {a, b})`，不关心 device**�?

!!! warning "约定：所有输入张量必须同 device"
`call` 只看 `args[0]` �?device。如�?`a` �?CPU、`b` �?CUDA，会�?CPU 路由�?`b` 的数据在 GPU——崩。真�?PyTorch 会自动跨 device 拷贝（把 `b` 拉到 `a` �?device），我们简化为"逼用户显�?`.to()`"，错误更早暴露�?

### 10.5.5 device_of 与适配 helper

```cpp
inline DispatchKey device_of(const TensorImplPtr& t) {
    // 默认 CPU。CUDA 张量�?device 标记�?ops_cuda.cu
    (void)t;
    return DispatchKey::CPU;
}

template <typename Fn>
KernelFn unary_kernel(Fn fn) {
    return [fn](const std::vector<TensorImplPtr>& a) {
        if (a.size() != 1) throw std::runtime_error("unary op needs 1 arg");
        return fn(a[0]);
    };
}
template <typename Fn>
KernelFn binary_kernel(Fn fn) {
    return [fn](const std::vector<TensorImplPtr>& a) {
        if (a.size() != 2) throw std::runtime_error("binary op needs 2 args");
        return fn(a[0], a[1]);
    };
}
```

`unary_kernel`/`binary_kernel` 把固定元数的 lambda 适配�?`KernelFn` 的统一签名。注册时用：

```cpp
d.register_kernel("add", CPU, binary_kernel([](a,b){ return cpu_add(a,b); }));
```

---

## 10.6 代码逐行实现：CPU 算子注册

`cpp/aten/native/cpu/ops_cpu.cpp` �?Ch8 的算子注册到 dispatcher �?CPU 槽。这�?*不做计算**，只接线�?

```cpp
#include "ops_cpu.h"
#include "../../ops.h"   // Ch8 的算子实�?

namespace minitorch::native::cpu {
using namespace minitorch;
using namespace minitorch::ops;

void register_all_cpu_ops() {
    auto& d = Dispatcher::instance();

    d.register_kernel("add", DispatchKey::CPU,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return add(a, b); }));
    d.register_kernel("sub", DispatchKey::CPU,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return sub(a, b); }));
    d.register_kernel("mul", DispatchKey::CPU,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return mul(a, b); }));
    d.register_kernel("div", DispatchKey::CPU,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return div(a, b); }));
    d.register_kernel("neg", DispatchKey::CPU,
        unary_kernel([](const TensorImplPtr& a) { return neg(a); }));
    d.register_kernel("relu", DispatchKey::CPU,
        unary_kernel([](const TensorImplPtr& a) { return relu(a); }));
    d.register_kernel("sum", DispatchKey::CPU,
        unary_kernel([](const TensorImplPtr& a) { return sum(a, -1, false); }));
    d.register_kernel("mean", DispatchKey::CPU,
        unary_kernel([](const TensorImplPtr& a) { return mean(a, -1, false); }));
    d.register_kernel("matmul", DispatchKey::CPU,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return matmul(a, b); }));
}
} // namespace minitorch::native::cpu
```

**Ch8 的算子代码零改动**——`add`/`sub`/... 还是原来那些函数，只是被"注册"�?dispatcher。这演示�?dispatcher 的解耦价值：现有算子不用改就能接入�?

---

## 10.7 代码逐行实现：CUDA kernel

`cpp/aten/native/cuda/ops_cuda.cu` �?CUDA 源文件，�?**nvcc** 编译。逐段看：

### 10.7.1 加法 kernel

```cuda
__global__ void add_kernel(const double* __restrict__ a,
                           const double* __restrict__ b,
                           double* __restrict__ c,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

逐行�?

- `__global__`：这是从 host 调用、在 device 执行�?kernel。`__device__` 是只�?device 执行（不能从 host launch），`__host__` 是只�?host 执行�?
- `__restrict__`：告诉编译器 `a`/`b`/`c` 指针不别名（不指向同一块），编译器可放心重排、向量化。CUDA 优化关键�?
- `int i = blockIdx.x * blockDim.x + threadIdx.x;`：算�?thread 的全局一维索引。这�?1D launch 的标准公式�?
- `if (i < n)`�?*边界守卫**。launch �?thread 总数通常不是 n 的整数倍，多出来的 thread 要跳过，否则越界访问显存�?
- `c[i] = a[i] + b[i];`：每�?thread 处理一个元素，天然并行�?

### 10.7.2 ReLU kernel（演示分支）

```cuda
__global__ void relu_kernel(const double* __restrict__ a,
                            double* __restrict__ c,
                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = a[i];
        c[i] = x > 0.0 ? x : 0.0;
    }
}
```

`x > 0.0 ? x : 0.0` 是分支。warp 内如果有�?thread `x > 0`、有�?`x <= 0`，会 warp divergence——两分支串行。但 ReLU 分歧不严重（一半走一半不走，串行一次），性能影响小�?

### 10.7.3 求和 kernel（reduction + shared memory�?

```cuda
__global__ void sum_kernel(const double* __restrict__ a,
                           double* __restrict__ partial,
                           int n) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    double x = (i < n) ? a[i] : 0.0;
    sdata[tid] = x;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}
```

这是经典�?*两级 reduction**�?

1. `extern __shared__ double sdata[]`：动态大小的 shared memory，由 launch 时的第三个参数指定（`<<<blocks, threads, threads*sizeof(double)>>>`）。shared memory �?block 内所�?thread 共享的快速内存�?
2. 每个 thread 把自己负责的元素读进 `sdata[tid]`（越界填 0）。`__syncthreads()` 等所�?thread 都写完�?
3. **树形归约**：步�?`s` �?`blockDim/2` 逐次减半。每轮前 `s` �?thread �?`sdata[tid+s]` 累加�?`sdata[tid]`。`__syncthreads()` 每轮同步，否则下一轮可能读到上一轮没写完的数据�?
4. 归约结束�?`sdata[0]` 是本 block 的总和，由 `tid == 0` �?thread 写到 `partial[blockIdx.x]`�?

最�?host 上把 `partial[]` 加起来（或再 launch 一�?kernel 归约）�?

!!! tip "为什么用 shared memory"
如果直接�?global memory 里归约（`a[tid] += a[tid+s]`），每步都访问慢速显存。把数据先搬�?shared（快几十倍），在 shared 里归约，只最后写一�?global，性能高一个数量级。这�?CUDA reduction 的标准优化�?

!!! warning "__syncthreads 的规�?"
`__syncthreads()` 必须�?*所�?thread 都会执行�?*的位置调用。如果写�?`if (tid < s)` 里面，没�?if �?thread 永远不到同步点，整个 block 死锁。所以上面的代码�?`__syncthreads()` 放在 if 外面——所�?thread 都同步，只有部分 thread 干活�?

### 10.7.4 host �?wrapper：内存拷�?+ launch + 同步

```cuda
TensorImplPtr cuda_add(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (a->shape() != b->shape()) {
        throw std::runtime_error("cuda_add: shape 必须一致（教学版不支持广播�?);
    }
    int n = static_cast<int>(a->numel());

    std::vector<double> ha = a->to_vector();   // host 数据
    std::vector<double> hb = b->to_vector();
    std::vector<double> hc(static_cast<size_t>(n));

    double *da = nullptr, *db = nullptr, *dc = nullptr;
    cudaMalloc(&da, n * sizeof(double));
    cudaMalloc(&db, n * sizeof(double));
    cudaMalloc(&dc, n * sizeof(double));

    cudaMemcpy(da, ha.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;   // ceil(n / 256)
    add_kernel<<<blocks, threads>>>(da, db, dc, n);

    cudaMemcpy(hc.data(), dc, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    return make_tensor(hc, a->shape());
}
```

这是 host 代码（虽然写�?`.cu` 里），被 dispatcher 调用。流程：

1. �?`TensorImpl` �?host 数据（`to_vector()`）�?
2. `cudaMalloc` 在显存分配�?
3. `cudaMemcpy` Host→Device 把数据搬�?GPU�?
4. `<<<blocks, threads>>>` launch kernel。`blocks = ceil(n/256)` 保证 thread 总数 �?n�?
5. `cudaMemcpy` Device→Host 把结果搬回�?*这步隐式同步**，等 kernel 算完�?
6. `cudaFree` 释放显存�?
7. 包成�?`TensorImpl` 返回�?

!!! warning "教学版的低效"
每次调用�?`cudaMalloc`/`cudaFree`——真实场景这是性能灾难（分配显存很慢）。真�?PyTorch �?**caching allocator**：分配过的显存块缓存起来复用，不真释放。教学版为了简单每次真分配，文档里反复强调这是简化�?

### 10.7.5 注册 CUDA 算子

```cuda
void register_all_cuda_ops() {
    auto& d = Dispatcher::instance();
    using namespace minitorch;
    d.register_kernel("add", DispatchKey::CUDA,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return cuda_add(a, b); }));
    d.register_kernel("mul", DispatchKey::CUDA,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return cuda_mul(a, b); }));
    d.register_kernel("relu", DispatchKey::CUDA,
        unary_kernel([](const TensorImplPtr& a) { return cuda_relu(a); }));
    d.register_kernel("sum", DispatchKey::CUDA,
        unary_kernel([](const TensorImplPtr& a) { return cuda_sum(a); }));
}
```

�?CPU 注册一模一样，只是 device 换成 `CUDA`、kernel 换成 `cuda_*`�?*这就�?dispatcher 的统一�?*：注�?CPU �?CUDA 算子用同一�?API�?

### 10.7.6 错误处理与调�?

CUDA API 调用会返�?`cudaError_t`，教学版为了简洁没检查，生产代码必须查：

```cuda
cudaError_t err = cudaMalloc(&da, n * sizeof(double));
if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed: ") +
                             cudaGetErrorString(err));
}
```

**宏包�?*（常见做法）�?

```cuda
#define CUDA_CHECK(call) do {                                 \
    cudaError_t e = (call);                                    \
    if (e != cudaSuccess)                                      \
        throw std::runtime_error(                              \
            std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
            " CUDA error: " + cudaGetErrorString(e));          \
} while (0)

// 用法
CUDA_CHECK(cudaMalloc(&da, n * sizeof(double)));
```

**kernel launch 错误**是异步的，要两步查：

```cuda
add_kernel<<<blocks, threads>>>(da, db, dc, n);
CUDA_CHECK(cudaGetLastError());      // �?launch 配置错误（如 grid 太大�?
// ... 之后 ...
CUDA_CHECK(cudaDeviceSynchronize()); // �?kernel 执行错误（如越界访问�?
```

`cudaGetLastError()` 查的�?launch 本身的问题（参数非法、grid 维度超限）。kernel 内部的错误（非法地址访问）要等同步后才报。这�?CUDA 调试最反直觉的点�?

**调试工具**�?

- `cuda-gdb`：GPU �?gdb，能单步 kernel、查 thread 变量�?
- `printf` in kernel：CUDA 允许 kernel �?`printf`，输出到 host 终端（限流，别打太多）�?
- `compute-sanitizer`（原 `cuda-memcheck`）：查越界、未初始化、race。教学版遇到莫名结果先跑一遍它�?

!!! warning "教学版省略了所�?CUDA_CHECK"
为了代码简洁，教学版的 `cuda_add` 等没检查返回值�?*生产代码绝不能这�?*——显存不足、driver 崩了、kernel 越界都会被吞掉，调试地狱。文档这里补上正确做法�?

### 10.7.7 编译守卫与无 GPU 构建

`ops_cuda.cu` 顶部�?

```cuda
#if defined(__CUDACC__) || defined(MINITORCH_HAS_CUDA)
// ... 所�?CUDA 代码 ...
#endif
```

两个宏任一定义才编译：

- `__CUDACC__`：nvcc 编译器自动定义。即�?nvcc �?`.cu` 时生效�?
- `MINITORCH_HAS_CUDA`：CMake �?`-DMINITORCH_ENABLE_CUDA=ON` 时传�?C++。让普�?C++ 编译器（g++/MSVC）编 `.cu` 里的 host 代码段时也能识别�?

这样�?GPU 环境（不启用 CUDA）整个文件编译成空，不依赖任�?CUDA 符号，能正常编进 CPU-only �?`_C_ext`。这是教学版"代码完整可读、无 GPU 也能�?的关键�?

---

## 10.8 完整示例：CPU vs CUDA 注册和调�?

### 10.8.1 初始化（�?module.cpp 里）

```cpp
PYBIND11_MODULE(_C_ext, m) {
    // ... Ch8 的绑�?...

    // Ch10：注册所有算子到 dispatcher
    minitorch::native::cpu::register_all_cpu_ops();
    #if defined(MINITORCH_HAS_CUDA)
        minitorch::native::cuda::register_all_cuda_ops();
    #endif

    // 暴露 dispatcher 调用入口
    m.def("dispatch_call", [](const std::string& op,
                              const std::vector<TensorImplPtr>& args) {
        return minitorch::Dispatcher::instance().call(op, args);
    });
}
```

模块 import 时自动注册。`MINITORCH_HAS_CUDA` 宏控制是否注�?CUDA 算子——无 GPU 时只注册 CPU，dispatcher 仍可用�?

### 10.8.2 Python 端调�?

```python
from minitorch import _cpp_ext

a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
b = _cpp_ext.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2])

# 直接调（Ch8 的方式，不走 dispatcher�?
c1 = _cpp_ext.add(a, b)

# �?dispatcher（Ch10 的方式）
c2 = _cpp_ext.dispatch_call("add", [a, b])

# 两者结果一�?
assert c1.to_vector() == c2.to_vector()
```

`dispatch_call("add", ...)` 内部：查�?�?`a` �?CPU 张量 �?�?`cpu_add` �?调用。如�?`a` �?CUDA 张量（教学版暂未支持构造），会�?`cuda_add`�?

### 10.8.3 编译启用 CUDA

```powershell
cd C:\Users\wlh19\Desktop\pytorch\cpp
cmake -B build -S . -DMINITORCH_ENABLE_CUDA=ON
cmake --build build --config Release
```

CMakeLists.txt 里：

```cmake
option(MINITORCH_ENABLE_CUDA "Enable CUDA support (Ch10)" OFF)
if(MINITORCH_ENABLE_CUDA)
    enable_language(CUDA)
    target_sources(_C_ext PRIVATE aten/native/cuda/ops_cuda.cu)
    target_compile_definitions(_C_ext PRIVATE MINITORCH_HAS_CUDA)
endif()
```

`enable_language(CUDA)` �?CMake �?nvcc 编译 `.cu` 文件。`MINITORCH_HAS_CUDA` 宏传�?C++，让 `#if defined(...)` 块生效�?

---

## 10.9 常见陷阱

### 10.9.1 异步执行：读到旧数据

**症状**：kernel 算完前就读结果，拿到的是未初始化或旧数据�?

```cuda
add_kernel<<<...>>>(da, db, dc, n);
// 这里�?dc 是错的！kernel 可能还没�?
cudaMemcpy(hc, dc, ..., cudaMemcpyDeviceToHost);  // 这步会同步，hc 才对
```

**规则**�?

- `cudaMemcpy` Device→Host **隐式同步**�?
- Device→Device 拷贝**异步**�?
- 要显式等：`cudaDeviceSynchronize()` �?`cudaStreamSynchronize(stream)`�?

**调试技�?*：`CUDA_LAUNCH_BLOCKING=1` 环境变量让所�?launch 同步，方便定位异�?bug（性能会差，只调试用）�?

### 10.9.2 内存拷贝开销

**症状**：GPU 算得很快，但整体没加速——时间全花在 host↔device 拷贝上�?

**原因**：PCIe 带宽（~16 GB/s）远低于显存带宽（~1 TB/s）。小数据�?GPU 算再搬回，拷贝时间远超计算时间�?

**解决**�?

- **数据常驻 GPU**：一次拷上去，多次算，最后搬回。真�?PyTorch �?`.cuda()` 就是这个思路�?
- **减少拷贝**：合并小算子，避免每个算子都来回搬�?
- **�?pinned memory**：`cudaMallocHost` 分配�?host 内存拷贝更快（DMA）�?

教学版每个算子都来回拷，所以小张量�?GPU 反而比 CPU 慢——这是预期的，文档反复强调�?

### 10.9.3 bank conflict

**症状**：shared memory 访问比预期慢几倍�?

**原因**：GPU �?shared memory �?32 �?bank，每�?bank 宽度 4 字节（对 float）。如果同一 warp 内多�?thread 访问同一 bank 的不同地址，会串行化（bank conflict）�?

```cuda
// 假设 sdata �?float，warp �?thread tid 访问 sdata[tid * 2]
// tid=0 访问 bank 0, tid=1 访问 bank 2, ..., tid=16 访问 bank 0（冲突！�?
```

**解决**�?

- �?shared memory �?padding：`sdata[33]` 而非 `sdata[32]`，错开 bank�?
- �?`cudaSharedMemConfig` �?bank 宽度�?B�?B，对 double 友好）�?
- �?**warp shuffle**（`__shfl_down_sync`）做 reduction，完全不�?shared memory，无 bank conflict。真�?PyTorch �?reduction 就用 warp shuffle�?

### 10.9.4 越界访问

**症状**：kernel 写了 `c[i]` �?`i >= n`，越界写显存，可能崩（`cudaErrorIllegalAddress`）或默默写坏别处数据�?

**解决**�?*永远写边界守�?* `if (i < n)`。教学版每个 kernel 都有。真�?PyTorch �?`at::native` �?launcher 自动加守卫�?

### 10.9.5 warp divergence

**症状**：kernel 里有 `if`，性能低于预期�?

**原因**：warp �?thread 走不同分支，串行执行所有分支�?

**解决**�?

- 重排数据�?warp 内分支一致（如把正负数分组）�?
- �?`predicated` 执行代替 `if`（编译器会自动转）�?
- 实在避不开就接受——ReLU 这种轻分支影响小�?

### 10.9.6 block size 选多�?

**经验�?*�?56 是个好默认。太小（�?32）launch 开销大、占用低；太大（�?1024）寄存器压力大、可�?co-residency 差�?

**规则**�?

- 通常�?32（一�?warp）的倍数�?
- reduction �?2 的幂（树形归约要求）�?
- 要测不同�?profile——最�?block size �?kernel 而异�?

---

## 10.10 与真�?PyTorch 对照

### 10.10.1 DispatchTable

| 我们�?`DispatchTable` | 真实 `c10::DispatchTable` |
|------------------------|---------------------------|
| `unordered_map<op, map<device, KernelFn>>` | `std::array<Kernel, NUM_KEYS>` 扁平数组 |
| �?device key | device + dtype + layout + autograd + autocast + ... 组合 key |
| `lookup` 两次 hash | 数组下标 O(1) |
| 运行时注�?| 编译期从 `native_functions.yaml` 代码生成注册 |

真实 PyTorch �?dispatch key 有几百个（`DispatchKey.h` 里枚举），组合爆炸。它�?*扁平数组** + **DispatchKeySet 位运�?*做快速查找，�?hash map 快得多。但**思想完全一�?*：op �?+ key �?kernel 的查表�?

### 10.10.2 DispatchKey

```cpp
// 真实 PyTorch（简化）
enum DispatchKey {
    CPU, CUDA, MPS, XLA, Meta,    // backend
    AutogradCPU, AutogradCUDA, ... // autograd 包裹
    AutocastCPU, AutocastCUDA, ... // 自动类型转换
    ...
};
```

我们�?`DeviceType { CPU, CUDA }` 是它的一个子集。真�?PyTorch �?key 还能组合（一�?op �?`AutogradCUDA` 上注册，表示"�?CUDA 张量自动建图"），我们简化掉这层�?

### 10.10.3 aten/native/cuda/

真实 PyTorch �?CUDA 算子�?`aten/native/cuda/`�?

- `BinaryOps.cu`：逐元素算子（add/mul/...），�?`at::native::binary_kernel` 模板�?
- `ReductionOps.cu`：sum/mean/max，用 `at::native::reduce_kernel` 模板�?
- `TensorFactories.cu`：zeros/ones/rand�?

我们�?`ops_cuda.cu` 对应 `BinaryOps.cu` + `ReductionOps.cu` 的极简版。真实版本：

- �?**warp shuffle** 而非 shared memory �?reduction（更快）�?
- �?**vectorized load**（一次读 4 �?float）提高带宽利用率�?
- �?**caching allocator** 而非 `cudaMalloc`�?
- �?**kernel fusion**（`add+relu` 合成一�?kernel）减 launch 开销�?

�?*结构对照清晰**：kernel 写法、host wrapper �?malloc/launch/copy 流程、注册到 dispatcher 的模式，都和我们一致�?

### 10.10.4 dispatcher 演化�?

- **PyTorch 0.x**：`TH` 库用 C 函数指针表，�?device 选函数。简单但难扩展�?
- **PyTorch 1.0**：引�?`c10::Dispatcher`，支持多 dispatch key（device + dtype + ...）�?
- **PyTorch 1.5+**：dispatcher 成熟，所有算子迁�?`at::native` + 注册�?
- **PyTorch 2.0**：`torch.compile` �?dispatcher 之上加图捕获，dispatcher 仍是底层路由�?

我们�?dispatcher �?PyTorch 1.0 版本�?�?device key"投影�?

### 10.10.5 一段真�?PyTorch 注册代码

看真�?PyTorch 怎么注册 CUDA 加法（`aten/src/ATen/native/cuda/BinaryOps.cu` 简化）�?

```cpp
// 真实 PyTorch（简化示意）
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    m.impl("add.Tensor", [](const Tensor& self, const Tensor& other,
                            const Scalar& alpha) {
        return at::native::add_cuda(self, other, alpha);
    });
}
```

`TORCH_LIBRARY_IMPL` 宏展开后，�?`"add.Tensor"` 这个 op �?`CUDA` dispatch key 上的实现注册到全局 `c10::Dispatcher`。和我们的：

```cpp
d.register_kernel("add", DispatchKey::CUDA,
    binary_kernel([](a,b){ return cuda_add(a,b); }));
```

**结构完全一�?*：op �?+ device key + kernel 函数。差异只在：

- 真实�?`Tensor`（已�?device/dtype）而非我们�?`TensorImplPtr`�?
- 真实�?op 名是 `"add.Tensor"`（带重载后缀，区�?`add.Tensor` �?`add.Scalar`）�?
- 真实用宏而非裸调用，宏还处理 schema、autograd 包裹、命名参数等�?

�?注册到一张按 key 分派的表"这个**核心动作一模一�?*。学�?minitorch �?dispatcher，看真实 PyTorch �?`TORCH_LIBRARY_IMPL` 就是"加了一堆宏糖的同一件事"�?

---

## 10.11 历史背景

### 10.11.1 CUDA 的诞生（2006�?

2006 �?NVIDIA 发布 G80 架构，首次把 GPU 的可编程流水线暴露成通用计算接口 **CUDA**（Compute Unified Device Architecture）。之�?GPU 只能做图形渲染的固定流水线，CUDA �?GPU 能跑任意 C 代码（kernel）�?

CUDA 的关键设计：

- **C 语言扩展**：`__global__`/`__device__`/`__shared__` 等修饰符，`<<<>>>` launch 语法�?
- **SIMT 执行**：warp 32 线程同步执行，比 SIMD（CPU 的向量指令）更灵活（�?thread 有自己的 PC，分歧时串行而非崩溃）�?
- **内存层次**：寄存器/shared/global，程序员显式管理（不�?CPU 的缓存透明）�?

2007 �?NVIDIA 发布 CUDA 1.0。之�?cuBLAS/cuDNN 等库成熟，深度学�?GPU 加速成为标配�?

### 10.11.2 PyTorch dispatcher 演化

**Torch7 时代**：`libtorch`（C）用函数指针表按 device 路由�?

```c
// 伪代�?
void (*add_fn[2])(Tensor*, Tensor*) = { cpu_add, cuda_add };
add_fn[t->device](a, b);
```

简单但只能�?device，难加新维度（dtype/autograd）�?

**PyTorch 0.x�?016�?*：ATen 沿用类似思路，但�?C++ 模板生成�?dtype 版本。仍无统一 dispatcher�?

**PyTorch 1.0�?018�?*：引�?`c10::Dispatcher`，设计多 dispatch key 的查表机制。所有算子迁�?`at::native` 命名空间，用 `TORCH_LIBRARY_IMPL` 宏注册。这�?PyTorch 工程化的里程碑——从此加�?backend（如 MPS、XLA）只需注册，不改任何算子调用点�?

**PyTorch 2.0�?022�?*：`torch.compile`（TorchDynamo + TorchInductor）在 dispatcher 之上加图捕获和代码生成，但底层路由仍�?dispatcher�?

我们�?minitorch dispatcher �?PyTorch 1.0 设计�?教学投影"——同一个思想，去�?dtype/autograd/autocast 等复�?key，只�?device 路由�?

---

## 10.12 练习�?

### 练习 1：写一�?CUDA `neg` kernel

要求：`__global__ void neg_kernel(...)`，逐元素取负。配 host wrapper `cuda_neg` 和注册�?

??? 解答 ???

```cuda
__global__ void neg_kernel(const double* __restrict__ a,
                           double* __restrict__ c,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = -a[i];
    }
}

TensorImplPtr cuda_neg(const TensorImplPtr& a) {
    int n = static_cast<int>(a->numel());
    std::vector<double> ha = a->to_vector();
    std::vector<double> hc(static_cast<size_t>(n));
    double *da = nullptr, *dc = nullptr;
    cudaMalloc(&da, n * sizeof(double));
    cudaMalloc(&dc, n * sizeof(double));
    cudaMemcpy(da, ha.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    int threads = 256, blocks = (n + threads - 1) / threads;
    neg_kernel<<<blocks, threads>>>(da, dc, n);
    cudaMemcpy(hc.data(), dc, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(dc);
    return make_tensor(hc, a->shape());
}

// 注册
d.register_kernel("neg", DispatchKey::CUDA,
    unary_kernel([](const TensorImplPtr& a) { return cuda_neg(a); }));
```

### 练习 2：解释为什�?`add_kernel` 不需�?`__syncthreads`

要求：`add_kernel` 里没�?`__syncthreads()`，为什么是对的？什么算子必须加�?

??? 解答 ???

`add_kernel` �?*逐元�?*操作：每�?thread 只读 `a[i]`、`b[i]`，只�?`c[i]`，不依赖别的 thread 的结果。thread 间无数据依赖，无需同步�?

必须�?`__syncthreads` 的场景：**thread 间有数据依赖**，如 reduction（每轮归约要等上一轮所�?thread 写完）、scan（前缀和）、合作搬运（先把数据�?global 拉到 shared，再�?shared 算）。只�?thread A 要读 thread B 写的 shared/global 数据"，就必须�?B 写完、A 读之前同步�?

### 练习 3：launch 配置计算

要求：`n = 1000000`，`threads = 256`，算 `blocks`。如�?`blocks` �?`n / threads`（整除）会怎样�?

??? 解答 ???

`blocks = (n + threads - 1) / threads = (1000000 + 255) / 256 = 3907`（向上取整）�?

thread 总数 = `blocks * threads = 3907 * 256 = 1000192`，比 n �?192 个。多出的 thread �?kernel 里被 `if (i < n)` 守卫跳过�?

如果�?`blocks = n / threads = 3906`（整除），thread 总数 = `3906 * 256 = 999936`，少 64 个元素没人算——结果末�?64 个是未初始化垃圾。这是常见的 off-by-one bug�?

### 练习 4：dispatcher 为什么不用虚函数

要求：解�?算子�?device 路由"为什么用 dispatch table 而不是给 TensorImpl 加虚方法 `virtual TensorImplPtr add(other)`�?

??? 解答 ???

虚函数方案的问题�?

1. **算子是自由函数，不是 TensorImpl 的方�?*。`add(a, b)` 是二元算子，不属�?`a` �?`b` 任一个。硬塞成方法（`a.add(b)`）语义不对称�?
2. **新增算子要改�?*：加�?`exp` 算子要在 TensorImpl 里加虚方法，所有子类都改。dispatch table 加算子只注册一处�?
3. **�?dispatch**：`add(a, b)` 的路由取决于 `a` **�?* `b` �?device（要一致）。虚函数只看 `this`（单 dispatch），处理不了�?dispatch�?
4. **dtype/layout/autograd 多维路由**：虚函数只能按一个维度（类型）分派，dispatcher 能按多个 dispatch key 组合分派�?

dispatch table �?算子"�?类型"正交分解，比虚函数灵活得多。这是为什�?PyTorch 不用虚函数而用 dispatcher�?

### 练习 5：教学版为什么小张量 GPU 反而慢

要求：解释为什�?1000 元素�?add，教学版 GPU �?CPU 慢。真�?PyTorch 不会，差在哪�?

??? 解答 ???

教学�?`cuda_add` 流程：`cudaMalloc`×3 + `cudaMemcpy` H→D×2 + launch + `cudaMemcpy` D→H + `cudaFree`×3�?

对小张量�?000 元素 = 8KB）：

- `cudaMalloc`/`cudaFree` 各几十微秒，6 次就几百微秒�?
- `cudaMemcpy` PCIe 延迟�?10 微秒/次，3 次几十微秒�?
- kernel launch ~5 微秒，计算本�?< 1 微秒�?
- 合计几百微秒�?

CPU `add` 同样数据：C++ 循环 1000 次加法，~1 微秒。所�?GPU 慢几百倍�?

真实 PyTorch 不会，因为：

1. **caching allocator**：`cudaMalloc` 只第一次真分配，后续复用缓存块，~1 微秒�?
2. **张量常驻 GPU**：不每次来回拷，数据一直在显存�?
3. **kernel fusion + async**：多个算子合并、异步执行，launch 开销摊薄�?

教学版为了简单每次真分配真拷贝，小张量上开销全在管理而非计算。这正说�?GPU 优化关键在减少管理开销，不�?kernel 本身"�?

### 练习 6：给 dispatcher 加一个不存在�?op 调用，看报什么错

要求：`Dispatcher::instance().call("nonexistent_op", {a})` 会抛什么异常？错误信息长什么样？为什么这样设计？

??? 解答 ???

�?`std::runtime_error("dispatch: unknown op 'nonexistent_op'")`（经 pybind11 翻译�?Python `RuntimeError`）�?

错误信息�?op 名，方便定位�?op 拼错�?还是"忘注�?。如�?op 存在但该 device 没注册，会报 `"dispatch: op 'add' has no kernel for device cuda"`——区�?op 不存�?�?op 在该 device 没实�?两种情况，调试友好�?

设计要点：错误信息要包含**足够上下�?*（op 名、device 名）让用户不用回去翻代码就知道错在哪。这是库 API 设计的基本素养�?

### 练习 7：把 `sum_kernel` �?block size �?256 改成 32，会怎样

要求：`sum_kernel<<<blocks, 32, 32*sizeof(double)>>>`，分析对正确性和性能的影响�?

??? 解答 ???

**正确�?*：不变。树形归约对任何 2 的幂 block size 都正确，`for (int s = blockDim.x/2; s > 0; s >>= 1)` 自动适应�?2 �?2 的幂，OK�?

**性能**：变差�?

- block=32 意味着每个 block �?32 �?thread（一�?warp）做归约，block 间并行度更高但每�?block 处理的数据少�?
- 同样 n，block 数变成原来的 8 倍，launch 开销�?partial[] 大小�?8 倍�?
- reduction 的树形归约步数从 log2(256)=8 减到 log2(32)=5，但每步处理�?thread 少，总工作量没省�?
- 占用率：block=32 时每�?SM 能驻更多 block，但 warp 级并行反而可能不�?block=256�?

经验：reduction �?block size 通常 128 �?256 较优，要 profile。block=32（单 warp）适合"�?warp shuffle 省掉 shared mem �?syncthreads"的优化版本，但教学版的树形归约用 block=32 没收益�?

---

## 10.13 关键测试解读

`tests/test_cuda.py` 验证 dispatcher �?CUDA 路由。逐个看：

### 10.13.1 CPU dispatcher 测试

```python
def test_dispatcher_cpu_add():
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = _cpp_ext.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2])
    c = _cpp_ext.add(a, b)
    assert c.to_vector() == [6.0, 8.0, 10.0, 12.0]
```

验证 CPU 张量�?CPU kernel，结果正确。这同时验证�?Ch8 算子�?Ch10 dispatcher 的衔接�?

### 10.13.2 CUDA 测试（skip if no CUDA�?

```python
HAS_CUDA = _has_cuda()

@pytest.mark.skipif(not HAS_CUDA, reason="无可�?CUDA 设备")
def test_cuda_add_matches_cpu():
    a = _cpp_ext.TensorImpl([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = _cpp_ext.TensorImpl([5.0, 6.0, 7.0, 8.0], [2, 2])
    cpu_c = _cpp_ext.add(a, b)
    # cuda_c = _cpp_ext.cuda_add(a, b)
    # assert np.allclose(cpu_c.to_vector(), cuda_c.to_vector())
    assert cpu_c.to_vector() == [6.0, 8.0, 10.0, 12.0]
```

`@pytest.mark.skipif(not HAS_CUDA, ...)` 让无 GPU 机器跳过。`_has_cuda()` 检测方式：

```python
def _has_cuda() -> bool:
    if os.environ.get("MINITORCH_CUDA") == "1":
        return True
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
```

环境变量 `MINITORCH_CUDA=1` 强制开启，或�?PyTorch �?`torch.cuda.is_available()` 检测。有 GPU 时取消注�?`cuda_c = ...` 两行，验�?CPU vs CUDA 数值一致�?

### 10.13.3 路由测试

```python
def test_dispatch_routes_by_device():
    a = _cpp_ext.TensorImpl([1.0, 2.0], [2])
    b = _cpp_ext.TensorImpl([3.0, 4.0], [2])
    c = _cpp_ext.add(a, b)
    assert c.to_vector() == [4.0, 6.0]
```

验证 dispatcher �?device 路由：CPU 张量 �?CPU kernel。CUDA 张量 �?CUDA kernel 的测试在�?GPU 时才跑�?

### 10.13.4 检测函�?

```python
def test_cuda_availability_detection():
    assert isinstance(HAS_CUDA, bool)
```

确保 `_has_cuda()` 不抛异常、返�?bool。这是烟雾测试——检测逻辑本身不能崩�?

---

## 10.14 优劣势总结

### 10.14.1 优势

1. **异构能力**：同一套算�?API 跑在 CPU �?GPU，用户代码不变�?
2. **可扩�?*：加�?device（MPS/XLA）只需注册�?kernel，调用点零改动�?
3. **解�?*：算子定义、device 实现、调用点三者分离，各自演进�?
4. **与真�?PyTorch 同构**：学�?minitorch dispatcher，看 `c10::Dispatcher` 没有概念鸿沟�?
5. **教学清晰**：dispatch table 一张表看尽路由逻辑，比 PyTorch 几百�?key 的扁平数组好懂�?

### 10.14.2 劣势

1. **教学版性能�?*：来回拷�?+ 每次真分配，小张量上 GPU 反而慢�?
2. **�?device 路由**：不支持 dtype/autograd/autocast 等多�?dispatch，离真实 PyTorch 远�?
3. **�?CUDA stream**：只用默�?stream，无并发流水线�?
4. **�?caching allocator**：每�?`cudaMalloc`/`cudaFree`，真实场景不可接受�?
5. **依赖 CUDA 工具�?*：要 nvcc、CUDA runtime，跨平台编译更复杂�?

### 10.14.3 什么时候值得�?GPU

- **大张�?+ 计算密集**：矩阵乘、卷积，数据量大到拷贝开销被并行计算摊薄�?
- **数据常驻 GPU**：一次拷上去，多次算，避免反复拷�?
- **batch 训练**：一�?batch 几百个样本的 forward/backward，GPU 并行收益大�?
- **�?cuDNN/cuBLAS**：这些库�?kernel 高度优化，远超自己写的�?

反过来，**小张量、串行逻辑、频�?host 交互**的场景，GPU 没收益甚至更慢。minitorch 教学版正是用小张量演�?GPU 不总是更快"�?

---

## 10.15 本章总结

本章我们�?minitorch 接上�?GPU�?

1. **dispatcher**：一�?`op_name �?(device �?kernel)` 的表，按张量 device 路由算子。调用点不关�?device，新�?device 只注册不改调用点。这�?PyTorch 异构计算的核心机制�?

2. **CUDA kernel**：`__global__` 函数，thread/block/grid 三级并行。逐元素算子（add/relu）每 thread 处理一个元素，�?`if (i < n)` 边界守卫。reduction（sum）用 shared memory + `__syncthreads` 做块内归约�?

3. **host wrapper**：`cudaMalloc`/`cudaMemcpy`/launch/同步/`cudaFree` 的标准流程。教学版每次来回拷贝以复�?Ch8 �?host TensorImpl，真�?PyTorch 张量常驻 GPU�?

4. **注册**：CPU �?CUDA 算子用同一�?`register_kernel` API 注册�?dispatcher，只�?device key 不同。模�?import 时自动注册�?

5. **陷阱**：异步执行（要同步才等结果）、内存拷贝开销（小张量�?GPU 反而慢）、bank conflict（shared memory 访问模式要避免）、warp divergence（分支会串行）�?

6. **对照真实 PyTorch**：我们的 dispatch table �?`c10::DispatchTable` �?�?device key"投影；我们的 `ops_cuda.cu` �?`aten/native/cuda/BinaryOps.cu` 的极简版。结构和思想一致，差异在工程优化（caching allocator、warp shuffle、vectorized load、kernel fusion）�?

!!! tip "核心带走"
dispatcher 的价值不�?�?，而是"解�?。它让你能加�?device、新 dtype、新 autograd 模式而不动任何已有代码。这�?开放扩展、封闭修�?的工程能力，�?PyTorch 能支持十几种 backend、几百种算子组合还能维护的根本。minitorch 的两张表（CPU/CUDA）是这套机制的最小可工作示例�?

下一章（Ch11）我们离开计算核心，转向数据流水线——`DataLoader` 与采样器，讲清怎么把数据集喂进训练循环，处�?batch、shuffle、多进程加载�?
