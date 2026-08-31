# A4 profiler 与性能分析

> 本附录对应主线 Ch3/Ch7/Ch9。Ch9 已实现 C++ Autograd Profiler，本附录深入讲解性能分析的方法论，并对照 PyTorch 的 `torch.profiler`。

---

## A4.1 为什么需要 profiler

训练慢的原因可能有很多，**不测量就不知道瓶颈在哪**：

| 瓶颈类型 | 表现 | profiler 能发现 |
|---------|------|----------------|
| 算子慢 | 某个算子耗时占比异常 | Node 执行时间排序 |
| 内存瓶颈 | 频繁 malloc/free 或 OOM | 内存分配统计 |
| 通信瓶颈 | DDP 通信等待 | NCCL event 追踪 |
| 数据加载慢 | GPU 空闲等数据 | DataLoader 时间占比 |
| Python 开销 | 算子调度慢 | Python 层 profiler |

**原则**：先 profiler 定位，再优化。不要盲目优化。

### A4.1.1 优化的黄金法则

```
1. 测量（profiler）→ 找到瓶颈
2. 优化瓶颈（只优化瓶颈）
3. 再测量 → 验证优化效果
4. 重复直到性能达标
```

**常见错误**：
- 不测量就优化（凭直觉，往往优化了非瓶颈）
- 优化后不测量（不知道是否有效）
- 过早优化（代码还没正确就优化，增加复杂度）

### A4.1.2 Amdahl 定律

优化的收益受限于瓶颈占总时间的比例：

```
加速比 = 1 / ((1 - p) + p / s)

其中:
  p = 瓶颈占总时间比例
  s = 瓶颈的加速倍数

例: 瓶颈占 80%，加速 10×
  加速比 = 1 / (0.2 + 0.8/10) = 1 / 0.28 = 3.57×

  瓶颈占 20%，加速 10×
  加速比 = 1 / (0.8 + 0.2/10) = 1 / 0.82 = 1.22×

→ 优化非瓶颈收益很小。必须先找瓶颈。
```

---

## A4.2 minitorch 的 C++ Profiler（Ch9 回顾）

### A4.2.1 设计

```cpp
// autograd/profiler.h
struct ProfileEvent {
    std::string node_name;       // "MulNode", "AddNode"
    double duration_us;          // 执行耗时（微秒）
    size_t memory_before;        // 执行前已分配字节
    size_t memory_after;         // 执行后已分配字节
    int thread_id;               // 线程 ID
};

class Profiler {
    bool enabled_ = false;
    std::vector<ProfileEvent> events_;
    void record(const std::string& name, double us,
                size_t mem_before, size_t mem_after, int tid);
};
```

### A4.2.2 集成到 run_backward

在 `autograd/engine.cpp` 的每个 Node 执行前后记录：

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
size_t mem_before = get_global_allocator().total_allocated();

auto grads = node->apply(grad);    // ← 实际执行

auto t1 = std::chrono::high_resolution_clock::now();
double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
if (profiler.enabled())
    profiler.record(node->name, us, mem_before,
                    get_global_allocator().total_allocated(), 0);
```

### A4.2.3 使用方式

```python
from minitorch import _cpp_ext

_cpp_ext.profiler_start()
x = _cpp_ext.TensorImpl([1.0, 2.0, 3.0], [3], True)
y = _cpp_ext.autograd_sum(_cpp_ext.autograd_mul(x, x), -1, False)
y.backward(_cpp_ext.TensorImpl([1.0], [1], False))
_cpp_ext.profiler_stop()

for event in _cpp_ext.profiler_events():
    name, dur, mem_b, mem_a, tid = event
    print(f"{name:20s} {dur:8.1f} us  mem: {mem_b} → {mem_a}")
```

### A4.2.4 输出示例

```
MulNode             12.3 us  mem: 48 → 96
SumNode              5.1 us  mem: 96 → 144
AccumulateGrad       2.8 us  mem: 144 → 192
```

从输出可以看出：MulNode 是反向传播的瓶颈（耗时最长）。

### A4.2.5 minitorch profiler 的实现细节

```cpp
// autograd/profiler.cpp
class Profiler {
    bool enabled_ = false;
    std::vector<ProfileEvent> events_;
    std::mutex mutex_;  // 多线程安全

public:
    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
        enabled_ = true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = false;
    }

    void record(const std::string& name, double us,
                size_t mem_before, size_t mem_after, int tid) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back({name, us, mem_before, mem_after, tid});
    }

    std::vector<ProfileEvent> events() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_;
    }
};
```

**设计要点**：
- `std::mutex` 保护 `events_`（多线程 Engine 并发记录）
- `enabled_` 标志避免未启用时的开销
- 记录内存差值 `mem_after - mem_before` 可发现分配热点

---

## A4.3 profiler 的三个层次

### A4.3.1 算子级（Autograd Profiler）

记录每个 Node/算子的执行时间。我们 Ch9 实现的就是这一层。

```
MulNode: 12.3 us
AddNode:  3.1 us
SumNode:  5.1 us
```

**回答的问题**：哪个算子慢？

### A4.3.2 内核级（Operator Profiler）

深入到算子内部，记录 kernel launch、内存拷贝等。

```
aten::mul:
  kernel launch:  8.2 us
  memory read:    2.1 us
  memory write:   1.5 us
  python dispatch: 0.5 us
```

**回答的问题**：算子内部时间花在哪？

### A4.3.3 系统级（System Profiler）

记录 CPU/GPU 利用率、内存带宽、PCIe 传输、温度等。

```
GPU Utilization: 73%
Memory Bandwidth: 412 GB/s (peak 900 GB/s)
PCIe RX: 1.2 GB/s
```

**回答的问题**：硬件资源是否打满？

### A4.3.4 三层配合

```
训练慢 → 系统级 profiler: GPU 利用率 40%（GPU 空闲）
       → 算子级 profiler: DataLoader 占 50% 时间
       → 结论: 数据加载是瓶颈，不是计算
       → 优化: 增加 num_workers, pin_memory
```

---

## A4.4 PyTorch 的 profiler

### A4.4.1 torch.autograd.profiler（旧版）

```python
with torch.autograd.profiler.profile() as prof:
    loss = model(x)
    loss.backward()

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

输出：

```
---------------------  ------------  ------------  ------------
Name                  CPU time      CUDA time     Calls
---------------------  ------------  ------------  ------------
aten::mul             12.30us       8.10us        1
aten::add              3.10us       2.00us        1
aten::sum              5.10us       3.50us        1
---------------------  ------------  ------------  ------------
```

### A4.4.2 torch.profiler（新版，推荐）

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(10):
        loss = model(x)
        loss.backward()
        optimizer.step()

# 打印统计
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 导出 Chrome Trace JSON（可在 chrome://tracing 可视化）
prof.export_chrome_trace("trace.json")
```

### A4.4.3 torch.profiler 的 schedule

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,     # 前 2 步不记录（warmup）
        warmup=2,   # 接下来 2 步 warmup（记录但不统计）
        active=6,   # 接下来 6 步正式记录
        repeat=1,   # 重复 1 次
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
) as prof:
    for step, (x, y) in enumerate(dataloader):
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        prof.step()  # ← 必须调用，告诉 profiler 进入下一步
```

**schedule 参数**：
- `wait`：跳过的步数（让缓存热起来）
- `warmup`：记录但不统计的步数（预热 profiler）
- `active`：正式记录的步数
- `repeat`：重复整个 wait/warmup/active 循环

### A4.4.4 Chrome Trace 可视化

`trace.json` 可在 `chrome://tracing` 打开，显示**时间线视图**：

```
时间 →
0ms        1ms        2ms        3ms        4ms
|----------|----------|----------|----------|
[aten::mul          ]                      # CPU thread
    [cuda::mul_kernel]                     # CUDA stream
                     [aten::add]           # CPU thread
                         [cuda::add_kernel]# CUDA stream
```

**关键洞察**：CPU 和 CUDA 的时间线可以对比，发现：
- CPU 算子 dispatch 是否成为瓶颈（CPU 忙但 GPU 空闲）
- kernel launch 延迟（CPU 发出但 GPU 还在等）
- CUDA stream 是否充分并行

### A4.4.5 record_shapes 和 stack trace

```python
with torch.profiler.profile(
    record_shapes=True,       # 记录每个算子的输入 shape
    with_stack=True,          # 记录 Python 调用栈
    with_flops=True,          # 估算 FLOPS
) as prof:
    ...

# 带 shape 的输出
# aten::mul      12.3us  shapes: [[32, 64, 64], [64, 64]]
# → 可以看到是哪个 shape 的 mul 慢

# 带栈的输出
# aten::mul      12.3us  stack: forward (model.py:42) → layer1 (model.py:15)
# → 可以定位到是哪行代码调用的
```

---

## A4.5 训练瓶颈定位方法论

### A4.5.1 第一步：整体时间分解

```python
with torch.profiler.profile() as prof:
    # 数据加载
    for x, y in dataloader:        # ← DataLoader 时间
        # 前向
        loss = model(x)             # ← 前向时间
        # 反向
        loss.backward()             # ← 反向时间
        # 优化器
        optimizer.step()            # ← 优化器时间
```

典型分解（单卡训练）：

| 阶段 | 占比 | 常见瓶颈 |
|------|------|---------|
| DataLoader | 10-30% | IO 慢、num_workers 太少 |
| 前向 | 20-30% | 算子慢、未融合 |
| 反向 | 40-60% | 通常是最大块 |
| 优化器 | 1-5% | 很少是瓶颈 |

### A4.5.2 第二步：算子级分析

```python
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

看 top-10 耗时算子。常见发现：

| 发现 | 优化方法 |
|------|---------|
| matmul 占 60% | 正常，无法优化（换更大矩阵 batch） |
| elementwise op 占 20% | 算子融合（torch.compile） |
| copy/cast 占 10% | 减少 .cuda()/.cpu() 调用 |
| softmax 占 5% | 用 fused softmax（flash attention） |
| dropout 占 5% | 用 fused dropout |

### A4.5.3 第三步：内存分析

```python
with torch.profiler.profile(profile_memory=True) as prof:
    ...

print(prof.key_averages().table(sort_by="self_memory_usage"))
```

常见内存问题：

| 问题 | 表现 | 解决 |
|------|------|------|
| 激活太大 | OOM | 梯度检查点（Ch9） |
| 频繁分配 | malloc 占比高 | 预分配 buffer |
| 内存碎片 | 峰值高但平均低 | `torch.cuda.empty_cache()` |
| 梯度累积 | 梯度未清零 | `optimizer.zero_grad()` |

### A4.5.4 第四步：GPU 利用率分析

```python
# 方法 1: nvidia-smi 监控
# watch -n 0.1 nvidia-smi
# → 看 GPU 利用率是否打满

# 方法 2: profiler 的 CUDA 时间
# 如果 CPU 时间 >> CUDA 时间 → CPU 瓶颈（dispatch 慢）
# 如果 CUDA 时间 >> CPU 时间 → GPU 瓶颈（计算密集）
```

**GPU 利用率低的原因**：

| 原因 | 表现 | 解决 |
|------|------|------|
| DataLoader 慢 | GPU 空闲等数据 | 增加 num_workers |
| CPU dispatch 慢 | CPU 忙 GPU 闲 | torch.compile |
| 频繁同步 | .item()/.cpu() 阻塞 | 减少同步 |
| batch 太小 | GPU 未打满 | 增大 batch |
| 算子太小 | kernel launch 开销大 | 算子融合 |

---

## A4.6 GPU profiler 工具

### A4.6.1 Nsight Systems

NVIDIA 的系统级 profiler，最强大：

```bash
nsys profile -t cuda,nvtx python train.py
```

生成 `.nsys-rep` 文件，用 Nsight Systems GUI 打开，显示：
- CUDA kernel 时间线
- CPU-GPU 同步点
- PCIe 传输
- NVLink 利用率

**关键视图**：
```
Timeline:
  CPU Thread 0: [forward] [backward] [optimizer]
  CUDA Stream 0:     [kernel1] [kernel2] [kernel3]
  CUDA Stream 1:               [copy_h2d]         [copy_d2h]
  PCIe:             [---upload weights---]

→ 可以看到 CPU/GPU 并行度、同步点、数据传输
```

### A4.6.2 Nsight Compute

NVIDIA 的 kernel 级 profiler：

```bash
ncu --set full python train.py
```

分析单个 kernel 的：
- 寄存器使用
- shared memory 利用率
- warp stall 原因
- 计算强度（arithmetic intensity）

**warp stall 原因**：

| 原因 | 含义 | 优化 |
|------|------|------|
| Memory Throttle | 等内存 | 改善访存模式（coalesced access） |
| Compute Throttle | 计算单元满 | 无法优化（已打满） |
| Not Selected | warp 调度未选中 | 增加 warp 数 |
| Wait | 等依赖 | 减少数据依赖 |

### A4.6.3 NVTX 标注

用 NVTX 在代码中标记区间，profiler 中可视化：

```python
import torch.cuda.nvtx as nvtx

for step, (x, y) in enumerate(dataloader):
    nvtx.range_push("forward")
    loss = model(x)
    nvtx.range_pop()

    nvtx.range_push("backward")
    loss.backward()
    nvtx.range_pop()

# Nsight Systems 中显示:
# [forward] [backward]  ← 自定义标记
#   [kernel1] [kernel2] ← CUDA kernel
```

### A4.6.4 CUDA Event 计时

比 profiler 更轻量的 GPU 计时方式：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
output = model(x)
end.record()

torch.cuda.synchronize()  # 等待 GPU 完成
elapsed = start.elapsed_time(end)  # 毫秒
print(f"GPU 时间: {elapsed:.2f} ms")
```

**注意**：CUDA event 是异步的——`record()` 立即返回，`elapsed_time` 需要 `synchronize()` 后才准确。

### A4.6.5 CUDA Stream 并行

profiler 可以看到不同 stream 的并行情况：

```python
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

with torch.cuda.stream(s1):
    a = heavy_compute_1(x)  # 在 stream 1
with torch.cuda.stream(s2):
    b = heavy_compute_2(y)  # 在 stream 2（并行）

# profiler 时间线:
# Stream 1: [heavy_compute_1]
# Stream 2: [heavy_compute_2]    ← 并行执行
```

---

## A4.7 常见性能反模式

### A4.7.1 CPU-GPU 频繁同步

```python
# 反模式：每步 .item() 触发同步
for x, y in dataloader:
    loss = model(x)
    if loss.item() < threshold:  # ← GPU→CPU 同步！
        break
```

**问题**：`.item()` 强制等 GPU 算完，CPU 空闲。
**解决**：减少同步频率，或用 `loss.detach()` 延迟同步。

**触发同步的操作**：

| 操作 | 同步？ | 说明 |
|------|--------|------|
| `.item()` | 是 | 取标量值到 CPU |
| `.cpu()` | 是 | GPU→CPU 拷贝 |
| `print(tensor)` | 是 | 需要值才能打印 |
| `tensor < 0` (GPU) | 否 | 仍在 GPU |
| `tensor.numpy()` | 是 | 需 CPU 数据 |
| `if tensor > 0:` | 是 | Python 需要标量 |

### A4.7.2 未融合的 elementwise

```python
# 反模式：4 个独立 kernel
a = x * 2.0       # kernel 1: mul
b = a + 1.0       # kernel 2: add
c = b.relu()      # kernel 3: relu
d = c * y         # kernel 4: mul

# 优化：torch.compile 自动融合成 1 个 kernel
d = torch.compile(lambda x, y: ((x * 2.0 + 1.0).relu() * y))(x, y)
```

**为什么融合快**：
- 减少 kernel launch 开销（4 次 → 1 次）
- 减少中间变量的内存读写（`a`, `b`, `c` 在寄存器中，不写回显存）
- 更好的 cache 局部性

### A4.7.3 DataLoader 瓶颈

```python
# 反模式：num_workers=0
dataloader = DataLoader(dataset, batch_size=32, num_workers=0)

# 优化：多进程预加载
dataloader = DataLoader(dataset, batch_size=32, num_workers=4,
                         pin_memory=True, persistent_workers=True)
```

**DataLoader 优化参数**：

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `num_workers` | 数据加载进程数 | 4-8（CPU 核数的一半） |
| `pin_memory` | 锁页内存（加速 H2D 拷贝） | True（GPU 训练） |
| `persistent_workers` | 不每 epoch 重建进程 | True |
| `prefetch_factor` | 每个 worker 预加载 batch 数 | 2（默认） |

### A4.7.4 不必要的梯度计算

```python
# 反模式：推理时未用 torch.no_grad()
model.eval()
for x in test_data:
    output = model(x)  # ← 仍然建图！浪费内存和时间

# 优化
with torch.no_grad():  # ← 不建图，省内存+加速
    for x in test_data:
        output = model(x)
```

### A4.7.5 未使用 mixed precision

```python
# 反模式：纯 fp32 训练
loss = model(x)  # fp32
loss.backward()

# 优化：fp16 混合精度（Ch12）
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    loss = model(x)  # 自动用 fp16
scaler.scale(loss).backward()
scaler.step(optimizer)
```

---

## A4.8 内存 profiling

### A4.8.1 显存监控

```python
# 当前显存使用
print(f"已分配: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"已保留: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"峰值:   {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# 显存快照
snapshot = torch.cuda.memory_snapshot()
# → 返回所有分配的详细信息（大小、调用栈、时间）
```

### A4.8.2 显存调优

```python
# 1. 梯度检查点（省显存，增计算）
from torch.utils.checkpoint import checkpoint
model.layer1 = checkpoint(model.layer1, x)  # 不存中间激活

# 2. 清空缓存（解决碎片化）
torch.cuda.empty_cache()  # 释放未使用的显存给系统

# 3. 设置显存分配策略
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# → 减少大块分配的碎片化

# 4. 显存统计
with torch.profiler.profile(profile_memory=True) as prof:
    model(x)
print(prof.key_averages().table(sort_by="self_memory_usage"))
```

### A4.8.3 内存泄漏检测

```python
# 训练循环中显存持续增长 → 泄漏
for step in range(100):
    loss = model(x)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        print(f"step {step}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
# 如果显存持续增长，有泄漏

# 常见泄漏原因:
# 1. 把 loss 存到 list 里（保留了计算图）
# 2. 未 detach 的中间结果被引用
# 3. hook 未移除
```

---

## A4.9 分布式 profiling

### A4.9.1 DDP 通信分析

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
) as prof:
    for x, y in dataloader:
        loss = model(x, y)     # DDP 在 backward 时 AllReduce
        loss.backward()
        optimizer.step()
        prof.step()
```

**DDP 的通信开销**：

```
反向传播时:
  每个参数梯度算完后 → AllReduce (NCCL)
  → 通信时间可能在 profiler 中显示为 "nccl:all_reduce"

常见问题:
  通信占比高 → 梯度分桶太小（增大 bucket_cap_mb）
  通信和计算未重叠 → 开启 DDP find_unused_parameters=False
  某些 rank 慢 → 检查数据均匀性
```

### A4.9.2 NCCL 调试

```python
import os
os.environ['NCCL_DEBUG'] = 'INFO'      # NCCL 日志
os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # 指定网络接口

# 运行后看 NCCL 日志:
# NCCL INFO Channel 00 : 0[xx] -> 1[xx] via P2P/IPC
# → 可以看到通信路径、拓扑、带宽
```

---

## A4.10 自定义 profiler

### A4.10.1 用 minitorch 的 hook 写 profiler

```python
# 基于 minitorch 的 autograd hook 实现自定义 profiler
from minitorch import Tensor

class CustomProfiler:
    def __init__(self):
        self.events = []

    def __enter__(self):
        # 注册全局 hook
        Tensor.register_global_backward_hook(self._hook)
        return self

    def __exit__(self, *args):
        Tensor.clear_global_backward_hook()

    def _hook(self, node_name, grad_input, grad_output):
        import time
        t0 = time.perf_counter()
        # hook 在 grad 计算后调用
        self.events.append({
            'name': node_name,
            'time': t0,
            'grad_shape': grad_input.shape,
        })

# 使用
with CustomProfiler() as prof:
    loss.backward()

for event in prof.events:
    print(f"{event['name']}: grad_shape={event['grad_shape']}")
```

### A4.10.2 用 Python context manager

```python
import time
from contextlib import contextmanager

@contextmanager
def time_block(name, stats):
    t0 = time.perf_counter()
    yield
    stats[name] = stats.get(name, 0) + time.perf_counter() - t0

stats = {}
for x, y in dataloader:
    with time_block("forward", stats):
        loss = model(x)
    with time_block("backward", stats):
        loss.backward()
    with time_block("optimizer", stats):
        optimizer.step()

print(stats)
# {'forward': 12.3, 'backward': 23.4, 'optimizer': 0.5}
```

---

## A4.11 与真实 PyTorch 对照

| 概念 | minitorch | PyTorch | 文件 |
|------|-----------|---------|------|
| Profiler | `Profiler` 类 | `torch.profiler.profile` | `torch/profiler/` |
| ProfileEvent | `ProfileEvent` | `Event` | `torch/profiler/event.py` |
| 算子级 | Ch9 实现 | `torch.autograd.profiler` | `torch/autograd/profiler.py` |
| Chrome Trace | 未实现 | `export_chrome_trace` | `torch/profiler/` |
| 内存统计 | `Allocator` 统计 | `profile_memory=True` | `c10::Allocator` |
| CUDA event | 未实现 | `ProfilerActivity.CUDA` | `torch/csrc/profiler/` |
| schedule | 未实现 | `torch.profiler.schedule` | `torch/profiler/profiler.py` |
| NVTX | 未实现 | `torch.cuda.nvtx` | `torch/csrc/cuda/Module.cpp` |
| TensorBoard | 未实现 | `tensorboard_trace_handler` | `torch/profiler/` |

---

## A4.12 优劣势总结

| minitorch profiler 优势 | minitorch profiler 劣势 |
|------------------------|------------------------|
| 轻量，直接集成在 Engine 中 | 仅算子级，无 kernel 级 |
| 同时记录内存分配 | 不支持 CUDA event |
| API 简单（start/stop/events） | 无可视化（需手动打印） |
| 多线程安全 | 无 schedule（全程记录） |

| PyTorch profiler 优势 | PyTorch profiler 劣势 |
|----------------------|----------------------|
| CPU + CUDA 全栈 | API 复杂 |
| Chrome Trace 可视化 | 开销较大 |
| 支持 DDP 通信追踪 | 需要理解大量选项 |
| TensorBoard 集成 | schedule 配置繁琐 |

---

## A4.13 性能分析案例

### A4.13.1 案例 1: 训练速度慢，GPU 利用率 40%

```
症状: 训练 100 epoch 需要 10 小时，GPU 利用率 40%
诊断:
  1. nvidia-smi → GPU 40%，CPU 90%
  2. profiler → DataLoader 占 50% CPU 时间
  3. 原因: num_workers=0，数据加载在主进程

优化:
  DataLoader(num_workers=8, pin_memory=True, persistent_workers=True)
  → GPU 利用率 85%，训练时间 5 小时
```

### A4.13.2 案例 2: OOM 在第 50 步

```
症状: 训练正常 49 步，第 50 步 OOM
诊断:
  1. memory_allocated 逐步增长
  2. profiler memory → 某个 list 不断追加 loss
  3. 原因: losses.append(loss) 保留了计算图

优化:
  losses.append(loss.item())  # 只存标量，不存图
  → 显存稳定
```

### A4.13.3 案例 3: DDP 通信占 30%

```
症状: 4 卡 DDP，通信占 30% 时间
诊断:
  1. profiler → nccl:all_reduce 占 30%
  2. 大量小梯度 AllReduce（每个参数一次）

优化:
  1. 增大 bucket: DDP(model, bucket_cap_mb=25)
     → 多个小梯度合并成一次大 AllReduce
  2. 开启 overlap: DDP(model, gradient_as_bucket_view=True)
     → 通信和计算重叠
  → 通信降到 15%
```

### A4.13.4 案例 4: 反向比前向慢 5×

```
症状: 前向 0.1s，反向 0.5s
诊断:
  1. profiler → 某个自定义算子的 backward 极慢
  2. 原因: backward 用了 Python 循环，未向量化

优化:
  # 反模式
  def backward(grad):
      result = torch.zeros_like(grad)
      for i in range(grad.shape[0]):
          result[i] = grad[i] * weight[i]  # Python 循环
      return result

  # 优化: 向量化
  def backward(grad):
      return grad * weight  # 一次算子调用
  → 反向 0.12s
```

---

## A4.14 小结

profiler 的核心价值：**用数据驱动优化决策**。

三个层次：
1. **算子级**（我们实现了）：哪个 Node 慢？
2. **内核级**（PyTorch torch.profiler）：算子内部时间花在哪？
3. **系统级**（Nsight Systems）：硬件资源是否打满？

优化流程：profiler 定位瓶颈 → 针对性优化 → 再 profiler 验证。**不要盲目优化**。

**关键工程实践**：
- 先测量再优化，先测量再验证（Amdahl 定律）
- GPU 利用率低 → 查 DataLoader / 同步 / batch size
- 显存 OOM → 梯度检查点 / 混合精度 / 清空缓存
- 通信慢 → 增大 bucket / 优化拓扑 / overlap
- 算子慢 → torch.compile 融合 / 换 fused kernel
- 用 NVTX 标记代码区间，profiler 中定位
