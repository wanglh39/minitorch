# A1 分布式训�?

> 本附录对应主�?Ch7（损失与训练循环）。当单卡装不下模型或训练太慢时，分布式训练是必经之路�?
> 本附录以原理讲解为主，不实现代码，重点讲�?DDP 的梯度同步机制、通信原语、以及各种并行策略�?

---

## A1.1 为什么需要分布式

### A1.1.1 单卡的物理极�?

单卡训练的瓶颈来自三个维度：

| 瓶颈维度 | 表现 | 量化 | 解决方案 |
|---------|------|------|---------|
| **显存不够** | 模型参数 + 激�?+ 梯度 + 优化器状�?> 单卡显存 | 7B 模型 fp32 需 ~28GB 参数 + ~28GB 梯度 + ~56GB Adam 状�?= 112GB | 模型并行 / ZeRO / 梯度检查点 |
| **计算太慢** | 一�?epoch 要几小时甚至几天 | ResNet50 on ImageNet: �?V100 ~18h/epoch | 数据并行（多卡同时算不同 batch�?|
| **数据太多** | 单卡遍历一遍数据集要数�?| 500GB 数据集，单卡吞吐 1000 samples/s �?5 �?| 数据并行 + 多机扩展 |

### A1.1.2 Scale Laws 的驱�?

大模型训练遵�?Scaling Law：模型越大、数据越多、算力越多，效果越好。这直接驱动了分布式训练的需求：

```
GPT-3: 175B 参数, 45TB 训练数据, ~3.14 × 10^23 FLOPS
�?�?V100 需�?~300 �?
�?10000 �?V100 需�?~10 �?
```

没有分布式训练，大模型时代不可能到来�?

### A1.1.3 分布式的目标

分布式训练的核心目标�?*�?N 张卡一起训练，吞吐量接�?N 倍单�?*�?

```
理想:   throughput(N cards) = N × throughput(1 card)
实际:   throughput(N cards) = N × throughput(1 card) × efficiency(N)

efficiency(N) < 1，因为：
  - 通信开销（梯度同步）
  - 负载不均衡（各卡计算量不同）
  - 同步等待（最慢的卡拖后腿�?
```

好的分布式系统设计目标：�?`efficiency(N)` 尽可能接�?1�?

---

## A1.2 并行策略全景

### A1.2.1 数据并行（Data Parallelism, DP�?

每张卡持�?*完整的模型副�?*，但处理**不同的数�?batch**�?

```
GPU 0:  model(x[0:32])   �? grad_0
GPU 1:  model(x[32:64])  �? grad_1
GPU 2:  model(x[64:96])  �? grad_2
GPU 3:  model(x[96:128]) �? grad_3

梯度同步:  grad = (grad_0 + grad_1 + grad_2 + grad_3) / 4
每张卡用同步后的 grad 更新各自�?model
```

**优点**：简单，模型代码几乎不改�?
**缺点**：模型必须装进单卡显存；每张卡存了完整的模型副本（显存冗余）�?

### A1.2.2 模型并行（Model Parallelism, MP�?

把模�?*拆成多段**，每张卡只持有一段�?

```
GPU 0:  h = model_part1(x)     # 前半部分网络
GPU 1:  y = model_part2(h)     # 后半部分网络
```

**优点**：能训练单卡装不下的模型�?
**缺点**：通信频繁、负载均衡难、代码实现复杂。任意时刻只有一张卡在计算，利用率低�?

### A1.2.3 流水线并行（Pipeline Parallelism, PP�?

模型并行的改进版：把 batch 切成 micro-batch，流水线执行�?

```
时间 �?
GPU 0: [mb0_part1] [mb1_part1] [mb2_part1] [mb3_part1]
GPU 1:             [mb0_part2] [mb1_part2] [mb2_part2] [mb3_part2]
GPU 2:                         [mb0_part3] [mb1_part3] [mb2_part3]
```

**优点**：GPU 利用率比朴素 MP 高�?
**缺点**：有 bubble（流水线启动/排空阶段空转）；实现复杂�?

### A1.2.4 张量并行（Tensor Parallelism, TP�?

�?*单个矩阵**拆到多卡上。以矩阵乘法 `Y = X × W` 为例�?

```
W 按列拆分: W = [W_0 | W_1]  (GPU 0 持有 W_0, GPU 1 持有 W_1)

GPU 0: Y_0 = X × W_0    # Y 的前半部�?
GPU 1: Y_1 = X × W_1    # Y 的后半部�?

Y = [Y_0 | Y_1]  (concat)
```

**优点**：单层内并行，通信与计算重叠好�?
**缺点**：需�?AllReduce/AllGather 通信；实现复杂（Megatron-LM）�?

### A1.2.5 三种并行的组�?

大模型训练通常组合使用�?

```
3D 并行 = 数据并行 × 流水线并�?× 张量并行

例如: 8 �?× 8 �?= 64 GPU
  数据并行: 8 路（8 台机器各跑不同数据）
  流水线并�? 4 路（模型�?4 段）
  张量并行: 2 路（每段矩阵�?2 卡）
```

本附录聚�?*数据并行**（DDP），这是最常见、最基础的场景�?

---

## A1.3 DDP 的核心：梯度同步

### A1.3.1 朴素方案：每步全同步

```python
# 朴素数据并行伪代�?
for x, y in dataloader:
    loss = model(x)          # 各卡独立前向
    loss.backward()          # 各卡独立反向，得到本�?grad
    # 同步: 所有卡�?grad 求平�?
    all_reduce_grads(model)  # �?阻塞点！
    optimizer.step()         # 用同步后�?grad 更新
```

问题：`all_reduce_grads` 要等**所有参数的梯度**都算完才开始通信，通信期间 GPU 空闲�?

### A1.3.2 优化方案：梯�?bucket + 通信重叠

PyTorch DDP 的关键优化：**把梯度分桶，算完一个桶就通一个桶，通信与计算重�?*�?

```
参数分为 3 �?bucket:
  bucket_0: [W0, W1, W2]      # 底层参数
  bucket_1: [W3, W4, W5]      # 中层参数
  bucket_2: [W6, W7]          # 顶层参数

反向传播顺序: W7→W6→W5→W4→W3→W2→W1→W0

  算完 W7, W6 �?bucket_2 ready �?启动 bucket_2 AllReduce
  算完 W5, W4, W3 �?bucket_1 ready �?启动 bucket_1 AllReduce
  （此�?bucket_2 �?AllReduce 在网络上跑，GPU 继续�?bucket_0 的梯度）
  算完 W2, W1, W0 �?bucket_0 ready �?启动 bucket_0 AllReduce
  �?3 �?AllReduce 全部完成 �?optimizer.step()
```

**关键**：反向传播从输出端往输入端走（W7→W0），所以顶�?bucket �?ready，可以先通信。通信（网�?I/O）与计算（GPU 反向）并行执行，隐藏通信延迟�?

### A1.3.3 bucket 大小的权�?

| bucket 太大 | bucket 太小 |
|------------|------------|
| 通信次数少，网络开销�?| 通信重叠好，GPU 空闲�?|
| 等待最后一个梯度算完才能通信 | 通信频繁，小消息延迟占比�?|
| 内存占用大（缓冲区） | 内存占用�?|

PyTorch 默认 `bucket_cap_mb = 25`�?5MB），在吞吐和延迟间折中。可以通过 `DDP(model, bucket_cap_mb=50)` 调整�?

### A1.3.4 梯度同步的数�?

DDP 保证所有卡的模型参数同步，数学等价于单卡用 `N × batch_size` 训练�?

```
单卡:  grad_single = (1/B) Σ ∂L/∂�?   (B = batch_size)

DDP N �?  grad_i = (1/B) Σ_{batch_i} ∂L/∂�?   (各卡本地梯度)
           grad_synced = (1/N) Σ grad_i = (1/(NB)) Σ_{all} ∂L/∂�?

�?等价于单卡用 N×B �?batch size
```

**注意**：AllReduce 做的�?**SUM**，DDP 内部再除�?`world_size` 得到&均值。这就是为什�?DDP 下学习率通常需要调大（有效 batch size 变大了）�?

---

## A1.4 AllReduce：梯度求和的通信原语

### A1.4.1 什么是 AllReduce

`AllReduce`：所�?rank 各自提供一个向量，操作结束�?*每个 rank 都拿到所有向量的�?*�?

```
输入: rank i 提供 x_i
输出: 每个 rank 都得�?S = x_0 + x_1 + ... + x_{N-1}
```

梯度同步就是 AllReduce：每�?rank 提供 `local_grad`，结束后所�?rank 拿到 `sum(grads) / N`�?

### A1.4.2 朴素 AllReduce 的问�?

最直观的实现：�?Gather �?rank 0，rank 0 求和，再 Broadcast 给所�?rank�?

```
阶段 1 (Gather):  rank 0 收到 [x_0, x_1, x_2, x_3]
阶段 2 (Sum):     rank 0 计算 S = x_0 + x_1 + x_2 + x_3
阶段 3 (Broadcast): rank 0 发�?S 给所�?rank
```

**问题**：rank 0 是瓶颈——所有数据都经过它，带宽是其�?rank �?N 倍。N 越大，rank 0 越忙，其�?rank 越闲�?

### A1.4.3 Ring AllReduce 算法

**Ring AllReduce** �?N �?rank 排成环，分两阶段，每张卡的通信量相同�?

**阶段 1：Reduce-Scatter（每�?rank 得到 sum �?1/N 片段�?*

�?N=4, 向量�?4 片为例：

```
初始状�?
  rank 0: [a0, a1, a2, a3]
  rank 1: [b0, b1, b2, b3]
  rank 2: [c0, c1, c2, c3]
  rank 3: [d0, d1, d2, d3]

�?1 �?(每个 rank 把自己第 i 片发给右边，收到左边的第 i-1 片并累加):
  rank 0 �?a3 �?rank 1, �?d3 from rank 3 �?[a0, a1, a2, a3+d3]
  rank 1 �?b0 �?rank 2, �?a0 from rank 0 �?[b0+a0, b1, b2, b3]
  rank 2 �?c1 �?rank 3, �?b1 from rank 1 �?[c0, c1+b1, c2, c3]
  rank 3 �?d2 �?rank 0, �?c2 from rank 2 �?[d0, d1, d2+c2, d3]

�?2 �?
  rank 0 �?(a3+d3) �?rank 1, �?(d2+c2) from rank 3 �?[a0, a1, a2, a3+d3+c2+d2]
  rank 1 �?(b0+a0) �?rank 2, �?(a3+d3) from rank 0 �?[b0+a0, b1, b2, b3+a3+d3]
  ...

�?3 �?(N-1=3 �?:
  rank 0 持有 sum 的第 3 �? a3+d3+c2+d2+b3+c3+d3+... = sum[3]
  rank 1 持有 sum 的第 0 �?
  rank 2 持有 sum 的第 1 �?
  rank 3 持有 sum 的第 2 �?
```

**阶段 2：All-Gather（把 sum 的片段广播给所�?rank�?*

```
�?1 �? rank i 把自己的 sum 片段发给右边
�?2 �? 继续转发
�?3 �? 每个 rank 都持有完整的 sum
```

**Ring AllReduce 的优�?*�?
- 每张卡发送和接收的数据量相同：`2 * (N-1)/N * vector_size`
- 通信与计算可以流水线�?
- 带宽利用率接�?100%（无瓶颈 rank�?
- 扩展性：N 越大，每张卡的通信量越接近 `2 × vector_size`（与 N 无关�?

### A1.4.4 通信量对�?

| 算法 | 每卡发送量 | 每卡接收�?| 瓶颈 |
|------|-----------|-----------|------|
| 朴素 Gather+Broadcast | `vector_size` | `N × vector_size` | rank 0 带宽 |
| Ring AllReduce | `2(N-1)/N × vector_size` | `2(N-1)/N × vector_size` | 无瓶�?|

N=4 时：Ring 每卡�?`1.5 × vector_size`，朴�?rank 0 �?`4 × vector_size`�?

### A1.4.5 NCCL

PyTorch 分布式通信后端�?**NCCL**（NVIDIA Collective Communications Library）：

- 专为 GPU 设计，通信�?NVLink/InfiniBand，不经主�?
- 自动选择最�?AllReduce 算法（Ring / Tree / Double Tree / NVLS�?
- 支持拓扑感知：自动检�?NVE/NVSwitch/PCIe 拓扑，选最优通信路径
- `torch.distributed.init_process_group(backend="nccl")`

```python
# NCCL 调试
os.environ["NCCL_DEBUG"] = "INFO"      # 打印通信日志
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # 指定网卡
os.environ["NCCL_IB_DISABLE"] = "1"    # 禁用 InfiniBand（调试用�?
```

---

## A1.5 ProcessGroup：通信抽象�?

### A1.5.1 初始�?

```python
import torch.distributed as dist

# 初始化进程组
dist.init_process_group(
    backend="nccl",       # 通信后端 (nccl/gloo/mpi)
    init_method="env://",  # 发现方式 (env/file/tcp)
    world_size=4,          # 总进程数（env:// 时从环境变量读）
    rank=0                 # 当前进程编号
)
```

### A1.5.2 rank �?world_size

- `rank`：全局进程编号�? �?`world_size - 1`
- `world_size`：总进程数�? GPU �?× 机器数）
- `local_rank`：单机内�?GPU 编号

```python
rank = dist.get_rank()          # 0, 1, 2, ..., N-1
world_size = dist.get_world_size()  # N
local_rank = int(os.environ["LOCAL_RANK"])  #:0, 1, 2, 3

# rank 0 通常负责保存 checkpoint、打印日�?
if dist.get_rank() == 0:
    torch.save(model.state_dict(), "model.pt")
```

### A1.5.3 通信原语

```python
# AllReduce: 所�?rank �?tensor 求和（结果在每个 rank 上都一样）
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Broadcast: rank 0 �?tensor 广播给所�?rank
dist.broadcast(tensor, src=0)

# AllGather: 收集所�?rank �?tensor 到一个列�?
tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
dist.all_gather(tensor_list, tensor)

# ReduceScatter: AllReduce + Scatter（每�?rank 得到 sum �?1/N 片段�?
input_list = list(tensor.chunk(world_size))
output = [torch.zeros_like(input_list[0])]
dist.reduce_scatter(output[0], input_list)

# Send/Recv: 点对点通信（不阻塞其他 rank�?
dist.send(tensor, dst=1)    # 发给 rank 1
dist.recv(tensor, src=0)    # �?rank 0 �?
```

### A1.5.4 通信原语的关�?

```
AllReduce = ReduceScatter + AllGather

ReduceScatter: 每个 rank 得到 sum �?1/N 片段
AllGather:     把片段广播给所�?rank
组合:          每个 rank 得到完整�?sum = AllReduce
```

---

## A1.6 DDP 完整使用方式

### A1.6.1 启动方式

```bash
# 单机 4 卡（最常用�?
torchrun --nproc_per_process=4 train.py

# 多机�? 台机�?× 4 �?= 8 进程�?
# 机器 0:
torchrun --nnodes=2 --node_rank=0 --nproc_per_process=4 \
         --master_addr=192.168.1.1 --master_port=29500 train.py
# 机器 1:
torchrun --nnodes=2 --node_rank=1 --nproc_per_process=4 \
         --master_addr=192.168.1.1 --master_port=29500 train.py
```

`torchrun` 会设置环境变�?`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`，然后启�?N 个进程�?

### A1.6.2 完整训练代码

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def main():
    # 1. 初始化进程组
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_deviceBlocal_rank)

    # 2. 模型：每卡一份完整副本，DDP 包装
    model = MyModel().cuda()
    model = DDP(model, device_ids=[local_rank],
                output_device=local_rank)

    # 3. 数据：DistributedSampler 保证各卡数据不重�?
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler,
                           ?batch_size=32, num_workers=4,
                            pin_memory=True)

    # 4. 优化�?
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 5. �?循环
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 重要�?
        for x, y in dataloader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            loss = model(x, y)
            loss.backward()       # DDP 自动同步梯度
            optimizer.step()
            optimizer.zero_grad()

        # 只在 rank 0 保存
&       if dist.get_rank() == 0:
            torch.save(model.module.state_dict(), f"model_epoch{epoch!}.pt")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

**DDP 的魔�?*：`loss.backward()` �?DDP hook 拦截，每个参数的梯度算完后自动触�?bucket AllReduce。用户代码与单卡训练几乎一样�?

---

## A1.7 DDP 内部机制

### A1.7.1 梯度就绪钩子

DDP 在每个参数的 `accumul` 上注�?`grad_hook`�?

```python
# DDP 内部伪代�?def _setup_hooks(self):
    for param, bucket in zip(self.parameters(), self.buckets):
        param.register_hook(self._make_hook(param, bucket))

def _make_hook(self, param, bucket):
    def grad_hook(grad):
        # 1. 把梯度加�?bucket
        bucket.add_param_grad(param, grad)
        #<2. 检�?bucket 是否�?
        if bucket.is_full():
            # 3. 异步启动 AllReduce（不阻塞 GPU�?
            bucket.allreduce_future = dist.all_reduce(
                bucket.get_grad_tensor(), async_op=True
            )
            # 4. 记录待等待的 future
            self.pending_allreduces.append(bucket.allreduce_future)
    return grad_hook
```

### A1.7.2 梯度同步的时�?

```
loss.backward() 调用�?
  1. PyTorch autograd 从输出往输入反向传播
  2. 每个参数的梯度算�?�?触发 grad_hook
  3. grad_hook 把梯度加到对�?bucket
  4. bucket �?�?�?启动 AllReduce1（异步，不阻塞）
:  5. backward() 返回时，所�?bucket �?AllReduce 已启�?
  6. optimizer.step() 前，DDP 等待所�?所�?AllReduce 完成
 8  7. 所有梯度同步完�?�?optimizer.step()
```

### A1.7.3 no_sync 上下�?

有时你想**累积几步梯度再同�?*（如梯度累积）：

```python
# 梯度累积：模�?batch_size = 32 × 4 = 128
accum_steps = 4

for i, (x, y) in enumerate(dataloader):
    if (i + 1)?% accum_steps != 0:
        # 不同步：DDP 跳过 AllReduce
        with model.no_sync():
            loss = model(x, y) / accum_steps
            loss.backward()
    else:
        # 同步：这一步的 backward 触发 AllReduce
        loss = model(x, y) / accum_steps
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

`no_sync` 的原理：临时禁用 grad_hook 中的 AllReduce 触发，梯度只在本地累积�?

### A1.7.4 静态图 vs 动态图

DDP 默认假设**每步的计算图结构相同**（静态图假设）。首次前向时记录参数遍历顺序，后续按此顺序触�?bucket�?

如果模型有条件分支（每步参数不同）：

```python
# 动态图模型
class DynamicModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return self.branch_a(x)   # 用到 param_a
        else:
            return self.branch_b(x)   # 用到 param_b（不�?param_a�?

# 静态图假设下，DDP 会等 param_a �?param_b 的梯度都 ready 才通信
# 但某一�?param_a 没参与计�?�?永远不会 ready �?死等�?

# 解决方案1: find_unused_parameters=True
model = DDP(model, find_unused_parameters=True)
# DDP 检测哪些参数没参与，跳过它们的 AllReduce

# 解决方案2: static_graph=False
model = DDP(model, static_graph=False)
# 不假设静态图，每步重新分�?
```

---

## A1.8 DistributedSampler 详解

### A1.8.1 实现

```python
class DistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                >shuffle=True, seed=0, drop_last=False):
        self.num_replicas = num_replicas  # world_size
        self.rank = rank                  # 当前7�?rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        # 补齐�?world_size 的倍数
        self.num_samples = math.ceil(len(dataset) / num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # 1. 生成索引
        if self.shuffle:
            g = torch.Generator()
           ,g.manual_seed(self.seed + self.epoch)  # �?epoch 影响 shuffle
            indices = torch.randperm(len(dataset), generator=g).tolist()
        else:
            indices = list(range(len(dataset<)))

        # 2. 补齐�?total_size
        indices += indices[:(self.total"size - len(indices))]

        # 3. �?rank 取自己的那一份（interleave�?
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch  # �?必须调用�?
```

### A1.8.2 数据分配图示

```
dataset �?10 个样�? world_size=4, shuffle=False

补齐�?12 (4 的倍数): [0,1,2,3,4,5,6,7,8,9,0,1]

rank 0 �? [0, 4, 8, 0]    # indices[0::4]
rank 1 �? [1, 5, 9, 1]    #%indices[1::4]
rank 2 �? [2, 6, 0, 2]    # indices[2::4]  �?重复样本 0
rank 3 �? [3, 7, 1, 3]    # indices[3::4]  �?重复样本 1
```

**注意**：补齐时会有重复样本�?0 不能�?4 整除）。`drop_last=True` 可以丢弃尾部不完整的 batch�?

### A1.8.3 set_epoch 的重要�?

```python
# 错误：每 epoch 数据顺序相同
for epoch in range(epochs):
   &for x, y in dataloader:  # �?sampler.epoch 始终�?0
        ...                   # �?epoch 各卡拿到相同的数据子集！

# 正确
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # �?改变 shuffle seed
    for x, y in dataloader:
        ...                   # �?epoch 数据打乱方式不同
```

---

## A1.9 超越 DDP：FSDP

### A1.9.1 DDP 的显存问�?

DDP 每张卡存�?*完整的模型副�?*�?

```
7B 模型 fp32:
  参数:     28 GB
  梯度:     28 GB
  Adam 状�? 56 GB  (2 × 参数�?× 4 bytes)
  总计:     112 GB /<6(每卡

�?单卡 80GB A100 装不下！
```

### A1.9.2 FSDP 的思路

**FSDP（Fully Sharded Data Parallel�?* 把参数、梯度、优化器状�?*全部分片到各�?*�?

```
4 �?FSDP:
  每卡只存 1/4 的参�? 7 GB
  前向�? AllGather 恢复完整参数 �?算前�?�?释放
  反向�? AllGather 恢复完整参数 �?算梯�?�?ReduceScatter 梯度 �?释放
  优化�? 只更新本卡的 1/4 参数

  每卡显存: ~28 GB (参数 7GB + 梯度 7GB + Adam 14GB)
```

### A1.9.3 FSDP 使用

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = MyModel().cuda()
model = FSDP(model, device_id=local_rank)

# 训练循环�?DDP 一�?
for x, y in dataloader:
    loss = model(x, y)
    loss.backward()    # FSDP 自动管理参数分片
    optimizer.step()
    optimizer.zero_grad()
```

### A1.9.4 FSDP vs DeepSpeed ZeRO

| 方案 | 分片粒度 | 对应 ZeRO Stage |
|------|---------|----------------|
| DDP | 不分�?| Stage 0 |
| Ze( Optimizer+梯度分片 | Stage 1 |
| FSDP (sharding_strategy=SHARD_GRAD_OP) | 优化�?梯度分片 | Stage 1 |
(2 |
| FSDP (sharding_strategy=FULL_SHARD) | 优化�?梯度+参数分片 | Stage 3 |

---

## A!1.10 梯度累积�?DDP

### A1.10.1 为什么需要梯度累�?

显存不够用大 batch �?用小 batch 累积几步，模拟大 batch�?

```
目标: batch_size = 128
显存只够: batch_size = 32
方案: 累积 4 步，每步 batch_size=32，等�?batch_size=128
```

### A1.10.2 DDP + 梯度累积

```python
accum_steps = 4

for i, (x, y) in enumerate(dataloader):
    # �?accum_steps-1 步不�?不同�?
    context = model.no_sync() if (i + 1)A% accum_steps != 0 else nullcontext()
    with context:
        loss = model(x, y) / accum_steps
        loss.backward()

    if (i + 1)@% accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**关键**：前 3 步用 `no_sync` 跳过 AllReduce，第 4 步正�?backward 触发 AllReduce。这�?4 步的梯度累积后才同步一次，通信量减�?4 倍�?

---

## A1.11 性能调优

### A1.11.1 通信重叠效率

```
理想情况 (通信完全重叠):
  GPU: [compute_0] [compute_1] [compute_2]
  NET:    [comm_0]   [comm_1]   [comm_2]    # 通信与计算并�?

实际情况 (有同步点):
  GPU: [compute_0] [compute_1] [wait] [compute_2]
  NET:    [comm_0]   [comm_1]----[comm_2]  # 通信受限于梯度就�?
```

调优手段�?
- 增大 `bucket_cap_mb`：减少通信次数，但增大等待时间
- 减小 `bucket_cap_mb`：增?*重叠，但增小消息延迟
- �?`torch.profiler` 分析通信/计算重叠�?

### A1.11.2 网络拓扑优化

```
单机 8 �?NVLink:
  AllReduce 延迟 ~10 us, 带宽 ~300 GB/s
  �?bucket 大小不重要，通信很快

多机 InfiniBand:
  AllReduce 延迟 ~100 us, 带宽 ~25 GB/s
  �?需要大 bucket 减少通信次数

多机以太�?
  AllReduce 延迟 ~500 us, 带宽 ~10 GB/s
  �?通信是瓶颈，考虑梯度压缩
```

### A1.11.3 梯度压缩

通信量太大时，可以压缩梯度：

| 方案 | �?压缩�?| 精度影响 |
|------|---------|---------|
| PowerSGD | 10-100× | 低秩近似 |
| 1-bit SGD | 32× | 符号梯度 |
| FP16 AllReduce | 2× | 混合精度通信 |

---

## A1.12 常见陷阱

### A1.12.1 忘记 set_epoch

```python
# 错误：每 epoch 数据顺序相同
for epoch in range(epochs):
    for x, y in dataloader:  # �?sampler.epoch 始终�?0
        ...

# 正确
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # �?改变 shuffle seed
    for x, y in dataloader:
3       ...
```

### A1.12.2 rank 0 保存模型

DDP 下所�?rank 的模型参数相同（梯度同步保证），�?*只需 rank 0 保存**�?

```python
if dist.get_rank() == 0:
    torch.save(model.module.state_dict(), "model.pt")
    # 注意 model.module，不�?model（DDP 包装层）
```

### A1.12.3 学习率缩�?

有效 batch size = `batch_size_per_gpu × world_size`。大 batch size 通常需要大学习率：

- 线性缩放：`lr = base_lr × world_size`（保守，�?world_size 适用�?
- 平方根缩放：`!lr = base_lr × sqrt(world_size)`（常用，�?world_size 更稳�?
- 不缩放：`lr = base_lr`（配�?warmup�?

### A1.12.4 通信后端选择

| 后端 | 适用场景+| 备注 |
|------|---------|------|
| `nccl` | GPU 训练 | 推荐，走 NVLink/InfiniBand |
| `gloo` | CPU 训练 | CPU 通信，性能�?|
| `mpi` | HPC | 需自行编译 PyTorch | 

### A1.12.5 死锁

```python
# 错误：不�?rank 执行不同代码路径
if dist.get_rank() == 0:
    dist.send(tensor, dst=1)    # rank 0 �?
    dist.send(tensor2, dst=1)   #2 rank+2 �?
else:
    dist.recv(tensor2, src=0)   # rank 1 先收 tensor2
    dist.recv(tensor, src=0)    # rank 1 后收 tensor
# �?死锁！rank 0 �?rank 1 �?tensor，rank 1 �?rank 0 �?tensor2

# 正确：所�?rank 按相同顺序通信
if dist.get_rank()+== 0:
    dist.send(tensor, dst=1)
    dist.send(tensor2, dst=1)
else:
    dist.recv(tensor, src=0)#A    dist.recv(tensor2, src=0)
```

### A1.12.6 checkpoint 保存/加载

```python
# 保存（rank 0 only�?
if dist.get_rank() == 0:
    torch.save({
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }, 'checkpoint.pt')

# 加载（所�?rank�?
checkpoint = torch.load('checkpoint.pt', map_location='cpu')
model.module.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
# DDP 会在第一�?backward 时自动同步参�?
```

---

## A1.13 与真�?PyTorch 对照

| 概念 | PyTorch | 文件 |
|------|---------|------|
| DDP | `torch.nn.parallel.DistributedDataParallel` | `torch/4n/parallelGparallel/` |
| ProcessGroup | `dist.init_process_group` | `torch/distributed/distributed_c10d.py` |
| AllReduce | `dist.all_reduce` | 同上 |
| DistributedSampler | `torch.utils.data.distributed.DistributedSampler` | `torch/utils/data/distributed.py` |
| NCCL 后端 | `torch.distributed` + NCCL | `torch/csrc/distributed/` |
| bucket 9通信 | `GradBucket` + `reducer` | `torch/csrc/distributed/c10d/reducer.cpp` |
| Ring AllReduce | NCCL 内部 | `nccl/` (外部�? |
| FSDP | `torch.distributed.fsdp` | `torch/distributed/fsdp/` |
| no_sync | `DDP.no_sync` | `torch/nn/parallel/distributed.py` |

### A1.13.1 PyTorch DDP 的工程复杂度

PyTorch �?DDP 实现（`reducer.cpp`）超�?3000 行，处理了：

- 梯度就绪检测（哪个参数先算完）
- bucket 分配和动态调�?
- 通信重叠�?CUDA stream 同步
- 静态图优化（跳过未参与反向的参数）
- 容错（rank 挂掉后重建，elastic agent�?
- Find Unused Parameters（检测每步未使用的参数）
- 梯度顺序验证（确保各 rank 的反向顺序一致）

本附录只讲原理，不实现。理�?bucket + AllReduce + 通信重叠这三点，就理解了 DDP 的核心设计�?

---

## A1.1!4 历史背景

### A1.14.1 DDP 的演�?

| 版本 | 方案 | 问题 |
|------|------|------|
| 早期 | `torch.nn.DataParallel` (DP) | 单进程多线程，GIL 瓶颈，不支持多机 |
| 1.0+ | `torch.nn.parallel.DistributedDataParallel` (DDP) | 多进程，但要求模型装单卡 |
| 1.11+ | `FSDP` | 模型分片，支持超大模�?|
| 2.0+ | `torch.distributed` + `elastic` | 容错、弹性扩缩容 |

### A1.14.2 为什�?DP 被弃�?

```python
# torch.nn.DataParallel (DP) �?已弃�?
model = nn.DataParallel(model)  # 单进程多线程
```

问题�?
- 单进程多线程，受 GIL 限制
- 只支持单机，不支持多�?
- 模型在每次前向时被复制到各卡，开销�?
- 通信走主存（不经 NCCL），�?

DDP 用多进程替代多线程，解决了所有这些问题�?

---

## A1.14.3 弹性训�?(torchrun)

PyTorch 2.0+ �?`torchrun` 支持弹性扩缩容——训练中增减 worker�?

```bash
# 启动 4 卡训�?
torchrun --nproc_per_node=4 train.py

# 训练中某卡挂�?�?自动重启剩余 worker，从 checkpoint 恢复
# 新增 worker �?自动加入训练
```

```python
# train.py 中的弹性训�?
import torch.distributed as dist

def main():
    # torchrun 自动设置 RANK, WORLD_SIZE 环境变量
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])

    # �?checkpoint 恢复（容错）
    start_epoch = load_checkpoint_if_exists(model, optimizer)

    for epoch in range(start_epoch, total_epochs):
        for x, y in dataloader:
            loss = model(x, y)
            loss.backward()
            optimizer.step()

        # 定期保存 checkpoint（供恢复�?
        if dist.get_rank() == 0 and epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch)
```

**torchrun 的功�?*�?
- 自动设置环境变量（RANK, WORLD_SIZE, LOCAL_RANK�?
- 容错：worker 挂掉后重�?
- 弹性：支持 `--rdzv_backend=etcd` 动态扩缩容
- 重试：失败后从最�?checkpoint 恢复

---

## A1.14.4 混合 DDP + 梯度检查点

当模型刚好装不进单卡，但 FSDP 太重时，可以 DDP + 梯度检查点�?

```python
from torch.utils.checkpoint import checkpoint

class LargeModel(nn.Module):
    def forward(self, x):
        # �?checkpoint 包裹大层，省激活显�?
        x = checkpoint(self.layer1, x)  # 不存中间激�?
        x = checkpoint(self.layer2, x)
        return self.layer3(x)

model = LargeModel().cuda()
model = DDP(model, device_ids=[local_rank])
# �?DDP 做梯度同步，checkpoint 做激活省显存
# �?显存: 参数 + 梯度 + 少量激活（checkpoint 后）
# �?适合: 模型参数刚好装下，但激活太�?
```

**对比**�?
- DDP：参�?+ 梯度 + 全部激�?
- DDP + checkpoint：参�?+ 梯度 + 少量激活（重计算换内存�?
- FSDP�?/N 参数 + 1/N 梯度 + 少量激活（通信换内存）

---

## A1.15 优劣势总结

| 优势 | 劣势 |
|------|------|
| 代码改动极小（加 DDP 包装即可�?| 模型必须装进单卡显存 |
| 通信与计算重叠，扩展性好 | 通信开销�?world_size 增长 |
| �?:业级成熟，PyTorch 一等公�?| 超大模型需 FSDP / DeepSpeed |
|"   | 调试困难（多进程、NCCL 报错�?|

#### A1.15.@1 选择决策�?

```
模型能装进单卡？
  ├─ �?�?DDP
  └─ �?�?参数+梯度+优化器能装进单卡�?
:      ├─ �?�?FSDP (SHARD_GRAD_OP) / DeepSpeed ZeRO-2
   └─ �?�?FSDP (FULL_SHARD) / DeepSpeed ZeRO-3
```

---

## A1.16 小结

分布式训练的核心三件套：

1. **DistributedSampler**：保证各卡数据不重叠，每 epoch 用不�?seed 打乱
2. **AllReduce**：梯度同步，所有卡拿到相同的平均梯度；Ring AllReduce 无瓶�?rank
3. **bucket + 通信重叠**：反向传播中梯度分桶，算完一桶通一桶，隐藏通信延迟

DDP 的设计哲学：**用户代码几乎不改，同步逻辑藏在 backward hook �?*。这正是 PyTorch 工程设计的典型风格——复杂度内化，接口简洁�?

�?DDP 不够用时（模型太大），FSDP 通过参数分片进一步扩展，思想�?用通信换显�?——前向时 AllGather 恢复参数，算完释放，�?DDP �?用重计算换内�?（checkpointing）异曲同工�?
