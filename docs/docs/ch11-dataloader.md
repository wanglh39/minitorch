# 第 11 章：数据加载与采样

> 本章我们离开"张量与算子"的纯净世界，踏入真实训练里最繁杂、也最容易被忽视的一块——**数据管线**。
> 一个深度学习训练循环里，前向、反向、优化器加起来可能只占代码的 30%，剩下 70% 都在折腾"怎么把数据整整齐齐地喂进模型"。
> PyTorch 把这件事拆成了三个角色：`Dataset`、`Sampler`、`DataLoader`，三者各司其职、组合使用。
> 本章我们就从零实现这套机制，并讲清楚为什么是这样切分的。

---

## 11.1 本章目标

读完这一章，你应当能够：

1. 说出 `Dataset`、`Sampler`、`DataLoader` 三者各自的职责，并能解释"为什么要把它们分开"。
2. 区分 **map-style** 与 **iterable** 两种数据集协议，知道什么场景该用哪一种。
3. 实现 `SequentialSampler`、`RandomSampler`、`BatchSampler`，并理解它们如何层层组合。
4. 写出 `default_collate` 的"转置 + stack"逻辑，解释它为什么对 `(x, y)` 形式的样本有效。
5. 说清 `drop_last` 的含义，以及它为何对 `BatchNorm` 这类对 batch size 敏感的层至关重要。
6. 理解多进程预取（`num_workers > 0`）绕过 GIL 的原理，以及教学版为什么省略它。
7. 用 `ConcatDataset` 把多个数据集拼成一个，并解释"累积大小索引"如何把全局下标映射回子集。
8. 把以上零件组装成一个端到端的小型训练数据管线，并跑通它。

---

## 11.2 原理铺垫

### 11.2.1 训练循环里的数据流

一个最朴素的训练循环长这样：

```python
for epoch in range(epochs):
    for x, y in somehow_get_batches(train_data):
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
```

那个 `somehow_get_batches` 就是本章要实现的东西。把它拆开，里面其实有四个子问题：

| 子问题            | 通俗说法                       | 谁来管           |
| --------------- | -------------------------- | ------------- |
| 数据从哪儿来？怎么取一条？ | "我有一堆样本，给我第 i 个"           | `Dataset`     |
| 按什么顺序取？        | "先打乱再取，还是顺序取？要不要放回？"      | `Sampler`     |
| 一次取几条？怎么打包成批？ | "一次 32 条，最后不够 32 条怎么办？"   | `BatchSampler` + `collate_fn` |
| 怎么把这一批喂给模型？    | "把 32 个 `(x, y)` 拼成两个大张量"  | `DataLoader`  |

这种切分不是 PyTorch 拍脑袋想出来的，而是从实践中长出来的——下一节的"历史背景"会讲它的演化。

### 11.2.2 map-style vs iterable：两种数据集协议

**map-style Dataset** 长这样：

```python
class Dataset:
    def __getitem__(self, index): ...   # 给我第 index 个
    def __len__(self): ...              # 一共多少个
```

它假设数据集是一个**有长度、可随机访问**的容器。就像一本字典：你问"第 3 条是什么"，它直接给你。

**IterableDataset** 长这样：

```python
class IterableDataset:
    def __iter__(self): ...             # 我只能一条一条吐给你
```

它假设数据是**流**，没有长度概念，也不能跳着读。就像一根水管：你只能接住流出来的水，不能问"第 3 滴是什么"。

::: tip 什么时候用 IterableDataset？
- 数据太大放不下内存（TB 级日志、数千万行数据库）。
- 数据来自网络流、Kafka、生成器。
- 数据是"无限"的（强化学习的 episode 流）。
- 数据本身就有顺序依赖（时序预测，打乱会破坏时间结构）。
:::

::: warning map-style 的代价
map-style 要求你能 O(1) 跳到任意位置。如果你的数据存在云端对象存储、每条都要单独 HTTP 请求，那 `__getitem__(10000)` 会非常慢——这时候反而 IterableDataset 顺序读更高效。
:::

### 11.2.3 Sampler：把"顺序"这件事独立出来

为什么不直接在 `Dataset` 里加个 `shuffle=True`？因为**顺序和数据是两件事**：

- 同一份数据，训练时要打乱，验证时要顺序，做 k-fold 时要按特定子集取。
- 如果把 shuffle 写死在 Dataset 里，每次想换顺序都得改 Dataset，没法复用。

所以 PyTorch 把"取哪些下标、按什么顺序"单独抽出来叫 `Sampler`。它只产出一串**整数下标**：

```
SequentialSampler  → 0, 1, 2, 3, ..., n-1
RandomSampler      → 7, 2, 9, 0, 5, ...  (一个随机排列)
RandomSampler(replacement=True, num_samples=10000) → 任意 10000 个（可重复）
```

然后 `BatchSampler` 包一层，把这一串下标**切成 batch**：

```
BatchSampler(SequentialSampler, batch_size=3, drop_last=False)
  → [0,1,2], [3,4,5], [6,7,8], [9]
```

注意这里产出的是**下标的 batch**，不是样本的 batch。这样 DataLoader 拿到 `[0,1,2]` 后再去 `dataset[0], dataset[1], dataset[2]` 取真实样本。这种"先定顺序、再取数据"的两步走，让顺序逻辑和数据读取彻底解耦。

### 11.2.4 collate：把一捆样本缝成一个 batch

假设 `dataset[i]` 返回一个 tuple `(x_i, y_i)`，其中 `x_i` 是形状 `[3]` 的特征，`y_i` 是标量标签。
DataLoader 拿到一个 batch 的下标 `[0, 1, 2]` 后，先取出来：

```python
samples = [dataset[0], dataset[1], dataset[2]]
        = [(x0, y0), (x1, y1), (x2, y2)]
```

现在要把这 3 个 tuple 缝成两个大张量：

```python
batch_x = stack([x0, x1, x2])   # shape [3, 3]
batch_y = stack([y0, y1, y2])   # shape [3]
```

这个"**先转置、再逐列 stack**"的操作就是 `default_collate`。图示：

```
样本列表（按样本分组）:        转置（按字段分组）:        stack:
[(x0, y0),                  →  [(x0, x1, x2),         →  (stack(x0,x1,x2),
 (x1, y1),                       (y0, y1, y2)]            stack(y0,y1,y2))
 (x2, y2)]
```

为什么是"转置"？因为样本是"行优先"（一个样本里所有字段挨着），但 batch 要"列优先"（同字段的样本挨着）。`zip(*batch)` 正好做这个转置。

### 11.2.5 drop_last：为什么有时候必须丢掉尾巴

假设 10 个样本，batch_size=3，最后会剩一个 `[9]` 的孤儿 batch。
大多数情况无所谓，但有两种情况必须丢：

1. **BatchNorm**：`running_mean / running_var` 在训练时按 batch 统计。如果最后一个 batch 只有 1 个样本，方差是 0 或未定义，会污染统计量。
2. **形状固定的算子**：某些自定义层在编译时假设了 batch 维度，变长 batch 会触发重编译或报错。

`drop_last=True` 就是把那个不完整的尾批直接扔掉。代价是浪费不到一个 batch 的数据——在大数据集上可以忽略。

### 11.2.6 多进程预取：绕过 GIL

Python 有 GIL（全局解释器锁），同一时刻只有一个线程执行 Python 字节码。数据加载是 I/O + 解码密集型，单线程下 CPU 和 GPU 都得等它。

PyTorch 的解法是 `num_workers > 0`：开多个**子进程**（不是线程），每个子进程独立解析数据，通过队列把 batch 送给主进程。子进程各有自己的 GIL，互不阻塞，主进程一边训练一边从队列取下一批——这就是"预取"。

教学版为什么省略？因为多进程要处理：子进程间 Tensor 的共享内存（避免拷贝）、worker 的生命周期管理、`IterableDataset` 的分片、随机种子的正确传播、`pin_memory`、异常传播……这些工程细节会淹没教学主线。所以我们只实现 `num_workers=0` 的单进程版，把"组合"这件事讲透，多进程留作练习。

### 11.2.7 ConcatDataset：累积大小索引

把两个数据集 `ds1`（3 条）和 `ds2`（4 条）拼成一个 7 条的大数据集。问题：用户问 `cds[5]`，怎么知道该去 `ds2` 取？

解法是预先算一个**累积大小数组**：

```
ds1: 3 条 → cum_sizes = [3]
ds2: 4 条 → cum_sizes = [3, 7]
```

来一个全局下标 `i=5`，从左往右扫 `cum_sizes`：

- `i=5 < 3`? 否。
- `i=5 < 7`? 是。→ 落在 `ds2`，子集内下标 = `5 - 3 = 2`，返回 `ds2[2]`。

这是经典的"前缀和 + 二分查找"模式。数据集很多时可以二分加速，但教学版线性扫就够。

### 11.2.8 整体数据流图示

把前面所有零件串起来，一次 `for batch in loader` 的完整数据流长这样：

```
用户代码: for batch in DataLoader(ds, batch_size=4, shuffle=True):
              │
              ▼
DataLoader.__iter__
              │
              ▼
BatchSampler.__iter__          ← 切下标 batch
              │
              ▼
RandomSampler.__iter__         ← 产出打乱的下标
              │  (7, 2, 9, 0, 5, 8, 1, 6, 3, 4)
              ▼
切成长度 4 的列表: [7,2,9,0], [5,8,1,6], [3,4]
              │
              ▼
DataLoader: indices = [7,2,9,0]
              │
              ▼
samples = [ds[7], ds[2], ds[9], ds[0]]   ← 真正访问数据
              │
              ▼
default_collate(samples)                  ← 转置 + stack
              │
              ▼
batch = (stack(x7,x2,x9,x0), stack(y7,y2,y9,y0))
              │
              ▼
yield batch 给用户
```

这张图里最值得品味的是：**下标先于数据流动**。Sampler 完全不碰真实样本，只产出整数。这让"顺序策略"和"数据读取"可以独立测试、独立替换。比如你想做"按难度排序的 curriculum learning"，只要写一个 `CurriculumSampler`，DataLoader 和 Dataset 都不用改。

### 11.2.9 为什么不用迭代器协议而用 `__getitem__`

初学者常问：既然 DataLoader 最终是个迭代器，为什么 Dataset 不直接定义 `__iter__`？

答案是：**map-style 的核心价值是支持随机访问**。shuffle 要跳着取（`ds[7]`、`ds[2]`...），迭代器只能顺序取，没法跳。如果 Dataset 用 `__iter__`，每次 shuffle 都得先把所有样本 materialize 进内存再打乱——大数据集直接 OOM。

`__getitem__` + `__len__` 这个协议让 Sampler 可以**只产出下标**，DataLoader 再按需取——下标是整数，几乎不占内存，打乱下标几乎零成本。这是整个设计的关键洞察。

代价是：数据源必须支持 O(1) 随机访问。对于流式数据（网络、生成器）做不到，所以才需要单独的 `IterableDataset` 协议——它放弃 shuffle，换得流式能力。

---

## 11.3 设计决策与权衡

| 决策                          | 我们的选择                              | 理由                                            | 代价                                       |
| --------------------------- | ---------------------------------- | --------------------------------------------- | ---------------------------------------- |
| Dataset 协议                  | `__getitem__` + `__len__`（map-style 优先） | 随机访问是大多数训练场景的需求，shuffle 必须能跳着取                | 流式数据用不了，需要单独的 IterableDataset            |
| Sampler 产出                  | 整数下标的迭代器                           | 把"顺序"从"数据"里剥离，复用性强                              | 多了一层抽象，初学者要绕一下                          |
| BatchSampler 是不是 Sampler    | 是，产出"下标列表"的迭代器                    | 递归组合：BatchSampler 包任意 Sampler                  | `__len__` 语义变了（返回 batch 数而不是样本数）         |
| shuffle 怎么实现                | DataLoader 内部根据 `shuffle` 选 Sampler | 用户只要传 `shuffle=True`，不用手写 RandomSampler         | 想要更细控制（如 weighted）必须自己传 sampler          |
| collate 默认行为                | tuple/list 转置后递归 stack             | 覆盖 90% 的 `(x, y)` 场景                          | 不规则结构（dict、字符串）要用户自己写 collate_fn          |
| drop_last 默认                | `False`                            | 不浪费数据，符合直觉                                     | BatchNorm 用户必须记得开                       |
| num_workers                 | 只支持 0（单进程）                         | 教学主线清晰，避免 IPC/共享内存/种子传播的工程细节                  | 大数据集慢，不能演示真实预取                          |
| 随机数源                        | `numpy.random`                     | 与项目其他模块一致，方便 `np.random.seed` 复现               | 真实 PyTorch 用独立 `Generator`，多 worker 时更可控 |
| ConcatDataset 索引            | 线性扫 cum_sizes                      | 实现简单，数据集数通常很少                                 | 子集很多时 O(n)，可改二分到 O(log n)                |
| ArrayDataset vs TensorDataset | 都提供                                | 有的用户数据是 numpy array，有的是 Tensor，避免来回转换          | 两个类代码几乎重复，可统一但会牺牲类型清晰                   |

---

## 11.4 代码逐行实现

### 11.4.1 `dataset.py`：数据集协议

```python
"""Dataset：数据集协议（Ch11）。

设计要点：
  - Dataset: __getitem__/__len__ 协议（map-style）。
  - IterableDataset: __iter__ 协议（streaming）。
  - TensorDataset: 多个 Tensor 沿第一维对齐，__getitem__ 返回各 Tensor 同索引切片。
  - 与真实 PyTorch 的 utils/data/dataset.py 对应。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from ..tensor import Tensor


class Dataset:
    # 之所以只定义两个 dunder 而不提供默认实现，是要把它当"协议"（interface）用。
    # 子类必须重写，否则直接 NotImplementedError —— 强制用户明确表态。
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class IterableDataset:
    # 注意它没有 __getitem__/__len__：流式数据根本没有"第 i 个"和"总长"的概念。
    def __iter__(self) -> Iterator:
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors: Tensor):
        # 取第一个 tensor 的第一维作为基准长度
        n = tensors[0].shape[0]
        for t in tensors:
            if t.shape[0] != n:
                raise ValueError("所有 Tensor 第一维必须相同")
        # 不拷贝，只持有引用 —— Dataset 是"视图"，不拥有数据
        self.tensors = tensors

    def __getitem__(self, index):
        # 关键：对每个 tensor 取同一下标，返回 tuple
        # 这样 (x, y) 配对永远对齐，不会因为各自索引乱掉
        return tuple(t[index] for t in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]


class ArrayDataset(Dataset):
    # 与 TensorDataset 完全对称，只是底层是 numpy array
    # 提供这个类是因为很多用户的原始数据就是 ndarray，省一次转换
    def __init__(self, *arrays: np.ndarray):
        n = len(arrays[0])
        for a in arrays:
            if len(a) != n:
                raise ValueError("所有数组长度必须相同")
        self.arrays = arrays

    def __getitem__(self, index):
        return tuple(arr[index] for arr in self.arrays)

    def __len__(self):
        return len(self.arrays[0])


class ConcatDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = list(datasets)
        # 预算累积大小：cum_sizes[i] = 前 (i+1) 个子集的总长
        # 之后 __getitem__ 用它做"全局下标 → (子集, 子集内下标)"的映射
        self._cum_sizes: list[int] = []
        cum = 0
        for ds in self.datasets:
            cum += len(ds)
            self._cum_sizes.append(cum)

    def __len__(self):
        # 总长就是最后一个累积值
        return self._cum_sizes[-1] if self._cum_sizes else 0

    def __getitem__(self, index):
        # 支持负下标，与 Python list 习惯一致
        if index < 0:
            index += len(self)
        # 线性扫 cum_sizes 找第一个"门槛"高于 index 的子集
        for i, cum_size in enumerate(self._cum_sizes):
            if index < cum_size:
                # prev 是该子集的起始全局下标
                prev = self._cum_sizes[i - 1] if i > 0 else 0
                return self.datasets[i][index - prev]
        raise IndexError(index)
```

**几个值得注意的细节：**

- `TensorDataset.__getitem__` 返回 `tuple`，不是 `list`。tuple 不可变，语义上更符合"一个样本"这种不会被修改的东西。
- `ConcatDataset` 没有校验子集之间的类型一致性——这是有意的，允许你拼异构数据集（虽然不常见）。
- `_cum_sizes` 用下划线开头表示"私有"，但 Python 没有真正的私有，用户实在想读也能读到。这是 PyTorch 也用的约定。

### 11.4.2 `sampler.py`：采样器

```python
"""Sampler：采样器（Ch11）。

Sampler 产出索引序列，DataLoader 据此从 Dataset 取样本。
  - SequentialSampler: 0, 1, 2, ..., n-1
  - RandomSampler: 随机排列（可 replacement / num_samples）
  - BatchSampler: 把索引列表切成 batch

对应真实 PyTorch 的 utils/data/sampler.py。
"""

from __future__ import annotations

import numpy as np


class Sampler:
    # 同 Dataset，纯协议。子类必须实现 __iter__。
    # __len__ 可选但推荐：DataLoader 要靠它算 len(loader)
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        # 只持有引用，不拷贝数据。Sampler 永远不碰数据本身，只看它的长度。
        self.data_source = data_source

    def __iter__(self):
        # range 是惰性的，省内存
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self, data_source, replacement: bool = False, num_samples: int | None = None):
        self.data_source = data_source
        self.replacement = replacement
        # num_samples 默认等于数据集大小；允许 > n（配合 replacement=True 做过采样）
        self._num_samples = num_samples if num_samples is not None else len(data_source)

    @property
    def num_samples(self):
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            # 有放回：每个样本独立从 [0, n) 抽
            # randint 高效生成 num_samples 个独立下标
            yield from (int(x) for x in np.random.randint(0, n, size=self._num_samples))
        else:
            # 无放回：生成一个排列，取前 num_samples 个
            # 如果 num_samples < n，permutation 后切片比 sample 快
            yield from (int(x) for x in np.random.permutation(n)[: self._num_samples])

    def __len__(self):
        return self._num_samples


class BatchSampler(Sampler):
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        # 包一个底层 sampler，自己负责"切 batch"
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch        # yield 一个满 batch
                batch = []
        # 循环结束后处理尾巴
        if batch and not self.drop_last:
            yield batch            # 不丢就 yield 不完整 batch
        # 如果 drop_last=True，这里什么都不做，孤儿 batch 被丢弃

    def __len__(self):
        # batch 总数。drop_last 时整除；否则向上取整
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size
```

**关键设计点：**

- `BatchSampler` 也是一个 `Sampler`，只不过它产出的元素是 `list[int]` 而不是 `int`。这种"递归语义"是 PyTorch 数据管线最巧妙的地方——你可以继续包一层（比如加权重采样）。
- `__len__` 的两个公式分别对应"整除"和"向上取整"。`(n + b - 1) // b` 是整数向上取整的常用写法，避免引入 `math.ceil`。
- `RandomSampler` 用 `yield from` 把生成器委托出去，避免在内存里一次性存 `num_samples` 个下标（虽然 numpy 已经存了，但语义上保持惰性）。

### 11.4.3 `dataloader.py`：加载器

```python
"""DataLoader：数据加载器（Ch11）。

组合 Dataset + Sampler + collate_fn。
  - batch_size: 每批样本数
  - shuffle: 是否随机打乱
  - drop_last: 丢弃不完整尾批
  - collate_fn: 把 list of samples 拼成 batch tensor

教学版 num_workers=0 单进程。真实 PyTorch 多进程预取绕 GIL。
对应真实 PyTorch 的 utils/data/dataloader.py。
"""

from __future__ import annotations

import numpy as np

from .dataset import Dataset
from .sampler import BatchSampler, RandomSampler, SequentialSampler


def default_collate(batch: list) -> list:
    """把 list of samples 拼成 batch。

    sample 是 tuple/list → 转置后逐元素 stack。
    sample 是 ndarray → stack 成 batch array。
    """
    # 第一类：样本是 tuple 或 list（如 (x, y)）
    if isinstance(batch[0], tuple | list):
        # zip(*batch) 做转置：[(x0,y0),(x1,y1)] → [(x0,x1),(y0,y1)]
        # strict=True 保证所有样本字段数相同，否则报错（防止数据脏）
        transposed = zip(*batch, strict=True)
        # 对每一列递归 collate：x 列 stack 成 batch_x，y 列 stack 成 batch_y
        return [default_collate(list(col)) for col in transposed]
    # 第二类：样本直接是 ndarray（只有 x 没有 y）
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)   # 沿新第一维拼起来
    # 第三类：标量或数字列表，fallback 用 np.array
    return np.array(batch)


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        # 没传 collate_fn 就用默认的；用户可注入自定义的
        self.collate_fn = collate_fn or default_collate

        # 这里是"便利接口"：用户只说 shuffle=True/False，
        # 我们替他选好 Sampler。想精细控制可以自己传 sampler（教学版省略了这个参数）
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        # 再包一层 BatchSampler 切 batch
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self):
        # 核心循环：拿一批下标 → 取样本 → collate 成 batch → yield
        for indices in self.batch_sampler:
            samples = [self.dataset[i] for i in indices]
            yield self.collate_fn(samples)

    def __len__(self):
        # 让 `len(loader)` 返回 batch 数，方便 tqdm 进度条
        return len(self.batch_sampler)
```

**逐行要点：**

- `default_collate` 是**递归**的。如果样本是嵌套 tuple `((a, b), c)`，它会一层层转置到底。这个设计让 collate 对任意嵌套结构都有效。
- `zip(*batch, strict=True)` 的 `strict` 是 Python 3.10+ 才有的参数，保证所有样本字段数一致。教学版用上了，更安全。
- `DataLoader.__init__` 里把 `shuffle` 翻译成 `Sampler`，这是"便利接口"模式：底层只认 Sampler，但给用户一个更简单的开关。
- `__iter__` 是生成器，每次 yield 一个 batch。这意味着 `for batch in loader` 会**惰性**地一个一个产出，不会一次性把所有 batch 都装进内存。

---

## 11.5 完整示例

下面是一个端到端的小例子，覆盖所有零件：

```python
import numpy as np
from minitorch import Tensor
from minitorch.data import (
    ArrayDataset, TensorDataset, ConcatDataset,
    SequentialSampler, RandomSampler, BatchSampler, DataLoader,
)

# ── 1. 准备数据 ──────────────────────────────────────────
np.random.seed(0)
X = np.arange(20).reshape(10, 2).astype(np.float64)   # 10 个样本，每个 2 维特征
Y = np.arange(10)                                      # 10 个标签
print("X shape:", X.shape, "Y shape:", Y.shape)

# ── 2. 用 ArrayDataset 包装 ─────────────────────────────
ds = ArrayDataset(X, Y)
print("len(ds):", len(ds))
print("ds[3]:", ds[3])          # (array([6, 7]), 3)

# ── 3. 直接用 Sampler 看顺序 ────────────────────────────
print("Sequential:", list(SequentialSampler(ds)))
np.random.seed(42)
print("Random:", list(RandomSampler(ds)))

# ── 4. BatchSampler 切批 ────────────────────────────────
bs = BatchSampler(SequentialSampler(ds), batch_size=3, drop_last=False)
print("Batches:", list(bs))
print("len(bs):", len(bs))

# ── 5. DataLoader 端到端 ────────────────────────────────
loader = DataLoader(ds, batch_size=4, shuffle=False, drop_last=False)
print("len(loader):", len(loader))
for i, batch in enumerate(loader):
    print(f"batch {i}:", batch)

# ── 6. shuffle=True 看打乱 ──────────────────────────────
np.random.seed(42)
loader_sh = DataLoader(ds, batch_size=4, shuffle=True)
for i, batch in enumerate(loader_sh):
    print(f"shuffled batch {i}:", batch)

# ── 7. drop_last=True ───────────────────────────────────
loader_dl = DataLoader(ArrayDataset(X), batch_size=3, drop_last=True)
print("drop_last batches:", len(loader_dl))
for batch in loader_dl:
    print("  batch shape:", batch.shape)

# ── 8. ConcatDataset 拼接 ───────────────────────────────
ds1 = ArrayDataset(np.arange(3))
ds2 = ArrayDataset(np.arange(3, 7))
cds = ConcatDataset([ds1, ds2])
print("len(cds):", len(cds))
print("cds[0], cds[3], cds[6]:", cds[0], cds[3], cds[6])

# ── 9. TensorDataset + 自定义 collate ───────────────────
tx = Tensor.from_numpy(X)
ty = Tensor.from_numpy(Y)
tds = TensorDataset(tx, ty)

def my_collate(batch):
    xs = np.stack([b[0].numpy() for b in batch])
    ys = np.stack([b[1].numpy() for b in batch])
    return xs, ys

loader_t = DataLoader(tds, batch_size=2, collate_fn=my_collate)
bx, by = next(iter(loader_t))
print("custom collate batch_x:\n", bx)
print("custom collate batch_y:", by)
```

预期输出（节选）：

```
X shape: (10, 2) Y shape: (10,)
len(ds): 10
ds[3]: (array([6., 7.]), 3)
Sequential: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Random: [3, 7, 8, 5, 9, 0, 1, 6, 2, 4]
Batches: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
len(bs): 4
len(loader): 3
batch 0: [array([[0., 1.], [2., 3.], [4., 5.], [6., 7.]]), array([0, 1, 2, 3])]
...
len(cds): 7
cds[0], cds[3], cds[6]: (array([0]),) (array([3]),) (array([6]),)
```

### 11.5.1 接进训练循环

把 DataLoader 接进一个最小训练循环，看完整数据流：

```python
import numpy as np
from minitorch import Tensor
from minitorch.nn import Linear, Sequential
from minitorch.optim import SGD
from minitorch.data import ArrayDataset, DataLoader

np.random.seed(0)
# 假数据：y = x0 + x1 + 噪声
X = np.random.randn(64, 2)
Y = X.sum(axis=1, keepdims=True) + 0.01 * np.random.randn(64, 1)

ds = ArrayDataset(X, Y)

def collate(batch):
    xs = np.stack([b[0] for b in batch])
    ys = np.stack([b[1] for b in batch])
    return Tensor.from_numpy(xs), Tensor.from_numpy(ys)

loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)

model = Sequential(Linear(2, 8), Linear(8, 1))
opt = SGD(model.parameters(), lr=0.05)

for epoch in range(5):
    total = 0.0
    n = 0
    for bx, by in loader:
        pred = model(bx)
        loss = ((pred - by) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
        n += 1
    print(f"epoch {epoch}  avg_loss={total/n:.4f}")
```

这个例子里 DataLoader 干的事：每 epoch 把 64 个样本打乱、切成 4 个 batch、每个 batch 把 `(x, y)` 配对 stack 成两个大 Tensor。模型只管吃 batch Tensor、吐 loss。**数据管线和训练逻辑彻底解耦**——换数据集只改 `ds`，换 batch 策略只改 loader 参数，模型和优化器代码一行不动。

---

## 11.6 常见陷阱

### 陷阱 1：忘了 `np.random.seed`，测试随机失败

`RandomSampler` 用全局 `np.random`。如果测试里不 seed，每次跑结果不同，断言会随机挂。

**解决**：测试或复现实验前永远 `np.random.seed(...)`。生产代码应该用独立的 `Generator`（教学版省略）。

### 陷阱 2：以为 `shuffle=True` 会原地打乱 Dataset

不会。`shuffle` 只影响 Sampler 产出的下标顺序，Dataset 内部数据顺序永远不变。这是好事——同一份 Dataset 可以同时给训练 loader（shuffle=True）和验证 loader（shuffle=False）用。

### 陷阱 3：`drop_last=False` 配 BatchNorm 训练崩

最后一个 batch 只有 1 个样本时，`BatchNorm` 算方差会出 NaN/0。

**解决**：训练 loader `drop_last=True`，或者用 `BatchNorm` 的 `track_running_stats=False`，或者干脆换 `LayerNorm`。

### 陷阱 4：自定义 `collate_fn` 忘了处理变长样本

NLP 里句子长度不一，直接 `np.stack` 会报错。

**解决**：写一个 pad_collate：

```python
def pad_collate(batch):
    xs = [b[0] for b in batch]
    maxlen = max(len(x) for x in xs)
    padded = np.zeros((len(xs), maxlen))
    for i, x in enumerate(xs):
        padded[i, :len(x)] = x
    return padded, np.array([b[1] for b in batch])
```

### 陷阱 5：`IterableDataset` 用 `shuffle=True`

DataLoader 的 `shuffle` 依赖 `__len__` 和随机下标访问，IterableDataset 都没有。传 `shuffle=True` 会报错或静默无效。

**解决**：IterableDataset 要自己实现打乱（通常用一个 buffer 重排，如 `torch.utils.data.IterableDataset` 的 shuffling 技巧）。

### 陷阱 6：以为 `len(loader)` 是样本数

不是。`len(loader)` 是 **batch 数**。样本数是 `len(loader.dataset)`。这个混淆会让 tqdm 总步数算错。

### 陷阱 7：`ConcatDataset` 子集顺序变了

`ConcatDataset` 按你传入的 `datasets` 列表顺序拼接。如果两次运行传的顺序不同，全局下标含义就变了——已保存的"第 5 个样本"可能对应到不同数据。

**解决**：固定 `datasets` 顺序，或用字典按 key 排序后再传。

### 陷阱 8：`__getitem__` 返回的 Tensor 离开 Dataset 后被修改

```python
class BadDataset(Dataset):
    def __init__(self):
        self.cache = Tensor.zeros(4, 3)
    def __getitem__(self, i):
        return self.cache[i]   # ← 返回的是视图，共享底层 storage
```

用户拿到 `ds[0]` 后做 `ds[0][0] = 999`，会改到 Dataset 内部缓存——下一个 epoch 数据就脏了。

**解决**：`__getitem__` 里返回前 `.clone()`，或者文档明确说"返回值不可修改"。教学版 `TensorDataset` 没克隆，因为 `t[index]` 走 `__getitem__` 已经返回新 Tensor（看 `Tensor.__getitem__` 实现是 `from_numpy(np.asarray(arr))`，会拷贝）。

### 陷阱 9：`batch_size=1` 时 collate 多余一层维度

```python
loader = DataLoader(ds, batch_size=1)
batch = next(iter(loader))   # shape [1, ...] 而不是 [...]
```

`np.stack` 永远加一维。batch_size=1 时这层维度多余，但符合"batch 维"语义。嫌麻烦可以在 collate 里 `if len(batch) == 1: return batch[0]`，但不推荐——破坏了"batch 永远有 batch 维"的不变量。

### 陷阱 10：在 `__getitem__` 里做重 I/O

```python
class ImageDataset(Dataset):
    def __getitem__(self, i):
        return imread(self.paths[i])   # ← 每次磁盘 I/O
```

单进程下每个 batch 都要同步等 I/O，GPU 闲置。真实 PyTorch 用 `num_workers > 0` 把 I/O 扔到子进程。教学版没多进程，只能建议：数据小就预加载进内存，数据大就用真实 PyTorch。

### 陷阱 11：以为 `DataLoader` 是 reentrant 的

```python
loader = DataLoader(ds, batch_size=4)
for b1 in loader:
    for b2 in loader:     # ← 嵌套迭代同一个 loader
        ...
```

教学版 `__iter__` 是生成器，每次 `iter(loader)` 新建一个生成器，嵌套迭代**看似**能跑。但 `RandomSampler` 用全局 `np.random`，两层会互相消耗随机数，结果不可预期。真实 PyTorch 的 DataLoader 也是一次性的（`_BaseDataLoaderIter` 有状态），嵌套要新建独立 loader。

**解决**：嵌套循环用两个独立 DataLoader 实例。

---

## 11.7 与真实 PyTorch 对照

| minitorch                              | torch                                     | 关键差异                                                     |
| -------------------------------------- | ----------------------------------------- | -------------------------------------------------------- |
| `Dataset`（协议）                          | `torch.utils.data.Dataset`                | 一致                                                       |
| `IterableDataset`                      | `torch.utils.data.IterableDataset`        | 一致；真实版还提供 `get_state_dict` / `set_state_dict` 用于断点续训     |
| `TensorDataset`                        | `torch.utils.data.TensorDataset`          | 一致                                                       |
| `ArrayDataset`                         | （无直接对应，类似 `TensorDataset` 包 ndarray） | minitorch 额外提供，方便 numpy 用户                              |
| `ConcatDataset`                        | `torch.utils.data.ConcatDataset`          | 真实版用二分查找（`bisect`），O(log n)；教学版线性扫 O(n)                |
| `SequentialSampler` / `RandomSampler`  | 同名                                        | 真实版接受 `generator` 参数，多 worker 时种子隔离；教学版用全局 np.random   |
| `BatchSampler`                         | 同名                                        | 一致                                                       |
| `default_collate`                      | `torch.utils.data._utils.collate.default_collate` | 真实版支持 Tensor / Mapping / 命名元组 / 自定义 `__collate_fn__`；教学版只处理 tuple/list/ndarray |
| `DataLoader(num_workers=0)`            | `torch.utils.data.DataLoader`             | 真实版支持 `num_workers>0`、`pin_memory`、`prefetch_factor`、`persistent_workers`、`worker_init_fn` |
| `DataLoader(sampler=...)`              | 同名                                        | 真实版允许直接传 `sampler` 或 `batch_sampler`，互斥检查；教学版只暴露 `shuffle` |
| 无                                      | `WeightedRandomSampler`                   | 按样本权重采样，类别不平衡时用；教学版未实现                                  |
| 无                                      | `DistributedSampler`                     | 多卡训练切分数据，每张卡只取一部分；教学版未实现                                |

::: tip 真实 PyTorch 多了什么
最关键的是 **`num_workers > 0` 的多进程预取**。它涉及：
- 子进程用 `multiprocessing` 启动，各自独立 GIL。
- 数据通过 `multiprocessing.Queue` 传递，Tensor 走共享内存避免拷贝。
- 每个 worker 有独立随机种子（`worker_init_fn`），保证增强多样且可复现。
- `prefetch_factor` 控制每个 worker 预先准备几个 batch。
- `persistent_workers=True` 避免每个 epoch 重启子进程。

这些都是工程细节，不影响理解数据管线的"骨架"，所以教学版省略。
:::

---

## 11.8 历史背景

PyTorch 的数据管线并非一开始就长这样。

- **0.1 ~ 0.3（2017 前）**：只有简单的 `DataLoader`，shuffle 写死在内部，没有 Sampler 抽象。用户要做加权采样得自己 monkey patch。
- **0.4（2017 末）**：引入 `Sampler` 体系，`Dataset`/`Sampler`/`DataLoader` 三层分离定型。这是受当时 TensorFlow 的 `tf.data` API 竞争压力推动——后者强调"数据集 + 变换"的函数式风格，PyTorch 选择走更显式的面向对象路线。
- **1.0（2018 末）**：`IterableDataset` 加入，应对流式数据和大规模训练。
- **1.2（2019）**：`ConcatDataset` 等工具类完善，`random_split` 出现。
- **1.8 ~ 1.12**：多进程稳定性大幅改进，`persistent_workers`、`prefetch_factor` 加入，解决长 epoch 下反复 fork 子进程的开销。
- **2.0+**：`torch.compile` 开始介入数据管线（如 `DataLoader2` 实验），试图把数据预处理也编译进图。但经典 `DataLoader` 仍是主流。

minitorch 这套实现对应的是 PyTorch 1.0 前后的"经典三层"模型，去掉了多进程和分布式，保留核心抽象。

### 11.8.1 为什么 TensorFlow 走了不同的路

对比一下 TF1.x 的 `tf.data`：它用函数式 API（`dataset.map(...).shuffle(...).batch(...)`），强调"数据集是一串变换"。好处是可编译、可并行规划；坏处是抽象层次高、调试难（一个 pipeline 报错难定位到哪一步）。

PyTorch 选了更显式的 OO 路线：Dataset 是个类、Sampler 是个类、DataLoader 是个类，用户能 print、能断点、能继承改一个方法。代价是没法像 TF 那样整体编译优化数据管线。

2.0 后两边在靠拢：TF2 默认 eager，PyTorch 加 `DataLoader2` 试图编译。但经典 API 仍是各自的主流。

### 11.8.2 一个常被忽略的点：epoch 边界

注意 `DataLoader` 是**一次性的迭代器**——`for batch in loader` 跑完后，要再跑一个 epoch 必须**重新 `iter(loader)`**（即再写一次 `for batch in loader`）。这是因为 `RandomSampler` 每次迭代重新生成排列，保证每个 epoch 打乱不同。

如果错误地缓存 `it = iter(loader)` 然后反复 `next(it)`，会在某个点 `StopIteration`，且永远用同一个排列。这是初学者常踩的坑。

---

## 11.9 练习题

### 练习 1：实现 `WeightedRandomSampler`

写一个 `WeightedRandomSampler(data_source, weights, num_samples, replacement=True)`，按 `weights[i]` 的概率抽下标。

??? 解答
    ```python
    class WeightedRandomSampler(Sampler):
        def __init__(self, weights, num_samples, replacement=True):
            self.weights = np.asarray(weights, dtype=np.float64)
            self.num_samples = num_samples
            self.replacement = replacement

        def __iter__(self):
            # numpy 的 choice 支持按权重抽样
            idx = np.random.choice(
                len(self.weights), size=self.num_samples,
                replace=self.replacement, p=self.weights / self.weights.sum(),
            )
            yield from (int(x) for x in idx)

        def __len__(self):
            return self.num_samples
    ```
    关键点：权重归一化（`p` 必须和为 1）；`replacement=False` 时 `num_samples` 不能超过类别数。
???

### 练习 2：让 `DataLoader` 支持直接传 `sampler`

修改 `DataLoader.__init__`，让它接受可选的 `sampler` 参数，传了就用它，没传就按 `shuffle` 生成。注意 `sampler` 和 `shuffle` 互斥。

??? 解答
    ```python
    def __init__(self, dataset, batch_size=1, shuffle=False,
                 sampler=None, drop_last=False, collate_fn=None):
        if sampler is not None and shuffle:
            raise ValueError("sampler 和 shuffle 互斥")
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn or default_collate
        if sampler is not None:
            base = sampler
        elif shuffle:
            base = RandomSampler(dataset)
        else:
            base = SequentialSampler(dataset)
        self.batch_sampler = BatchSampler(base, batch_size, drop_last)
    ```
???

### 练习 3：实现 `random_split`

写 `random_split(dataset, lengths, generator)`，把一个 map-style Dataset 随机切成若干子集（返回 `Subset` 列表）。提示：`Subset(dataset, indices)` 的 `__getitem__` 是 `dataset[indices[i]]`。

??? 解答
    ```python
    class Subset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = list(indices)
        def __getitem__(self, i):
            return self.dataset[self.indices[i]]
        def __len__(self):
            return len(self.indices)

    def random_split(dataset, lengths):
        assert sum(lengths) == len(dataset)
        perm = np.random.permutation(len(dataset))
        subsets, off = [], 0
        for n in lengths:
            subsets.append(Subset(dataset, perm[off:off+n]))
            off += n
        return subsets
    ```
???

### 练习 4：解释为什么 `default_collate` 要递归

用具体例子 `(x, (a, b))` 说明递归的必要性，并写出每步发生了什么。

??? 解答
    样本是嵌套 tuple `(x_i, (a_i, b_i))`。`batch = [(x0,(a0,b0)), (x1,(a1,b1))]`。
    第一次 `zip(*batch)` 转置成 `[(x0,x1), ((a0,b0),(a1,b1))]`。
    - 第一列 `[x0, x1]` 是 ndarray，直接 stack 成 batch_x。
    - 第二列 `[(a0,b0),(a1,b1)]` 又是 tuple 列表，**递归**调用 collate：
      `zip(*[(a0,b0),(a1,b1)])` = `[(a0,a1), (b0,b1)]`，分别 stack。
    最终得到 `[batch_x, [batch_a, batch_b]]`，结构与单个样本对齐。如果不递归，第二层 tuple 就没法处理。
???

### 练习 5：估算多进程预取的加速

假设单进程下每个 batch 取数+解码耗时 50ms，前向+反向+优化器耗时 50ms。开 4 worker 后取数并行进行，理想情况下一个 step 多少 ms？加速比多少？

??? 解答
    单进程：50 + 50 = 100 ms/step。
    4 worker 预取：取数完全隐藏在前向反向里，瓶颈变成 max(50, 50) = 50 ms/step。
    加速比 2x。注意不是 4x——因为预取只能把 I/O 隐藏到计算时间之后，不能让计算本身变快。worker 数超过"取数时间/计算时间"后继续加 worker 没收益。
???

---

## 11.10 关键测试解读

`tests/test_data.py` 里的测试覆盖了所有零件，挑几个重要的讲：

```python
def test_random_sampler_covers_all():
    np.random.seed(42)
    data = list(range(20))
    sampler = RandomSampler(data)
    indices = list(sampler)
    assert sorted(indices) == list(range(20))
```

**解读**：不直接断言 `indices == 某个固定序列`（那样太脆），而是断言**排序后等于 0..19**——即"打乱但全覆盖"。这是无放回采样的核心不变量：每个样本恰好出现一次。

```python
def test_batch_sampler_drop_last():
    sampler = SequentialSampler(list(range(10)))
    bs = BatchSampler(sampler, batch_size=3, drop_last=True)
    batches = list(bs)
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert len(bs) == 3
```

**解读**：10 个样本 batch_size=3，drop_last=True 应该只剩 3 个完整 batch，孤儿 `[9]` 被丢。`len(bs) == 3` 验证 `__len__` 公式正确。

```python
def test_concat_dataset():
    ds1 = ArrayDataset(np.arange(3))
    ds2 = ArrayDataset(np.arange(3, 7))
    cds = ConcatDataset([ds1, ds2])
    assert len(cds) == 7
    assert cds[0][0] == 0
    assert cds[3][0] == 3      # 跨越子集边界
    assert cds[6][0] == 6
```

**解读**：`cds[3]` 是跨边界的关键测试——全局下标 3 应该映射到 `ds2[0]`（值 3）。如果累积大小算错或 `prev` 算错，这里会挂。

```python
def test_dataloader_custom_collate():
    ...
    def my_collate(batch):
        xs = np.stack([b[0] for b in batch])
        ys = np.stack([b[1] for b in batch])
        return xs, ys
    loader = DataLoader(ds, batch_size=2, collate_fn=my_collate)
    bx, by = next(iter(loader))
    assert np.allclose(bx, [[0, 1], [2, 3]])
```

**解读**：验证 `collate_fn` 注入机制工作。用户传了自定义函数，DataLoader 不再用 `default_collate`。这是"开放-封闭"原则的体现——对扩展开放（注入新 collate），对修改封闭（不用改 DataLoader 源码）。

```python
def test_dataloader_shuffle_covers_all():
    np.random.seed(42)
    x = np.arange(20).reshape(10, 2)
    ds = ArrayDataset(x)
    loader = DataLoader(ds, batch_size=3, shuffle=True)
    batches = list(loader)
    all_rows = np.vstack([b[0] for b in batches])
    assert sorted(all_rows[:, 0].tolist()) == list(range(0, 20, 2))
```

**解读**：shuffle 测试不固定具体顺序，而是验证"所有样本都出现且只出现一次"。`x` 的第一列是 `0, 2, 4, ..., 18`，shuffle 后把这些行打乱重排，排序后应该恢复原序。这是无放回采样的不变量测试——比断言"等于某个固定排列"健壮得多，不会因为 numpy 版本变化而挂。

```python
def test_dataloader_len():
    x = np.arange(10).reshape(5, 2)
    ds = ArrayDataset(x)
    loader = DataLoader(ds, batch_size=2)
    assert len(loader) == 3
```

**解读**：5 个样本 batch_size=2，向上取整得 3 个 batch（`[2,2,1]`）。验证 `__len__` 公式 `(n + b - 1) // b` 正确。如果公式写错成 `n // b`，这里会返回 2 而挂。

```python
def test_random_sampler_with_replacement():
    np.random.seed(42)
    data = list(range(10))
    sampler = RandomSampler(data, replacement=True, num_samples=15)
    indices = list(sampler)
    assert len(indices) == 15
```

**解读**：有放回采样 15 个，从 10 个样本里抽。`len(indices) == 15` 验证 `num_samples` 参数生效。注意这里**不**断言"覆盖所有样本"——有放回时可能某些样本没被抽到，断言覆盖会随机挂。只断言数量，是正确的测试姿势。

```python
def test_tensor_dataset():
    x = Tensor.from_numpy(np.arange(12).reshape(4, 3))
    y = Tensor.from_numpy(np.arange(4))
    ds = TensorDataset(x, y)
    assert len(ds) == 4
    sx, sy = ds[1]
    assert np.allclose(sx.numpy(), [3, 4, 5])
    assert sy.item() == 1
```

**解读**：`ds[1]` 应该返回 `(x[1], y[1])`，即第 1 行特征和第 1 个标签。`x[1]` 是 `[3,4,5]`（因为 `arange(12).reshape(4,3)` 第 1 行是 3,4,5），`y[1]` 是 1。这个测试验证**多 Tensor 对齐取索引**——如果 `__getitem__` 里两个 tensor 取了不同索引，这里会挂。

---

## 11.11 优劣势总结

**优势：**

- **三层分离清晰**：Dataset/Sampler/DataLoader 各管一摊，单元测试好写，复用性强。
- **组合性强**：BatchSampler 包任意 Sampler，ConcatDataset 拼任意 Dataset，像积木。
- **map-style 优先**契合大多数训练场景，API 简单。
- `collate_fn` 可注入，对不规则数据留了逃生口。

**劣势：**

- **单进程**：大数据集慢，无法演示真实预取。
- **随机性靠全局 np.random**：多实例间会互相污染，复现性差。
- **没有分布式支持**：多卡训练要自己切数据。
- **collate 覆盖窄**：dict、命名元组、字符串都不支持。
- **没有 `pin_memory`**：真实 GPU 训练里这能省一次拷贝。

**教学价值**：这套实现把"数据管线的骨架"讲透了——所有工程复杂度（多进程、分布式、pin_memory）都是在这个骨架上加肉。理解了这个骨架，去看真实 PyTorch 的 `dataloader.py` 几千行代码就不会迷路。

**何时该跳出教学版**：当你的数据集 > 内存、训练 > 1 小时、或多卡训练时，教学版就不够了——这时直接用真实 PyTorch 的 `DataLoader(num_workers=8, pin_memory=True)` + `DistributedSampler`。但理解了本章，你会知道每个参数在干什么，而不是当黑盒调参。

---

## 11.12 下一章预告

数据喂进模型后，训练出来的参数怎么**存下来**？下一章我们讲**持久化与混合精度**：

- `state_dict` 为什么用扁平的"点号路径"键，而不是嵌套字典？
- `save/load` 怎么把 Tensor 序列化进 pickle，又怎么无损读回来？
- fp16 训练为什么会下溢，`GradScaler` 怎么用"放大 loss"绕过去？
- `Autocast` 怎么按算子分类决定该用 fp16 还是 fp32？
- 一个完整的 AMP 训练循环长什么样？

这些是真正"上 GPU 训练"前必须搞懂的事。

::: tip 阶段小结
至此，minitorch 的数据管线已经完整：Dataset 定义数据、Sampler 定顺序、DataLoader 切批并 collate。下一章我们离开数据，进入"训练产物怎么存"和"怎么用低精度加速训练"这两个工程必修课。掌握本章后，你应当能独立设计一套数据加载方案，并理解每个设计选择背后的权衡。
:::

---

> 本章代码：`src/minitorch/data/dataset.py`、`src/minitorch/data/sampler.py`、`src/minitorch/data/dataloader.py`。测试：`tests/test_data.py`。
