# 第四章 计算图机制：生命周期、释放策略与梯度累加语义

> 前两章我们解决了"建图"和"反向走图"。本章聚焦图本身：
> 它在内存里长什么样？什么时候建、什么时候释放？为什么 `backward` 默认累加梯度？
> 那个让初学者头疼的 "Trying to backward through the graph a second time" 错误又是怎么来的？
> 这些都是计算图机制的核心问题。

---

## 4.1 本章目标

读完本章后，你应当能够：

1. 画出 `z = (x * x) + (x * x)` 的完整计算图，标出每个 `Node` 的 `next_edges` 邻接表。
2. 描述动态图的生命周期：建图（前向）→ 用图（反向）→ 释放图（默认）三个阶段各发生什么。
3. 解释 `retain_graph=True` 为什么允许二次 backward，并说出它的典型应用场景（RNN truncated BPTT）。
4. 区分 `retain_graph` 和 `retain_grad`：前者保留图结构，后者保留中间梯度，两者用途不同。
5. 用代码证明"backward 累加而非覆盖梯度"的语义，并解释这与 `optimizer.zero_grad()` 的关系。
6. 说明图释放策略"清空 `next_edges` 但不清 `grad_fn`"的设计理由。
7. 复现 "Trying to backward through the graph a second time" 错误，并给出三种解决方案。
8. 读懂 `tests/test_graph.py` 里的每一行断言，知道它们在防御哪一类图生命周期 bug。

---

## 4.2 原理铺垫：计算图是什么

### 4.2.1 从数学到数据结构

数学上，计算图是一个有向无环图（DAG）：

- **节点**：每个算子调用是一个节点（`Mul`、`Add`、`Sum`...）。
- **边**：如果算子 A 的输出是算子 B 的输入，就有一条边 A → B。
- **叶子**：用户创建的张量，不是任何算子的输出，没有入边。
- **根**：反向传播的起点（通常是 loss），没有出边。

数据结构上，minitorch 用**邻接表**表示：

```python
class Node:
    backward_fn: Callable          # 这个节点怎么算反向
    next_edges: list[Node | None]  # 指向输入的 grad_fn（前驱节点）
    output: Tensor | None          # 这个节点的输出张量
```

`next_edges` 是"指向前驱"的邻接表。注意方向：**边从输出指向输入**（反向图的方向），
这样反向引擎直接沿 `next_edges` 走就是反向传播方向。

### 4.2.2 画一张具体的图

考虑：

```python
x = Tensor([2.0]); x.requires_grad = True
y = x * x          # Mul
z = y + y          # Add
L = z.sum()        # Sum
```

计算图（用 `next_edges` 表示）：

```
L.grad_fn = Sum
  └─ next_edges: [z.grad_fn]

z.grad_fn = Add
  ├─ next_edges[0]: y.grad_fn (Mul)
  └─ next_edges[1]: y.grad_fn (Mul)   ← 同一个对象，两次出现

y.grad_fn = Mul
  ├─ next_edges[0]: AccumulateGrad(x) 实例 A
  └─ next_edges[1]: AccumulateGrad(x) 实例 B   ← 两个不同对象

A.next_edges = []   ← 叶子没有前驱
B.next_edges = []
A.variable = x
B.variable = x
```

注意三个关键现象：

1. **`z.grad_fn.next_edges` 里 `y.grad_fn` 出现两次，是同一个对象**。
   因为 `Add.apply(y, y)` 第 5 步对中间节点 `y` 复用 `y.grad_fn`。
2. **`y.grad_fn.next_edges` 里是两个不同的 `AccumulateGrad(x)` 实例**。
   因为 `Mul.apply(x, x)` 第 5 步对叶子 `x` 每次新建 `AccumulateGrad`。
3. **`A.variable is B.variable is x`**：两个 AccumulateGrad 指向同一个叶子张量。

### 4.2.3 动态图 vs 静态图

!!! tip "核心区别"
    **静态图**（TF1/Theano）：先编译一张图，反复执行同一张图。
    **动态图**（PyTorch/minitorch）：每次前向**新建一张图**，反向后**丢弃**。

    动态图的含义：

    ```python
    for x, y in dataloader:
    pred = model(x)        # 每次前向都建一张新图
    loss = criterion(pred, y)
    loss.backward()        # 反向走这张图
    optimizer.step()
    optimizer.zero_grad()
    # 图在 backward 后被释放，下一轮前向建新图
    ```

    每个 batch 都是一张全新的图。如果模型里有 `if pred > 0:` 这种控制流，
    不同 batch 的图结构可能不同——动态图天然支持，静态图要 `tf.cond` 等特殊 API。

    代价是：动态图无法做跨 batch 的全局优化（如算子融合），每次前向都有建图开销。
    `torch.compile` 就是想"把动态图静态化"以获得优化机会。

### 4.2.4 图的生命周期三阶段

```
阶段 1：建图（前向）
  每次 Function.apply 都新建 Node，挂到输出.grad_fn
  图随前向计算逐层生长

阶段 2：用图（反向）
  run_backward 沿图走一遍，把梯度填到叶子.grad
  中间节点的梯度默认不保留（retain_grad=False）

阶段 3：释放图
  默认 retain_graph=False：
    清空所有 node.next_edges（切断引用链）
    清入口 tensor.grad_fn
  图被 GC 回收，下一次前向建新图
```

### 4.2.5 为什么默认释放图

三个理由：

1. **内存**：图持有所有中间张量（`saved_tensors`），大模型反向后这些中间值不再需要，留着会 OOM。
2. **正确性**：动态图每个 batch 应该独立，留着旧图容易误用（如改了叶子值后用旧图 backward）。
3. **性能**：GC 回收中间张量，下一个 batch 能复用内存。

代价是：如果想二次 backward（如高阶导数、RNN truncated BPTT），必须显式 `retain_graph=True`。

### 4.2.6 梯度累加语义

`backward` 默认**累加**到 `x.grad`，而非覆盖：

```python
x.grad = x.grad + new_grad   # 而非 x.grad = new_grad
```

为什么？两个场景：

**场景 1：单次反向内的累加**
`y = x * x` 反向时，两个 `AccumulateGrad(x)` 各加一次，`x.grad` 累加成 $2x$。
这是单次 backward 内部的正确性要求。

**场景 2：多次反向间的累加**
```python
for i in range(3):
    loss = model(batch_i)
    loss.backward()    # 不 zero_grad 的话，x.grad 会累加三个 batch 的梯度
```
这对应"**梯度累加**"技巧：小显存时用小 batch 多次 backward，等价于大 batch 的梯度。
要清零就 `optimizer.zero_grad()`（或 `x.grad = None`）。

PyTorch/minitorch 选择累加语义，让"梯度累加"技巧零成本可用。
代价是：用户必须记得 `zero_grad()`，否则梯度会越累越大。

---

## 4.3 设计决策与权衡

| 决策                              | 选择                                       | 理由                                                              | 代价                                                  |
| --------------------------------- | ------------------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------------- |
| 图的表示                          | `Node.next_edges` 邻接表                   | 反向引擎直接沿 edges 走，无需反向建图                             | 查"某节点的所有后继"要遍历全图（但反向不需要这个）   |
| 动态图 vs 静态图                  | 动态图                                     | 控制流天然支持，调试友好                                          | 无跨 batch 全局优化，每次前向有建图开销               |
| 默认是否保留图                    | 不保留（`retain_graph=False`）             | 内存安全，符合动态图"用完即弃"哲学                                | 二次 backward 要显式 `retain_graph=True`              |
| 默认是否保留中间梯度              | 不保留（`retain_grad=False`）              | 中间梯度通常不需要，省内存                                        | 调试可视化时要显式打开                                |
| 梯度语义                          | 累加（`x.grad = x.grad + new`）            | 支持梯度累加技巧；单次反向内多后继累加也靠它                       | 用户必须 `zero_grad()`，否则跨 batch 累加             |
| 图释放策略                        | 清 `next_edges`，不清 `grad_fn`            | 切断引用让 GC 回收；保留 `grad_fn` 让二次 backward 报明确错误      | 用户强引用旧 Tensor 仍可能持有悬挂 Node               |
| 二次 backward 错误                | `grad_fn is None` 抛 RuntimeError          | 提前失败，避免引擎里 NPE                                          | 错误信息要解释清楚原因                                |
| `retain_graph` 保留什么           | 保留 `next_edges` 和 `saved_tensors`       | 二次 backward 需要完整的图结构和中间值                            | 内存占用持续高                                        |
| `retain_grad` 保留什么            | 保留 `node.output.grad`                    | 调试时查看中间梯度分布                                            | 不影响图结构，可独立开关                              |
| 叶子 `grad` 初值                  | `None`，第一次累加时赋值                   | 区分"没反向过"和"反向得 0"                                        | 用户要判 `if x.grad is not None`                       |
| `zero_grad` 实现                  | `x.grad = None`（PyTorch 1.7+）            | 比 `x.grad = 0` 省 内存                                            | minitorch 没内置 optimizer，用户手动清                |

---

## 4.4 代码逐行实现

### 4.4.1 图的表示：`Node` 与 `next_edges`

```python
class Node:
    def __init__(self, backward_fn, next_edges: list, name: str = ""):
        self.backward_fn = backward_fn   # 反向函数
        self.next_edges = next_edges     # 邻接表：指向前驱的 grad_fn
        self.name = name
        self.output: Tensor | None = None  # 输出张量，retain_grad 用
```

`next_edges` 是图的**全部结构信息**。反向引擎靠它做拓扑遍历，
图释放就是清空它，`retain_graph` 就是保留它。

`output` 字段在默认反向里不参与，只有 `retain_grad=True` 时才用：
把中间梯度写回 `node.output.grad`，让用户能查看 `y.grad` 等中间梯度。

### 4.4.2 建图：`Function.apply` 的图生长

```python
@classmethod
def apply(cls, *args, **kwargs) -> Tensor:
    ctx = Context()
    result = cls.forward(ctx, *args, **kwargs)

    if not is_grad_enabled():
        return result

    needs_grad = any(isinstance(a, Tensor) and a.requires_grad for a in args)
    if not needs_grad:
        return result

    next_edges: list[Node | None] = []
    for a in args:
        if isinstance(a, Tensor) and a.requires_grad:
            if a.grad_fn is not None:
                next_edges.append(a.grad_fn)           # 复用中间节点的 grad_fn
            else:
                next_edges.append(AccumulateGrad(a))   # 叶子新建 AccumulateGrad
        else:
            next_edges.append(None)

    node = Node(
        backward_fn=lambda *grads: cls.backward(ctx, *grads),
        next_edges=next_edges,
        name=cls.__name__,
    )
    node.output = result
    result.requires_grad = True
    result.grad_fn = node                               # 把新 node 挂到输出
    return result
```

每次 `apply` 都**新建一个 Node**，挂到输出张量。这就是"图随前向逐层生长"。

关键点回顾：

- **复用中间 `grad_fn`**：`next_edges.append(a.grad_fn)` 让"同一中间张量多次使用"产生多条 edge 指向同一 node。
- **新建 `AccumulateGrad`**：`next_edges.append(AccumulateGrad(a))` 让"同一叶子多次使用"产生多个 AccumulateGrad 实例。
- **`node.output = result`**：保留输出张量引用，供 `retain_grad` 用。这也意味着**只要 node 活着，输出张量就不会被 GC**——这是 `retain_graph` 占内存的根源。

### 4.4.3 用图：`run_backward` 的图遍历

```python
def run_backward(root, root_grad, retain_graph=False, retain_grad=False):
    with no_grad():
        topo = _topological_sort(root)
        grad_map = {id(root): root_grad}

        for node in reversed(topo):
            grad = grad_map.get(id(node))
            if grad is None:
                continue

            if isinstance(node, AccumulateGrad):
                node.backward_fn(grad)                 # 累加到 variable.grad
                continue

            if retain_grad and node.output is not None:
                if node.output.grad is None:           # 保留中间梯度
                    node.output.grad = grad
                else:
                    node.output.grad = node.output.grad + grad

            grads = node.backward_fn(grad)
            if not isinstance(grads, tuple):
                grads = (grads,)

            for edge, g in zip(node.next_edges, grads, strict=True):
                if edge is None or g is None:
                    continue
                prev = grad_map.get(id(edge))
                grad_map[id(edge)] = g if prev is None else prev + g   # 多后继累加

        if not retain_graph:                           # 释放图
            for node in topo:
                node.next_edges = []
```

`retain_grad` 那段是 Ch4 的重点：

```python
if retain_grad and node.output is not None:
    if node.output.grad is None:
        node.output.grad = grad
    else:
        node.output.grad = node.output.grad + grad
```

- **默认 `retain_grad=False`**：跳过这段，中间张量的 `.grad` 保持 `None`，省内存。
- **打开后**：把当前节点的梯度写到 `node.output.grad`，让用户能 `print(y.grad)` 查看中间梯度。
- **累加语义**：和叶子的累加一致，多次 backward 时中间梯度也累加。

### 4.4.4 释放图：`next_edges = []`

```python
if not retain_graph:
    for node in topo:
        node.next_edges = []
```

为什么清 `next_edges` 而不清 `grad_fn` 或 `backward_fn`？

- **清 `next_edges`**：切断 node 对前驱的引用。前驱 node 如果没有别的引用，就可被 GC。
  这递归地释放整张图（除了被外部强引用的 Tensor）。
- **不清 `backward_fn`**：它是闭包，捕获了 `ctx` 和 `cls`。清它也能释放 `ctx.saved_tensors`，
  但 `next_edges` 已切断图结构，`backward_fn` 留着也不会被调用（引擎不会再走到这个 node）。
- **不清 `grad_fn`（挂在 Tensor 上的）**：保留它，让 `tensor.grad_fn is not None` 仍成立，
  下次 `backward` 时由 `variable.py` 入口抛"no grad_fn"错误——等等，这看似矛盾？

看 `variable.py`：

```python
def backward(tensor, gradient=None, retain_graph=False, retain_grad=False):
    ...
    run_backward(tensor.grad_fn, gradient, ...)
    if not retain_graph:
        tensor.grad_fn = None          # 入口的 grad_fn 在这里清
```

所以**入口张量的 `grad_fn` 在 `variable.backward` 里清**，而**中间 node 的 `next_edges` 在 `run_backward` 里清**。
两者分工：

- 中间 node 的 `next_edges` 清了，图结构断了，但中间 Tensor 的 `grad_fn` 字段还指向那个 node（悬挂 node）。
- 入口 Tensor 的 `grad_fn` 显式清 None，让 `tensor.backward()` 第二次调用时立即报错。

中间 Tensor 的 `grad_fn` 为什么不清？因为引擎拿不到所有中间 Tensor 的列表——它只有 node 列表。
要清中间 Tensor 的 `grad_fn`，得在 node 上记 `output` 字段然后清 `node.output.grad_fn`，
但这会破坏 `retain_graph` 的语义（保留图就包括保留 `grad_fn`）。
PyTorch 的做法是：中间 Tensor 的 `grad_fn` 在图释放后仍指向悬挂 node，
但悬挂 node 的 `next_edges` 已空，二次 backward 会走到那个 node 但没有前驱，结果错误。
所以 PyTorch 抛 "Trying to backward through the graph a second time" 在更早的位置——
检测到 `next_edges` 已空就报错。minitorch 简化为只在入口检查 `grad_fn`。

### 4.4.5 `retain_graph`：保留图用于二次 backward

```python
y.backward(retain_graph=True)   # 保留图
y.backward()                    # 可以再次 backward
```

`retain_graph=True` 时，`run_backward` 跳过最后那段清 `next_edges`，
`variable.backward` 也跳过 `tensor.grad_fn = None`。
图完整保留，可以再次 backward。

**典型场景：RNN truncated BPTT**

```python
for t in range(seq_len):
    out = rnn(x[t], h)
    loss = criterion(out, y[t])
    loss.backward(retain_graph=True)   # 保留图，下一步要接着反向
    optimizer.step()
    optimizer.zero_grad()
    if t % truncate == 0:
        # 每 truncate 步释放一次图，避免无限增长
        h = h.detach()
```

RNN 的反向要跨时间步传播，每步 backward 都要保留图给下一步用。
但图不能无限保留（内存爆炸），所以定期 `detach()` 切断。

### 4.4.6 `retain_grad`：保留中间梯度用于调试

```python
x = Tensor([2.0]); x.requires_grad = True
y = x * x
y.backward(retain_grad=True)
print(y.grad)   # Tensor([1.0]) —— 中间梯度被保留
```

默认 `y.grad is None`（中间梯度不保留）。
`retain_grad=True` 让引擎把中间节点的梯度写到 `node.output.grad`。

**典型场景：调试梯度消失/爆炸**

```python
loss.backward(retain_grad=True)
for name, param in model.named_parameters():
    print(name, param.grad.norm())   # 查看每层梯度
# 也能查看中间激活的梯度
for layer in model.layers:
    print(layer.output.grad.norm())  # 看梯度沿网络深度的变化
```

`retain_grad` 不影响图结构，可以和 `retain_graph` 独立开关。

### 4.4.7 梯度累加：`AccumulateGrad._accumulate`

```python
class AccumulateGrad(Node):
    def __init__(self, variable):
        self.variable = variable
        super().__init__(self._accumulate, [], name="AccumulateGrad")

    def _accumulate(self, grad):
        if self.variable.grad is None:
            self.variable.grad = grad                  # 第一次：赋值
        else:
            self.variable.grad = self.variable.grad + grad  # 后续：累加
```

累加语义的两层：

1. **单次 backward 内**：同一叶子多次出现（如 `x * x`），多个 `AccumulateGrad` 各加一次。
2. **多次 backward 间**：用户连续调 `loss.backward()` 不 `zero_grad`，梯度跨调用累加。

两层都用同一个 `_accumulate` 方法，逻辑统一。

**与 `zero_grad` 的关系**：

```python
# 假装是 optimizer.zero_grad()
def zero_grad():
    for p in params:
        p.grad = None    # 或 p.grad = 0
```

因为 backward 累加，所以每个 batch 开始前要 `zero_grad` 清零，
否则 `x.grad` 会包含历史 batch 的梯度，更新方向错误。
PyTorch 1.7+ 把 `zero_grad` 从 `p.grad = 0` 改成 `p.grad = None`，省内存且让下次 backward 走"赋值"分支。

### 4.4.8 `variable.backward` 的入口校验

```python
def backward(tensor, gradient=None, retain_graph=False, retain_grad=False):
    if tensor.grad_fn is None:
        raise RuntimeError(
            "backward() called on a tensor with no grad_fn "
            "(is it a non-leaf or created without requires_grad?)"
        )
    if gradient is None:
        if tensor.size != 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        gradient = Tensor.from_numpy(np.ones(tensor.shape, dtype=tensor.dtype))
    run_backward(tensor.grad_fn, gradient, retain_graph=retain_graph, retain_grad=retain_grad)
    if not retain_graph:
        tensor.grad_fn = None
```

两个校验：

1. **`grad_fn is None`**：tensor 没有 grad_fn，不能 backward。
   - 可能是叶子（用户创建的，没经过算子）。
   - 可能是 `no_grad` 下创建的。
   - 可能是已经 backward 过且没 `retain_graph`（`grad_fn` 被清了）。
   错误信息提示用户检查这三种情况。

2. **非标量且未传 gradient**：标量 loss 的默认梯度是 1，非标量没默认值。
   强制用户传 `gradient`，避免静默错误。

最后 `if not retain_graph: tensor.grad_fn = None`：清入口的 grad_fn，
让二次 backward 立即在第 1 个校验报错。

---

## 4.5 完整示例：图的生命周期可视化

```python
import numpy as np
from minitorch import Tensor

# === 阶段 1：建图 ===
x = Tensor.from_numpy(np.array([2.0]))
x.requires_grad = True
print("建图前: x.grad_fn =", x.grad_fn)             # None（叶子）

y = x * x
print("建图后: y.grad_fn =", y.grad_fn)             # Node(Mul)
print("         y.grad_fn.next_edges =", y.grad_fn.next_edges)
# [Node(AccumulateGrad), Node(AccumulateGrad)]

# === 阶段 2：用图（反向）===
y.backward()
print("反向后: x.grad =", x.grad.tolist())          # [4.0]

# === 阶段 3：释放图（默认）===
print("释放后: y.grad_fn =", y.grad_fn)             # None（被清了）
print("         x.grad 仍保留 =", x.grad.tolist())  # [4.0]（叶子的 grad 不清）

# === 二次 backward 失败 ===
try:
    y.backward()
except RuntimeError as e:
    print("二次 backward 报错:", e)
    # backward() called on a tensor with no grad_fn ...

# === retain_graph=True 的对比 ===
x2 = Tensor.from_numpy(np.array([2.0])); x2.requires_grad = True
y2 = x2 * x2
y2.backward(retain_graph=True)
print("retain_graph 后: y2.grad_fn =", y2.grad_fn)  # Node(Mul)（仍保留）
print("                  x2.grad =", x2.grad.tolist())  # [4.0]

# 再次 backward：梯度累加！
y2.backward()
print("再次 backward 后 x2.grad =", x2.grad.tolist())  # [8.0]（4 + 4）

# === retain_grad=True 的对比 ===
x3 = Tensor.from_numpy(np.array([2.0])); x3.requires_grad = True
y3 = x3 * x3
y3.backward(retain_grad=True)
print("retain_grad 后: y3.grad =", y3.grad.tolist())  # [1.0]（中间梯度保留）

# 对比：默认不 retain_grad
x4 = Tensor.from_numpy(np.array([2.0])); x4.requires_grad = True
y4 = x4 * x4
y4.backward()
print("默认 retain_grad: y4.grad =", y4.grad)         # None
```

预期输出：

```
建图前: x.grad_fn = None
建图后: y.grad_fn = Node(Mul)
         y.grad_fn.next_edges = [Node(AccumulateGrad), Node(AccumulateGrad)]
反向后: x.grad = [4.0]
释放后: y.grad_fn = None
         x.grad 仍保留 = [4.0]
二次 backward 报错: backward() called on a tensor with no grad_fn ...
retain_graph 后: y2.grad_fn = Node(Mul)
                  x2.grad = [4.0]
再次 backward 后 x2.grad = [8.0]
retain_grad 后: y3.grad = [1.0]
默认 retain_grad: y4.grad = None
```

---

## 4.6 常见陷阱

### 陷阱 1：Trying to backward through the graph a second time

```python
x = Tensor([2.0]); x.requires_grad = True
y = x * x
y.backward()
y.backward()   # RuntimeError: backward() called on a tensor with no grad_fn
```

**原因**：第一次 backward 默认 `retain_graph=False`，图被释放，`y.grad_fn` 被清 None。
**解决方案**（三选一）：

1. 第一次用 `retain_graph=True`：
   ```python
   y.backward(retain_graph=True)
   y.backward()   # OK，但注意梯度会累加！
   ```
2. 重新前向建图：
   ```python
   y = x * x
   y.backward()
   y = x * x       # 重新建图
   y.backward()
   ```
3. 用 `detach()` 切断梯度流（如果不需要二次反向）：
   ```python
   y = (x * x).detach()
   # y 不再连到 x，可任意用
   ```

### 陷阱 2：忘记 `zero_grad` 导致梯度累加

```python
x = Tensor([1.0]); x.requires_grad = True
for i in range(3):
    y = (x * 2).sum()
    y.backward()
    print(x.grad.tolist())   # [2.0], [4.0], [6.0] —— 累加！
```

**原因**：`backward` 累加语义，每次都往 `x.grad` 加。
**解决**：每个 batch 开始前 `x.grad = None`（或用未来的 `optimizer.zero_grad()`）。

### 陷阱 3：修改叶子值后用旧图 backward

```python
x = Tensor([2.0]); x.requires_grad = True
y = x * x
x._storage.data[0] = 999.0   # 原地改 storage
y.backward()                  # 梯度用的是新 x 值（999），但图结构是旧的
```

**原因**：动态图的 `saved_tensors` 是引用，改 storage 影响反向结果，但图不会自动重建。
**解决**：不要原地改叶子；要改就重新前向。PyTorch 对叶子原地操作会抛错，minitorch 暂未防御。

### 陷阱 4：`retain_graph` 但不 `zero_grad` 导致梯度翻倍

```python
x = Tensor([2.0]); x.requires_grad = True
y = x * x
y.backward(retain_graph=True)   # x.grad = [4.0]
y.backward()                    # x.grad = [8.0]（累加！）
```

**原因**：`retain_graph` 保留图是为了**再次反向**，但累加语义让第二次的结果加到第一次上。
**解决**：如果想要"独立的第二次反向"，先 `x.grad = None`：

```python
y.backward(retain_graph=True)
x.grad = None
y.backward()
```

### 陷阱 5：在 `no_grad` 下创建的 Tensor 不能 backward

```python
x = Tensor([2.0]); x.requires_grad = True
with no_grad():
    y = x * 2
y.backward()   # RuntimeError: no grad_fn
```

**原因**：`no_grad` 下 `apply` 跳过建图，`y.grad_fn is None`。
**解决**：要建图就别用 `no_grad`。`no_grad` 用于推理或反向内部。

### 陷阱 6：`retain_grad` 但不 `retain_graph`

```python
x = Tensor([2.0]); x.requires_grad = True
y = x * x
y.backward(retain_grad=True)
# y.grad 保留了，但图已释放
# 如果想用 y.grad 做后续运算，没问题
# 但如果想再次 backward，仍会失败
```

`retain_grad` 只保留中间梯度，不保留图。两者独立。混淆会导致"我以为保留了图"的意外。

### 陷阱 7：中间 Tensor 的 `grad_fn` 在图释放后仍非 None

```python
x = Tensor([2.0]); x.requires_grad = True
y = x * x
z = y + 1
z.backward()
print(y.grad_fn)   # 可能仍非 None（中间 node 的 grad_fn 没被显式清）
```

**原因**：`run_backward` 只清 `node.next_edges`，不清 `node.output.grad_fn`。
**影响**：`y.grad_fn` 指向悬挂 node（`next_edges` 已空），二次 backward 会走到它但无前驱。
**解决**：不要依赖中间 Tensor 的 `grad_fn` 判断图是否存活；只信入口 Tensor 的 `grad_fn`。

---

## 4.7 与真实 PyTorch 对照

| minitorch 概念                      | PyTorch 对应                                              | 关键差异                                                                 |
| ----------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------ |
| `Node.next_edges` 邻接表            | `Node::next_edges_` (function.h)                          | 一致；PyTorch 用 `edge_list` C++ 容器                                    |
| 动态图                              | 动态图（define-by-run）                                   | 完全一致；PyTorch 2.0 的 torch.compile 可选静态化                        |
| `retain_graph=False` 默认释放       | 同                                                        | 一致                                                                     |
| 清 `next_edges` 释放图              | `clear_graph` 清 `next_edges_` 和 `saved_tensors_`        | PyTorch 更彻底，minitorch 留 `backward_fn` 闭包                          |
| `retain_graph=True`                 | 同                                                        | 一致；PyTorch 还区分 `create_graph`（建反向图用于高阶导）                |
| `retain_grad=False` 默认             | 同                                                        | 一致；PyTorch 用 `y.retain_grad()` 方法而非参数                          |
| 梯度累加语义                        | 同                                                        | 完全一致；PyTorch 1.7+ `zero_grad` 设 None 而非 0                        |
| 二次 backward 报错                  | "Trying to backward through the graph a second time"      | minitorch 信息更朴素，PyTorch 信息更具体                                 |
| `AccumulateGrad._accumulate`        | `AccumulateGrad::apply`                                   | PyTorch 有原地加法 `at::add_`，minitorch 新建 Tensor                     |
| 中间 Tensor `grad_fn` 释放后        | 仍指向悬挂 Node，但访问会触发错误                         | minitorch 不主动报错，PyTorch 在 backward 入口检测                       |
| `detach()`                          | `Tensor::detach`                                          | minitorch 暂未实现（可手动 `Tensor.from_numpy(t.numpy())` 切断）         |
| 叶子原地操作检测                    | 抛 "a leaf Variable that requires grad is being used in an in-place operation" | minitorch 不检测，允许但结果可能错                                       |

!!! tip "PyTorch 的 `create_graph` vs `retain_graph`"
    PyTorch 的 `backward(create_graph=True)` 会在反向过程中**为反向图本身建图**，
    这样就能对梯度再求梯度（高阶导数）。
    `retain_graph=True` 只是保留前向图，不建反向图。
    两者正交：`retain_graph` 控制"前向图留不留"，`create_graph` 控制"反向图建不建"。
    minitorch 暂不支持 `create_graph`，反向固定在 `no_grad` 下。

    ---

## 4.8 历史背景

### 4.8.1 动态图的诞生

PyTorch 之前，TensorFlow 1.x 和 Theano 用静态图：先 `tf.Graph()` 编译，再 `session.run()` 反复执行。
调试时不能在图中间 `print`，控制流要 `tf.cond` / `tf.while_loop`，体验很差。

PyTorch（2016）受 Chainer 和 HIPS/autograd 启发，选择动态图：
每次前向实时建图，反向后丢弃。`print`、`if`、`for` 都按 Python 原生语义工作，
调试体验接近纯 Python，迅速赢得研究者青睐。

代价是：无法做跨 batch 的全局优化（算子融合、常量折叠），性能不如静态图。
PyTorch 2.0 的 `torch.compile` 就是想"动态图静态化"——用 Dynamo 捕获 Python 控制流，
Inductor 做算子融合，兼顾动态图体验和静态图性能。

### 4.8.2 `retain_graph` 的演化

早期 PyTorch（0.1）每次 backward 都保留图（`retain_graph=True` 是默认）。
这导致内存泄漏：用户不显式释放，图一直留着。
0.2 改成默认释放（`retain_graph=False`），让"用完即弃"成为默认哲学。
`retain_graph=True` 退化为高级用法，用于 RNN truncated BPTT、高阶导数等场景。

### 4.8.3 `zero_grad` 的演化

PyTorch 1.6 之前，`optimizer.zero_grad()` 把 `p.grad` 设为全 0 张量。
1.7+ 改成 `p.grad = None`，下次 backward 时 `AccumulateGrad` 走"赋值"分支而非"加 0"分支，省一次无意义的加法。
minitorch 的 `_accumulate` 已经是这个新形态：`if self.variable.grad is None: self.variable.grad = grad`。

### 4.8.4 "Trying to backward through the graph a second time"

这个经典错误信息从 PyTorch 0.2 就有，是动态图"用完即弃"哲学的必然产物。
错误信息几经修改，从早期的 "Trying to backward through the graph a second time, but the buffers have been freed"
到现在的 "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed)"，
越来越精确地告诉用户原因和解决方案。

minitorch 的信息更朴素："backward() called on a tensor with no grad_fn"，
教学上更直白，但不如 PyTorch 信息 actionable。

### 4.8.5 `detach` 的引入

`detach()` 在 PyTorch 0.4 引入，用于"切断梯度流"——返回一个新 Tensor，共享数据但 `grad_fn=None`。
典型用途：

- RNN truncated BPTT：`h = h.detach()` 切断历史梯度。
- 强化学习的 value function：从 actor 流出的 value 用 `detach()` 不让 critic 的梯度流回 actor。
- GAN 训练：`fake.detach()` 让 discriminator 训练时不更新 generator。

minitorch 暂未实现 `detach`，可用 `Tensor.from_numpy(t.numpy(), requires_grad=False)` 等价模拟。

---

## 4.9 练习题

### 练习 1：画出 `L = ((x * x) + (x * x)).sum()` 的完整计算图

要求：标出每个 Node、next_edges、AccumulateGrad 实例。

??? 解答 ???

```
L.grad_fn = Sum
  └─ next_edges: [Add]

Add (z = y1 + y2)
  ├─ next_edges[0]: Mul (y1)
  └─ next_edges[1]: Mul (y2)

Mul (y1 = x * x)
  ├─ next_edges[0]: AccumulateGrad_A (variable=x)
  └─ next_edges[1]: AccumulateGrad_B (variable=x)

Mul (y2 = x * x)
  ├─ next_edges[0]: AccumulateGrad_C (variable=x)
  └─ next_edges[1]: AccumulateGrad_D (variable=x)

A, B, C, D 是四个不同的 AccumulateGrad 实例，但 .variable 都 is x
```

反向时四个 AccumulateGrad 各加一次，`x.grad` 累加四次。
每次贡献 $x \cdot \text{grad}$，总和 $4x$，与 $\frac{d(2x^2)}{dx} = 4x$ 吻合。

### 练习 2：解释 `retain_graph` 和 `retain_grad` 的区别

要求：各举一个典型用例。

??? 解答 ???

| 参数          | 保留什么                     | 用途                         | 典型场景                       |
| ------------- | ---------------------------- | ---------------------------- | ------------------------------ |
| `retain_graph` | 图结构（`next_edges`、`saved_tensors`） | 允许再次 backward            | RNN truncated BPTT、高阶导数   |
| `retain_grad`  | 中间张量的 `.grad` 字段       | 查看中间梯度分布             | 调试梯度消失/爆炸              |

两者正交：

- `retain_graph=True, retain_grad=False`：能再次 backward，但中间 `.grad` 不保留。
- `retain_graph=False, retain_grad=True`：不能再次 backward，但本次的中间 `.grad` 保留。
- 两者都 True：图和中间梯度都保留，内存占用最大。

### 练习 3：为什么 `backward` 累加而非覆盖梯度

要求：从单次反向和多次反向两个角度解释。

??? 解答 ???

**单次反向内**：`y = x * x` 反向时，两个 `AccumulateGrad(x)` 各加一次。
如果覆盖，第二次会覆盖第一次，`x.grad` 只剩一份，结果错。
累加让两个贡献相加，得到正确的 $2x$。

**多次反向间**：梯度累加技巧——小显存时用小 batch 多次 backward，等价于大 batch：

```python
for micro_batch in big_batch.split(micro_size):
    loss = model(micro_batch)
    loss.backward()   # 不 zero_grad，梯度累加
optimizer.step()      # 等价于大 batch 一步
optimizer.zero_grad()
```

如果 backward 覆盖，这个技巧就不成立，每次 backward 都要立即 step，无法累积。

代价：用户必须记得 `zero_grad()`，否则跨 batch 累加导致更新方向错误。
这是 PyTorch 选择累加语义的代价——让高级技巧零成本可用，初学者多一个必做步骤。

### 练习 4：复现并解决 "Trying to backward a second time" 错误

要求：写代码复现错误，给出三种解决方案。

??? 解答 ???

**复现**：

```python
x = Tensor.from_numpy(np.array([2.0])); x.requires_grad = True
y = x * x
y.backward()
y.backward()   # RuntimeError: backward() called on a tensor with no grad_fn
```

**方案 1：retain_graph**

```python
y.backward(retain_graph=True)
y.backward()   # OK，x.grad = [8.0]（累加）
```

**方案 2：重新前向建图**

```python
y.backward()
y = x * x       # 重新建图
y.backward()    # OK，x.grad = [8.0]（累加）
```

**方案 3：detach 切断**（如果不需要二次反向）

```python
y = (x * x).detach()   # minitorch 暂无 detach，用 Tensor.from_numpy(y.numpy()) 模拟
# y 不再连到 x，可任意用，但 backward 不会有梯度流到 x
```

### 练习 5：图释放为什么不清 `grad_fn`（中间 Tensor 的）

要求：解释设计理由和潜在问题。

??? 解答 ???

**为什么不清**：

1. **引擎拿不到中间 Tensor 列表**：`run_backward` 只有 node 列表，要清中间 Tensor 的 `grad_fn` 得遍历 `node.output.grad_fn = None`，但 `node.output` 可能为 None（如 AccumulateGrad）。
2. **`retain_graph` 的语义**：保留图就包括保留 `grad_fn`，清了就破坏语义。默认 `retain_graph=False` 时清 `next_edges` 已足够切断图结构。
3. **检测二次 backward 的责任在入口**：`variable.backward` 检查入口 Tensor 的 `grad_fn`，中间 Tensor 的 `grad_fn` 不参与检测。

**潜在问题**：中间 Tensor 的 `grad_fn` 仍指向悬挂 node（`next_edges` 已空）。
如果用户拿着中间 Tensor 试图 backward，会走到悬挂 node 但无前驱，结果错误或报错。
PyTorch 在更深层次检测并报 "Trying to backward through the graph a second time"；
minitorch 简化，只在入口检查，中间 Tensor 的二次 backward 可能静默错误。

**实践建议**：不要依赖中间 Tensor 的 `grad_fn` 判断图是否存活；只信入口 Tensor 的 `grad_fn`。

---

## 4.10 关键测试解读

### 4.10.1 `test_graph_freed_after_backward`：默认释放

```python
def test_graph_freed_after_backward():
    x = Tensor.from_numpy(np.array([2.0])); x.requires_grad = True
    y = x * x
    y.backward()
    assert y.grad_fn is None
```

验证默认 `retain_graph=False`：backward 后入口 Tensor 的 `grad_fn` 被清 None。
这是"用完即弃"哲学的核心断言。

### 4.10.2 `test_retain_graph_allows_second_backward`：保留图允许二次 backward

```python
def test_retain_graph_allows_second_backward():
    x = Tensor.from_numpy(np.array([2.0])); x.requires_grad = True
    y = x * x
    y.backward(retain_graph=True)
    assert x.grad.tolist() == [4.0]
    y.backward()
    assert x.grad.tolist() == [8.0]    # 累加！4 + 4
```

验证 `retain_graph=True` 保留图，可再次 backward。
**关键**：第二次 backward 的梯度**累加**到第一次的结果上（`4 + 4 = 8`），
而非覆盖。这同时测了 `retain_graph` 和累加语义。

### 4.10.3 `test_no_grad_skips_graph`：no_grad 不建图

```python
def test_no_grad_skips_graph():
    x = Tensor.from_numpy(np.array([1.0, 2.0])); x.requires_grad = True
    with no_grad():
        y = x + 1
        assert y.grad_fn is None
        assert not y.requires_grad
```

验证 `no_grad` 上下文里 `apply` 跳过建图：`y.grad_fn is None` 且 `y.requires_grad is False`。
这是推理场景和反向内部的基础保证。

### 4.10.4 `test_retain_grad`：保留中间梯度

```python
def test_retain_grad():
    x = Tensor.from_numpy(np.array([2.0])); x.requires_grad = True
    y = x * x
    y.backward(retain_grad=True)
    assert y.grad is not None
    assert y.grad.item() == 1.0
```

验证 `retain_grad=True` 让中间 Tensor `y` 保留梯度。
`y.grad == 1.0` 因为 `y` 是反向的入口（标量默认梯度 1），它的"梯度"就是 root_grad。

### 4.10.5 `test_default_no_retain_grad`：默认不保留中间梯度

```python
def test_default_no_retain_grad():
    x = Tensor.from_numpy(np.array([2.0])); x.requires_grad = True
    y = x * x
    y.backward()
    assert y.grad is None
```

验证默认 `retain_grad=False`：中间 Tensor 的 `grad` 保持 None，省内存。
与 `test_retain_grad` 对照，确认默认行为。

### 4.10.6 `test_retain_graph_keeps_grad_fn`：retain_graph 保留 grad_fn

```python
def test_retain_graph_keeps_grad_fn():
    x = Tensor.from_numpy(np.array([2.0])); x.requires_grad = True
    y = x * x
    y.backward(retain_graph=True)
    assert y.grad_fn is not None
```

验证 `retain_graph=True` 不清入口 Tensor 的 `grad_fn`，与 `test_graph_freed_after_backward` 对照。
这是"允许二次 backward"的前提——`grad_fn` 还在，下次 backward 入口校验能通过。

### 4.10.7 `test_gradient_argument`：显式传梯度

```python
def test_gradient_argument():
    x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0])); x.requires_grad = True
    y = x * x
    g = Tensor.from_numpy(np.array([1.0, 0.0, 0.0]))
    y.backward(g)
    assert x.grad.tolist() == [2.0, 0.0, 0.0]
```

验证非标量 backward 显式传梯度：`g = [1, 0, 0]` 表示只对 `y[0]` 求梯度。
结果 `x.grad = 2x * g = [2*1*1, 2*2*0, 2*3*0] = [2, 0, 0]`。
这测了 `variable.backward` 的 `gradient` 参数处理和 `Mul.backward` 的逐元素乘法。

---

## 4.11 优劣势总结

### 优势

- **动态图体验**：控制流天然支持，调试友好，与 Python 心智模型一致。
- **内存安全**：默认释放图，大模型训练不会因图累积 OOM。
- **梯度累加零成本**：小 batch 训练技巧无需额外 API。
- **`retain_graph` / `retain_grad` 灵活**：高级场景（RNN BPTT、调试）可选保留。

### 代价

- **无跨 batch 优化**：每个 batch 独立建图，无法算子融合、常量折叠。
- **建图开销**：每次前向都有建图成本，小模型上显著。
- **二次 backward 易错**：默认释放导致用户频繁遇到 "second time" 错误。
- **中间 Tensor `grad_fn` 悬挂**：释放后仍非 None，可能误导用户。
- **无 `create_graph`**：不支持高阶导数。
- **无 `detach`**：不能方便地切断梯度流（RNN、GAN 场景受限）。

### 适用场景

- 教学与小规模实验：动态图体验最佳。
- 研究：控制流、动态结构友好。
- 不适用：生产高性能训练（用 PyTorch + torch.compile）、需要高阶导数（用 `torch.func`）。

---

## 4.12 下一章预告

本章我们讲清了计算图的生命周期。至此，自动微分的三章（建图 / 反向 / 图管理）闭环：

- **Ch2**：算子怎么建图（`Function.apply` 贴面单）。
- **Ch3**：反向引擎怎么走图（`run_backward` 拓扑 + 累加）。
- **Ch4**：图本身的生命周期（建 / 用 / 释放 / 保留）。

下一章将进入**神经网络模块层**：

1. `nn.Module`：所有模型的基类，`__call__` → `forward` 的转发，`parameters()` 递归收集。
2. `nn.Parameter`：特殊的叶子 Tensor，自动 `requires_grad=True`，被 Module 注册。
3. `nn.Linear`：第一个具体层，权重 + 偏置 + 前向。
4. `nn.Sequential`：容器，串联多个层。
5. `optim.SGD` / `optim.Adam`：优化器，用 `.grad` 更新 `.data`。
6. 训练循环：`zero_grad` → `forward` → `loss` → `backward` → `step` 的完整流程。

读完下一章，你就能用 minitorch 训练你的第一个神经网络。
