# 第三章 自动微分引擎：从链式法则到 `run_backward`

> 第二章我们解决了"算子怎么建图"——前向计算时每个可微算子都顺手在输出上贴了 `grad_fn` 面单。
> 本章回答下一个问题：**给定一张建好的计算图和输出端的梯度，怎么把梯度沿图反向传回去？**
> 这就是 `engine.py` 里 `run_backward` 的职责。

---

## 3.1 本章目标

读完本章后，你应当能够：

1. 用 Jacobian 矩阵形式写出反向模式自动微分的数学定义，并解释它为什么比正向模式更适合神经网络。
2. 手算一个三步复合函数 $L = (x^2 + x^2).sum()$ 的反向过程，写出每一步的梯度。
3. 解释拓扑排序的"DFS 后序逆序"为什么是正确的反向顺序，并能手画一张图给出排序结果。
4. 逐行讲清 `run_backward` 的 6 个阶段：建图入口、拓扑排序、初始化 grad_map、按序遍历、梯度分发、释放图。
5. 说明 `grad_map[id(edge)] = g if prev is None else prev + g` 这一行在多后继节点场景下的作用。
6. 解释为什么整个反向过程要包在 `with no_grad():` 里，否则会发生什么。
7. 区分 `AccumulateGrad` 与中间 `Node` 在引擎里的不同处理路径。
8. 读懂 `tests/test_autograd.py` 里的所有断言，包括数值梯度对照测试。

---

## 3.2 原理铺垫：反向模式自动微分

### 3.2.1 从单变量到多变量链式法则

单变量链式法则你已经熟悉：

$$
\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}
$$

但神经网络里 $x$ 和 $y$ 都是**向量**（张量）。设 $x \in \mathbb{R}^n$、$y \in \mathbb{R}^m$、$L \in \mathbb{R}$，
中间映射 $y = f(x)$。链式法则的向量形式是：

$$
\frac{\partial L}{\partial x_i} = \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}
$$

写成矩阵形式：

$$
\nabla_x L = J_f^\top \cdot \nabla_y L
$$

其中 $J_f \in \mathbb{R}^{m \times n}$ 是 $f$ 的 Jacobian，$J_{ji} = \frac{\partial y_j}{\partial x_i}$。

**关键观察**：要从 $\nabla_y L$ 算出 $\nabla_x L$，我们**不需要显式构造 $J_f$**，
只需要能计算"给定 $\nabla_y L$，输出 $J_f^\top \nabla_y L$"这个线性映射。
这正是 `backward(ctx, grad_output)` 做的事——它直接给出 $J_f^\top \nabla_y L$，不构造 $J$。

### 3.2.2 反向模式 vs 正向模式

考虑一个 $n$ 输入、$1$ 输出（loss）的网络。

| 模式     | 每次 forward/backward 算什么       | 总开销              | 适合场景               |
| -------- | ---------------------------------- | ------------------- | ---------------------- |
| 正向模式 | $(x, \dot{x}) \mapsto (y, \dot{y})$，$\dot{y} = J_f \dot{x}$ | $n$ 次前向 | 对每个输入都要一次     |
| 反向模式 | $y$ 算完后，给定 $\bar{y}$ 算 $\bar{x} = J_f^\top \bar{y}$  | $1$ 次前向 + $1$ 次反向 | 单输出（loss）远更高效 |

神经网络训练正是"百万参数（输入）、单 loss（输出）"，所以反向模式远更高效。
代价是反向模式必须**先做完前向、存好中间值**（tape），才能开始反向——内存开销比正向大。

### 3.2.3 为什么需要拓扑排序

反向模式要求"**算某个节点的梯度时，它所有的下游节点都已经算完**"。
这正是拓扑排序的定义：对 DAG 的节点排一个序，使每条边 $(u, v)$ 满足 $u$ 排在 $v$ 前面。

反向传播里"边"的方向是"梯度流动方向"——从输出流向输入。
所以我们要的拓扑序是：**输出在前，输入在后**。
等价地，按"原图"（前向图）的**逆拓扑序**遍历，也就是"输出到输入"的顺序。

!!! tip "直觉"
    前向图：输入 → ... → 输出（一条条边指向下游）。
    反向图：把所有边反向，输出 → ... → 输入。
    反向传播就是沿反向图走一遍，走法是"先算下游、再算上游"——也就是反向图的拓扑序。

### 3.2.4 DFS 后序 = 拓扑序

对一张 DAG，从某节点出发做 DFS，**后序遍历**（先递归子节点，再访问自己）得到的序列，
**逆序**就是一个拓扑序。

证明（直觉版）：后序遍历里，节点在自己所有"子节点"之后被访问。
对前向图做 DFS 后序，得到"输入在前、输出在后"的序列；
逆序就是"输出在前、输入在后"——正是反向传播要的顺序。

minitorch 的 `_topological_sort` 用的就是这个方法。

### 3.2.5 多后继节点的梯度累加

如果一个节点 $v$ 在图里有多个下游用途（如 `z = v + v`），那么 $v$ 的梯度是**所有下游传来的梯度之和**：

$$
\bar{v} = \sum_{\text{下游 } w} \bar{w} \cdot \frac{\partial w}{\partial v}
$$

这就是 `grad_map[id(edge)] = g if prev is None else prev + g` 这一行在做的事：
同一个 edge 可能被多个下游节点往 `grad_map` 里写，每次都要**累加**。

### 3.2.6 一个完整例子：手算反向

考虑：

```python
x = Tensor([1.0, 2.0]); x.requires_grad = True
y = x * x          # y = x^2
z = y + y          # z = 2 * y = 2 * x^2
L = z.sum()        # L = sum(z) = 2 * (1^2 + 2^2) = 10
L.backward()
```

数学上 $\frac{\partial L}{\partial x_i} = 4 x_i$，所以 `x.grad` 应为 `[4.0, 8.0]`。

反向步骤：

1. **入口**：`L.grad_fn = Sum`，`root_grad = [1.0]`（标量默认全 1）。
2. **Sum 节点**：`backward(grad=[1.0])` 返回 `(broadcast_to([1.0], (2,)) = [1.0, 1.0],)`。
   梯度传给 `z`：`grad_map[z.grad_fn] = [1.0, 1.0]`。
3. **Add 节点（z = y + y）**：`backward([1.0, 1.0])` 返回 `([1.0, 1.0], [1.0, 1.0])`。
   两条 edge 都指向 `y.grad_fn`，所以两次都往 `grad_map[y.grad_fn]` 累加：
   - 第一次：`grad_map[y.grad_fn] = [1.0, 1.0]`
   - 第二次：`grad_map[y.grad_fn] = [1.0, 1.0] + [1.0, 1.0] = [2.0, 2.0]`
4. **Mul 节点（y = x * x）**：`backward([2.0, 2.0])` 返回 `(grad*b, grad*a) = ([2.0, 2.0]*x, [2.0, 2.0]*x) = ([2.0, 4.0], [2.0, 4.0])`。
   两条 edge 是两个不同的 `AccumulateGrad(x)` 实例，分别累加到 `x.grad`：
   - 第一次：`x.grad = [2.0, 4.0]`
   - 第二次：`x.grad = [2.0, 4.0] + [2.0, 4.0] = [4.0, 8.0]`
5. **结果**：`x.grad = [4.0, 8.0]`，与 $4x$ 吻合。

注意第 3 步的"累加"和第 4 步的"两个 AccumulateGrad 各加一次"是**两种不同的累加**：
- 第 3 步是 `grad_map` 里同一个 key 多次写入（多后继节点的梯度求和）。
- 第 4 步是 `AccumulateGrad._accumulate` 里 `x.grad` 多次累加（同一叶子多次出现的梯度求和）。

两者数学上等价（都是全图对 $x$ 的总梯度），但代码路径不同。

---

## 3.3 设计决策与权衡

| 决策                              | 选择                                       | 理由                                                              | 代价                                                  |
| --------------------------------- | ------------------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------------- |
| 拓扑排序算法                      | DFS 后序逆序                               | 实现最短，天然递归；只遍历从 root 可达的节点，不浪费              | 递归深时栈溢出风险（PyTorch 改用迭代 + 显式栈）       |
| 梯度存储                          | `dict[int, Tensor]`，键是 `id(node)`       | 通用，支持任意节点；查询 O(1)                                    | 用 `id()` 作键要求 node 不能被 GC（反向期间确实不能） |
| 多后继累加                        | `prev + g`                                 | 语义清晰，直接复用 `Tensor.__add__`                              | 每次累加都新建 Tensor，无原地优化                     |
| 反向是否建图                      | `with no_grad():` 包裹                     | 反向里的中间运算（如 `grad * b`）不应再建图，否则图爆炸           | 用户在反向中手动建图需 `enable_grad` 嵌套             |
| 单线程 vs 多线程                  | 单线程                                     | 教学优先，逻辑清晰                                                | 大图反向慢；PyTorch 用 ReadyQueue 多线程并行无依赖节点 |
| `AccumulateGrad` 如何处理         | `isinstance` 判断 + 调 `backward_fn`       | 复用 `Node.backward_fn` 接口，不特判                              | 多一次 `isinstance` 开销                              |
| `retain_grad` 默认值              | `False`                                    | 中间梯度通常不需要，省内存                                        | 调试时要显式打开                                      |
| 图释放策略                        | 清空 `next_edges`，不清 `grad_fn`          | 切断图引用，让 GC 回收；保留 `grad_fn` 字段以便报"二次 backward"  | 用户若强引用旧 Tensor，仍可能持有悬挂 Node            |
| 标量默认梯度                      | `ones_like`                                | 与 PyTorch 一致；标量 loss 的隐式梯度就是 1                       | 非标量必须显式传 `gradient`，否则报错                 |
| `backward` 入口校验               | `grad_fn is None` 抛 RuntimeError          | 提前失败，避免引擎里 NPE                                          | 错误信息要写得清楚（"是否非叶子/未 requires_grad"）   |

---

## 3.4 代码逐行实现

### 3.4.1 `grad_mode.py`：全局开关

```python
_grad_enabled: bool = True              # 模块级全局变量，默认开

def is_grad_enabled() -> bool:
    return _grad_enabled                # apply 里查这个开关

class no_grad:
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled      # 保存进入前的状态
        _grad_enabled = False           # 关掉
        return self

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._prev      # 恢复（不是简单设 True，支持嵌套）

class enable_grad:
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = True
        return self

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._prev
```

几个细节：

- **用模块级全局变量**而非线程局部。PyTorch 用 `thread_local` 让多线程独立，
  minitorch 单线程所以简化。代价是：多线程下 `no_grad` 会互相影响——但 minitorch 不多线程。
- **`self._prev` 保存旧状态**：支持嵌套，如 `with no_grad(): with no_grad(): ...`，
  退出时恢复到进入前的状态，而不是无脑设 True。
- **`*exc` 吞掉异常信息**：即使块内抛异常，`__exit__` 仍会恢复全局变量，避免状态泄漏。
  不返回 True，所以异常会正常向外抛。

### 3.4.2 `engine.py`：拓扑排序

```python
def _topological_sort(root: Node) -> list[Node]:
    topo: list[Node] = []
    visited: set[int] = set()                # 用 id(node) 去重，避免 __eq__ 误判

    def dfs(node: Node | None) -> None:
        if node is None or id(node) in visited:
            return                           # 空节点或已访问，跳过
        visited.add(id(node))
        for edge in node.next_edges:         # 先递归所有前驱
            dfs(edge)
        topo.append(node)                    # 后序：自己加在子节点之后

    dfs(root)
    return topo
```

逐行解读：

- `visited` 用 `set[int]` 而非 `set[Node]`：Node 没定义 `__hash__`，默认按 `id` 哈希，
  但显式用 `id(node)` 更明确，也避免 Node 定义 `__eq__` 时出 bug。
- `dfs(edge)` 对每个 edge 递归：edge 是 `Node | None`，None 在第一行被短路。
- **后序**：先递归再 append，保证子节点在父节点之前进入 `topo`。
  所以 `topo` 是"输入在前、输出在后"，`reversed(topo)` 才是反向传播要的"输出在前、输入在后"。
- **复杂度**：$O(V + E)$，每个节点和边各访问一次。

!!! warning "递归深度"
    Python 默认递归深度 1000。如果计算图深度超过 1000（如长 RNN），会栈溢出。
    PyTorch 用显式栈迭代避免这个问题。minitorch 教学版保留递归以便阅读。

### 3.4.3 `engine.py`：`run_backward` 主循环

```python
def run_backward(
    root: Node,
    root_grad: Tensor,
    retain_graph: bool = False,
    retain_grad: bool = False,
) -> None:
    with no_grad():                                              # 1. 整个反向不建图
        topo = _topological_sort(root)                           # 2. 拓扑排序
        grad_map: dict[int, Tensor | None] = {id(root): root_grad}  # 3. 初始化：root 的梯度

        for node in reversed(topo):                              # 4. 按反拓扑序遍历
            grad = grad_map.get(id(node))
            if grad is None:                                     #    没有梯度（未被任何下游用到）
                continue

            if isinstance(node, AccumulateGrad):                 # 5. 叶子节点：直接累加到 .grad
                node.backward_fn(grad)
                continue

            if retain_grad and node.output is not None:          # 6. 可选：保存中间梯度
                if node.output.grad is None:
                    node.output.grad = grad
                else:
                    node.output.grad = node.output.grad + grad

            grads = node.backward_fn(grad)                       # 7. 算局部反向
            if not isinstance(grads, tuple):
                grads = (grads,)

            for edge, g in zip(node.next_edges, grads, strict=True):  # 8. 分发梯度到前驱
                if edge is None or g is None:
                    continue
                prev = grad_map.get(id(edge))
                grad_map[id(edge)] = g if prev is None else prev + g  # 9. 累加

        if not retain_graph:                                     # 10. 释放图
            for node in topo:
                node.next_edges = []
```

逐段解读 10 步：

**第 1 步：`with no_grad():`。**
反向过程中会调用各算子的 `backward`，里面要做 `grad * b` 之类的张量运算。
如果这些运算也建图，反向过程会**在反向图上再建一张反向图**，无限膨胀。
`no_grad` 让 `apply` 第 3 步短路，反向里的运算纯数值。

**第 2 步：拓扑排序。**
`topo` 是"输入在前、输出在后"，`reversed(topo)` 是"输出在前、输入在后"。
从 root（输出）开始，逐步往输入走。

**第 3 步：初始化 `grad_map`。**
只有 root 有初始梯度（用户传入或标量默认 1）。其他节点的梯度在反向过程中逐步填入。
`grad_map` 的键是 `id(node)`——用 id 而非 node 本身，避免 Node `__eq__` 干扰。

**第 4 步：按反拓扑序遍历。**
`reversed(topo)` 保证：处理某个 node 时，它所有"下游"（在 `topo` 里排在它之后）已经处理过，
`grad_map[id(node)]` 已经被填好了。

**第 5 步：叶子节点短路。**
`AccumulateGrad` 的 `backward_fn` 就是 `_accumulate`，把梯度加到 `variable.grad`。
注意这里 `continue`，跳过第 6-9 步——叶子没有更上游，不需要继续分发。

**第 6 步：可选 `retain_grad`。**
默认中间节点的 `.grad` 是 `None`（省内存）。打开 `retain_grad` 后，
中间张量也会保存梯度，用于调试可视化（见 Ch4）。

**第 7 步：算局部反向。**
`node.backward_fn(grad)` 调的是 `apply` 里那个闭包 `lambda *grads: cls.backward(ctx, *grads)`，
等价于 `cls.backward(ctx, grad)`。返回值是元组，长度等于 `forward` 输入数。
`if not isinstance(grads, tuple): grads = (grads,)` 是防御性代码，
允许 backward 只返回单个 Tensor 而不包装成元组。

**第 8 步：分发梯度。**
`zip(node.next_edges, grads, strict=True)` 把"第 i 个 edge"和"第 i 个梯度"配对。
`strict=True` 要求两者长度一致——这是防御性断言：
如果 backward 返回的梯度数和 next_edges 长度不符，说明算子写错了，立即报错而非静默错位。

**第 9 步：累加。**
`prev = grad_map.get(id(edge))`：这个 edge 之前可能已经被别的下游传过梯度了。
- `prev is None`：第一次，直接赋值 `g`。
- `prev is not None`：多后继场景，累加 `prev + g`。

这就是"多后继节点的梯度求和"在代码里的落点。

**第 10 步：释放图。**
默认 `retain_graph=False`，反向结束后清空所有 node 的 `next_edges`。
这切断了图的引用链，让 Node 及其捕获的 ctx（含 saved_tensors）可被 GC 回收。
不清 `grad_fn` 字段——保留它，让用户第二次调 `backward` 时能拿到 `grad_fn is not None` 的明确信号，
由 `variable.py` 里的入口抛"Trying to backward through the graph a second time"。

### 3.4.4 `variable.py`：`backward` 入口

```python
def backward(
    tensor: Tensor,
    gradient: Tensor | None = None,
    retain_graph: bool = False,
    retain_grad: bool = False,
) -> None:
    if tensor.grad_fn is None:                                  # 1. 入口校验
        raise RuntimeError(
            "backward() called on a tensor with no grad_fn "
            "(is it a non-leaf or created without requires_grad?)"
        )
    if gradient is None:                                        # 2. 标量默认梯度
        if tensor.size != 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        gradient = Tensor.from_numpy(np.ones(tensor.shape, dtype=tensor.dtype))
    run_backward(                                               # 3. 调引擎
        tensor.grad_fn, gradient, retain_graph=retain_graph, retain_grad=retain_grad
    )
    if not retain_graph:                                        # 4. 清入口的 grad_fn
        tensor.grad_fn = None
```

逐行解读：

**第 1 步：校验 `grad_fn`。**
如果 `tensor.grad_fn is None`，说明它要么是叶子（用户创建的，没有 grad_fn），
要么是 `no_grad` 下创建的，要么是已经 backward 过一次且没 retain_graph。
三种情况都不能 backward，抛错。错误信息提示用户检查"是否非叶子/未 requires_grad"。

**第 2 步：标量默认梯度。**
PyTorch 允许对标量 loss 调 `loss.backward()` 不传梯度，默认梯度是 1。
minitorch 复刻这个行为：`tensor.size != 1` 时强制要求传 `gradient`。
非标量 backward 必须显式传梯度，因为"非标量的隐式梯度"数学上没定义。

**第 3 步：调引擎。**
把 `tensor.grad_fn`（图的入口 node）和初始梯度传给 `run_backward`。

**第 4 步：清入口的 `grad_fn`。**
`run_backward` 内部清了所有中间 node 的 `next_edges`，但入口 node 的 `grad_fn` 还挂在 `tensor` 上。
这里把它清掉，让 `tensor` 也变成"无 grad_fn"状态，下次再 backward 会触发第 1 步的校验报错。
如果 `retain_graph=True`，保留 `grad_fn`，允许再次 backward（见 Ch4）。

---

## 3.5 完整示例：走一遍 `run_backward`

```python
import numpy as np
from minitorch import Tensor

x = Tensor.from_numpy(np.array([2.0]))
x.requires_grad = True
y = x * x          # Mul
z = y + y          # Add
L = z.sum()        # Sum
L.backward()
print(x.grad.tolist())   # [8.0]
```

数学：$L = \text{sum}(2 x^2) = 2 x^2$，$\frac{dL}{dx} = 4x = 8$。

### 3.5.1 前向建图后的状态

```
x (叶子, grad=None, grad_fn=None)
  │
  ├── AccumulateGrad_x_A ──┐
  │                         ├── Mul ──> y (grad_fn=Mul)
  └── AccumulateGrad_x_B ──┘
                            │
                            ├── Add ──> z (grad_fn=Add, next_edges=[y.grad_fn, y.grad_fn])
                            │
                            └── Sum ──> L (grad_fn=Sum, next_edges=[z.grad_fn])
```

注意：
- `Mul.apply(x, x)` 创建了**两个** `AccumulateGrad(x)` 实例 A 和 B。
- `Add.apply(y, y)` 的 `next_edges` 是 `[y.grad_fn, y.grad_fn]`——**同一个对象**出现两次。

### 3.5.2 `_topological_sort(L.grad_fn)` 的执行

从 `Sum` 节点开始 DFS：

```
dfs(Sum)
  dfs(z.grad_fn = Add)
    dfs(y.grad_fn = Mul)
      dfs(AccumulateGrad_x_A) → topo = [A]
      dfs(AccumulateGrad_x_B) → topo = [A, B]
    topo = [A, B, Mul]
  topo = [A, B, Mul, Add]
topo = [A, B, Mul, Add, Sum]
```

`reversed(topo) = [Sum, Add, Mul, A, B]`——正是反向传播的顺序。

### 3.5.3 主循环逐步

**初始**：`grad_map = {id(Sum): Tensor([1.0])}`（标量默认梯度）。

**第 1 次迭代：node = Sum**
- `grad = grad_map[id(Sum)] = [1.0]`
- 不是 AccumulateGrad，跳过第 5 步
- `retain_grad=False`，跳过第 6 步
- `grads = Sum.backward([1.0])`：返回 `(broadcast_to([1.0], (1,)) = [1.0],)`
- 分发：`edge = Add`，`g = [1.0]`，`grad_map[id(Add)] = [1.0]`

**第 2 次迭代：node = Add**
- `grad = grad_map[id(Add)] = [1.0]`
- `grads = Add.backward([1.0])`：返回 `([1.0], [1.0])`（Add 对两个输入的导数都是 1）
- 分发：
  - `edge = y.grad_fn = Mul`，`g = [1.0]`，`prev = None` → `grad_map[id(Mul)] = [1.0]`
  - `edge = y.grad_fn = Mul`（同一个），`g = [1.0]`，`prev = [1.0]` → `grad_map[id(Mul)] = [1.0] + [1.0] = [2.0]`
- **关键**：两次都写到同一个 `id(Mul)`，第二次触发累加。

**第 3 次迭代：node = Mul**
- `grad = grad_map[id(Mul)] = [2.0]`
- `grads = Mul.backward([2.0])`：`a, b = x, x = [2.0], [2.0]`，返回 `([2.0]*[2.0], [2.0]*[2.0]) = ([4.0], [4.0])`
- 分发：
  - `edge = A`（AccumulateGrad_x_A），`g = [4.0]`，`grad_map[id(A)] = [4.0]`
  - `edge = B`（AccumulateGrad_x_B），`g = [4.0]`，`grad_map[id(B)] = [4.0]`

**第 4 次迭代：node = A（AccumulateGrad）**
- `grad = grad_map[id(A)] = [4.0]`
- `isinstance(node, AccumulateGrad)` 为真，调 `node.backward_fn([4.0])` 即 `_accumulate([4.0])`
- `x.grad is None` → `x.grad = [4.0]`
- `continue`，跳过后续

**第 5 次迭代：node = B（AccumulateGrad）**
- `grad = grad_map[id(B)] = [4.0]`
- 调 `_accumulate([4.0])`
- `x.grad is not None` → `x.grad = [4.0] + [4.0] = [8.0]`
- **关键**：两个 AccumulateGrad 各加一次，最终 `x.grad = [8.0]`

**结果**：`x.grad = [8.0]`，与 $4x = 8$ 吻合。

### 3.5.4 图释放

`retain_graph=False`，遍历 `topo` 清空所有 `next_edges`：

```
Sum.next_edges = []
Add.next_edges = []
Mul.next_edges = []
A.next_edges = []   # 本来就是 []
B.next_edges = []
```

之后 `variable.backward` 把 `L.grad_fn = None`。

---

## 3.6 常见陷阱

### 陷阱 1：对非标量调 `backward()` 不传梯度

```python
x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
x.requires_grad = True
y = x * x          # y 是向量，shape (3,)
y.backward()       # RuntimeError: grad can be implicitly created only for scalar outputs
```

**原因**：非标量的"默认梯度"数学上没定义（梯度应是同形向量，但默认值是什么？）。
**解决**：要么 `y.sum().backward()`（标量化），要么 `y.backward(some_vector)`。

### 陷阱 2：对叶子直接 backward

```python
x = Tensor.from_numpy(np.array([1.0]))
x.requires_grad = True
x.backward()       # RuntimeError: backward() called on a tensor with no grad_fn
```

**原因**：叶子没有 `grad_fn`（它不是算子的输出）。
**解决**：叶子是 backward 的**终点**（接收梯度），不是**起点**。从算子的输出开始 backward。

### 陷阱 3：在 `no_grad` 下建图后 backward

```python
x = Tensor.from_numpy(np.array([1.0])); x.requires_grad = True
with no_grad():
    y = x * 2     # y.grad_fn is None（no_grad 下不建图）
y.backward()      # RuntimeError: no grad_fn
```

**原因**：`no_grad` 让 `apply` 跳过建图，`y` 没有 `grad_fn`。
**解决**：要建图就别用 `no_grad`；推理用 `no_grad`，训练不用。

### 陷阱 4：忘记 `retain_graph` 导致二次 backward 失败

```python
x = Tensor.from_numpy(np.array([2.0])); x.requires_grad = True
y = x * x
y.backward()
y.backward()       # RuntimeError: no grad_fn（第一次 backward 后 y.grad_fn 被清了）
```

**原因**：默认 `retain_graph=False`，反向后图被释放。
**解决**：第一次 `y.backward(retain_graph=True)`。详见 Ch4。

### 陷阱 5：修改叶子值后用旧图 backward

```python
x = Tensor.from_numpy(np.array([2.0])); x.requires_grad = True
y = x * x
x._storage.data[0] = 999.0   # 直接改 storage，没重新建图
y.backward()                  # 梯度用的是新 x 值，但图里 saved_tensors 可能是旧的
```

**原因**：动态图的 saved_tensors 是引用，改 storage 会影响反向结果，但图结构不会自动更新。
**解决**：不要原地改叶子；要改就重新前向建图。PyTorch 对此会抛 `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation`。

### 陷阱 6：递归深度溢出

```python
x = Tensor.from_numpy(np.array([1.0])); x.requires_grad = True
y = x
for _ in range(2000):
    y = y * 2     # 图深度 2000
y.sum().backward()   # RecursionError: maximum recursion depth exceeded
```

**原因**：`_topological_sort` 用递归 DFS，Python 默认递归深度 1000。
**解决**：`sys.setrecursionlimit(10000)`，或改用迭代版 DFS（PyTorch 的做法）。

---

## 3.7 与真实 PyTorch 对照

| minitorch 概念                      | PyTorch C++ 对应                                          | 关键差异                                                                 |
| ----------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------ |
| `run_backward(root, root_grad)`     | `Engine::execute_engine_root` (engine.cpp)                | PyTorch 多线程，minitorch 单线程                                         |
| `_topological_sort` 递归 DFS        | `find_remaining_nodes` + 显式栈                           | PyTorch 用迭代避免栈溢出，且能批量处理                                   |
| `grad_map: dict[int, Tensor]`       | `grads` 数组 + `variable_list`                            | PyTorch 用连续数组 + 索引，minitorch 用 dict + id                        |
| `with no_grad():` 包裹反向          | `AutoogradMode::is_grad_enabled()` RAII guard             | PyTorch 用 C++ RAII，minitorch 用 Python with                            |
| `node.backward_fn(grad)`            | `node.apply(grad)`                                        | PyTorch 的 Node 是多态类，apply 是虚函数；minitorch 用闭包              |
| `AccumulateGrad` 特判               | `AccumulateGrad::apply` 虚函数特化                        | PyTorch 靠虚函数分派，minitorch 靠 isinstance                            |
| `prev + g` 累加                     | `accumulate_grad` 函数                                    | PyTorch 有原地加法优化，minitorch 新建 Tensor                            |
| 单线程顺序执行                      | `ReadyQueue` + 多线程                                     | PyTorch 反向可多线程并行无依赖子图，minitorch 顺序                       |
| `retain_graph=False` 清 next_edges  | `clear_graph`                                              | 一致；PyTorch 还会清 `saved_tensors_`                                    |
| `retain_grad=False`                 | `AnomalyMetadata` / `retain_grad` hook                    | PyTorch 用 hook 机制，minitorch 内置参数                                 |
| 标量默认 `ones_like`                | `ones_like`                                                | 完全一致                                                                 |
| `grad_fn is None` 抛 RuntimeError   | `variable.h` 里类似检查                                   | 错误信息措辞不同，行为一致                                              |

!!! tip "PyTorch 多线程引擎详解"
    PyTorch 的反向引擎核心是 `ReadyQueue`：每个 Node 算完后，把它的前驱的"未完成依赖计数"减 1，
    计数归零的前驱入队。多个工作线程从队列取节点并行执行。
    依赖关系通过 `node->num_outputs_`（出度）追踪。
    这能让无依赖的子图（如两个独立的分支）并行反向，加速大模型训练。
    minitorch 单线程顺序执行，逻辑简单但无并行加速。

    ---

## 3.8 历史背景

### 3.8.1 反向模式 AD 的起源

反向模式自动微分最早可追溯到 1970 年代 Seppo Linnainmaa 的论文，
1986 年 Rumelhart/Hinton/Williams 的 backpropagation 论文让它名满天下。
但"反向模式 AD"作为通用技术，比"反向传播"概念更广——它不限于神经网络，
任何可微程序的梯度都能算。

### 3.8.2 PyTorch 引擎的演化

- **0.1 (2016)**：单线程引擎，与 minitorch 形态接近。
- **0.4 (2017)**：引入多线程反向，`ReadyQueue` 出现。
- **1.0 (2018)**：引擎重构，`Node` 改为 C++ 类，`execute_engine_root` 接口稳定。
- **1.5 (2020)**：引入 `accumulate_grad` 的 CUDA kernel，原地加法优化。
- **2.0 (2022)**：与 `torch.compile` 协作，反向图可被 Inductor 融合优化。

### 3.8.3 为什么 PyTorch 反向是多线程

大模型（如 Transformer）的反向传播里，很多子图相互独立（如多头注意力的各个头）。
单线程反向会让这些独立子图串行执行，浪费 CPU/GPU 并行能力。
多线程引擎让独立子图并行，能显著加速反向（典型 2-4x）。
代价是线程同步开销和调试复杂性——所以 minitorch 教学版选择单线程。

### 3.8.4 `no_grad` 的演化

早期 PyTorch 没有 `no_grad`，用户要用 `volatile=True` 标记张量。
0.4 把 `volatile` 移分进 `requires_grad`，引入 `torch.no_grad()` 上下文。
1.0 引入 `c10::AutoogradMode` 用 RAII 管理，支持嵌套和线程局部。
minitorch 的 `no_grad` 是 0.4 形态的简化：模块级全局变量，无线程局部。

---

## 3.9 练习题

### 练习 1：手算 `y = (x * 2).sum()` 的反向

`x = [1, 2, 3]`，求 `x.grad`。

??? 解答 ???

$L = \text{sum}(2x) = 2 \sum x_i$，$\frac{\partial L}{\partial x_i} = 2$，所以 `x.grad = [2, 2, 2]`。

代码验证：

```python
x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0])); x.requires_grad = True
y = (x * 2).sum()
y.backward()
assert x.grad.tolist() == [2.0, 2.0, 2.0]
```

### 练习 2：解释为什么 `reversed(topo)` 是正确的反向顺序

要求：用拓扑排序的定义证明。

??? 解答 ???

`topo` 是 DFS 后序：对每条边 `(u, v)`（u 是 v 的前驱，前向图里 v 由 u 算出），
DFS 先访问 v 再访问 u（后序把 u 加在 v 之后），所以 `topo` 里 `v` 在 `u` 之前。

`reversed(topo)` 里 `u` 在 `v` 之前。

反向传播要求"算 v 的梯度时，v 的所有下游（前向图里 v 的后继）已算完"。
前向图里 v 的后继就是"以 v 为输入的算子的输出"，即"v 指向的下游"。
在 `topo` 里这些下游排在 v 之后（因为 DFS 先递归子节点=下游，再访问 v），
所以 `reversed(topo)` 里它们排在 v 之前——先于 v 被处理。证毕。

### 练习 3：`grad_map` 里为什么用 `id(node)` 而不是 `node` 本身作键

??? 解答 ???

两个理由：

1. **Node 没定义 `__hash__` 和 `__eq__`**：Python 默认按 `id` 哈希，行为等价于用 `id`，
   但显式写 `id(node)` 更明确，不依赖默认行为。
2. **防御性**：如果未来给 Node 加了 `__eq__`（比如按 name 比较），用 Node 作键会出 bug：
   两个不同 Node 但 name 相同会被当成同一个键。用 `id` 永远唯一。

代价：`id` 在对象 GC 后可能被复用，但反向期间所有 node 都被 `topo` 列表强引用，不会 GC。

### 练习 4：如果 `backward_fn` 返回的梯度数和 `next_edges` 长度不符会怎样

要求：解释 `zip(..., strict=True)` 的行为。

??? 解答 ???

`strict=True` 是 Python 3.10+ 的参数，要求两个可迭代对象**长度相等**，否则抛 `ValueError`。

例如 `Mul.forward(ctx, a, b)` 有 2 个输入，`next_edges` 长度 2。
如果 `Mul.backward` 错误地返回 `(grad_a,)`（只有 1 个），
`zip(next_edges, grads, strict=True)` 会抛 `ValueError: zip() argument 2 is shorter than argument 1`。

这是**防御性断言**：让算子编写错误立即暴露，而非静默地把梯度错位分配。
如果没有 `strict=True`，`zip` 会按短的截断，`grad_b` 丢失，反向结果错误且难调试。

### 练习 5：为什么反向要包在 `no_grad` 里，而前向不要

??? 解答 ???

前向要建图（用户需要梯度），所以**不能**包在 `no_grad` 里——否则 `apply` 第 3 步短路，图建不出来。

反向过程中会调用各算子的 `backward`，里面要做张量运算（如 `Mul.backward` 里 `grad * b`）。
这些运算**不应该再建图**，原因有二：

1. **正确性**：反向的"图"不是用户想要的图，建出来会污染下游（如二次 backward 会走到反向图）。
2. **性能**：反向里的运算通常很多，建图开销显著，且这些中间梯度图用户用不到。

所以反向包在 `no_grad` 里，让 `apply` 短路，反向中的运算纯数值。

例外：`torch.autograd.grad` 里可以用 `create_graph=True` 让反向也建图，
用于高阶导数（如 `grad of grad`）。minitorch 暂不支持。

---

## 3.10 关键测试解读

### 3.10.1 `test_chain_rule`：链式法则基础

```python
def test_chain_rule():
    x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    x.requires_grad = True
    y = (x * 2).sum()
    y.backward()
    assert x.grad.tolist() == [2.0, 2.0, 2.0]
```

验证 `Mul` → `Sum` 的链式：$\frac{\partial \sum(2x)}{\partial x} = 2$。
这同时验证了 `Mul.backward`（返回 `grad * 2`）和 `Sum.backward`（broadcast）。

### 3.10.2 `test_shared_leaf_accumulation`：共享叶子累加

```python
def test_shared_leaf_accumulation():
    x = Tensor.from_numpy(np.array([2.0]))
    x.requires_grad = True
    y = x * x
    y.sum().backward()
    assert x.grad.tolist() == [4.0]
```

验证 `x * x` 的两个 `AccumulateGrad(x)` 各加一次：$\frac{d(x^2)}{dx} = 2x = 4$。
这是"叶子多次出现"的累加语义测试。

### 3.10.3 `test_mul_backward`：Mul 反向数值

```python
def test_mul_backward():
    a = Tensor.from_numpy(np.array([1.0, 2.0, 3.0])); a.requires_grad = True
    b = Tensor.from_numpy(np.array([4.0, 5.0, 6.0])); b.requires_grad = True
    (a * b).sum().backward()
    assert a.grad.tolist() == [4.0, 5.0, 6.0]   # ∂(ab)/∂a = b
    assert b.grad.tolist() == [1.0, 2.0, 3.0]   # ∂(ab)/∂b = a
```

验证 `Mul.backward` 的两个返回值正确：`grad_a = b`、`grad_b = a`。
同时验证两个不同叶子各自收梯度。

### 3.10.4 `test_matmul_backward`：Matmul 反向

```python
def test_matmul_backward():
    a = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]])); a.requires_grad = True
    b = Tensor.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0]])); b.requires_grad = True
    (a @ b).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 2)) @ np.eye(2).T)
    np.testing.assert_allclose(b.grad.numpy(), np.array([[1.0, 3.0], [2.0, 4.0]]) @ np.ones((2, 2)))
```

验证 `dA = grad @ B^T` 和 `dB = A^T @ grad`。
`b` 是单位矩阵，所以 `a.grad = ones @ eye.T = ones`。
用 `np.testing.assert_allclose` 而非 `tolist` 因为浮点矩阵可能有微小误差。

### 3.10.5 `test_sum_backward`：Sum 反向

```python
def test_sum_backward():
    x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0])); x.requires_grad = True
    x.sum().backward()
    assert x.grad.tolist() == [1.0, 1.0, 1.0]
```

验证 `Sum.backward` 把标量梯度 1 broadcast 到原 shape：每个元素的梯度都是 1。

### 3.10.6 `test_broadcast_backward`：广播反向

```python
def test_broadcast_backward():
    a = Tensor.from_numpy(np.ones((3, 4))); a.requires_grad = True
    b = Tensor.from_numpy(np.arange(4).astype(float)); b.requires_grad = True
    (a + b).sum().backward()
    assert a.grad.shape == (3, 4)
    assert b.grad.tolist() == [3.0, 3.0, 3.0, 3.0]
```

验证 `_reduce_grad`：`b` 被广播成 `(3, 4)`，反向时梯度沿 dim 0 sum 回 `(4,)`。
每个元素被加 3 次（3 行），所以 `b.grad = [3, 3, 3, 3]`。

### 3.10.7 `test_numerical_grad_comparison`：数值梯度对照

```python
def test_numerical_grad_comparison(numerical_grad):
    def f(v):
        t = Tensor.from_numpy(v); t.requires_grad = True
        return ((t * t).sum()).item()

    x0 = np.array([1.0, 2.0, 3.0])
    x = Tensor.from_numpy(x0); x.requires_grad = True
    ((x * x).sum()).backward()
    expected = numerical_grad(f, x0)
    np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-4)
```

**黄金测试**：把 autograd 算的梯度和数值差分（有限差分）算的梯度对照。
`numerical_grad` 是个 fixture，用 $f(x + h) - f(x - h) / (2h)$ 算数值梯度。
两者一致说明 autograd 实现正确。`atol=1e-4` 容忍数值差分的截断误差。

### 3.10.8 `test_complex_graph_accumulation`：复杂图累加

```python
def test_complex_graph_accumulation():
    x = Tensor.from_numpy(np.array([1.0, 2.0])); x.requires_grad = True
    y = x * x
    z = y + y
    z.sum().backward()
    assert x.grad.tolist() == [4.0, 8.0]
```

验证 3.5 节手算的例子：$L = \text{sum}(2x^2)$，$\frac{dL}{dx} = 4x = [4, 8]$。
这同时测了"多后继累加"（`y + y`）和"共享叶子累加"（`x * x`）两种累加路径。

### 3.10.9 `test_no_grad_context` / `test_enable_grad_context`：模式开关

```python
def test_no_grad_context():
    x = Tensor.from_numpy(np.array([1.0, 2.0])); x.requires_grad = True
    with no_grad():
        y = x + x
        assert not y.requires_grad
        assert y.grad_fn is None
    assert not is_grad_enabled() or is_grad_enabled()   # 退出后恢复

def test_enable_grad_context():
    with no_grad():
        assert not is_grad_enabled()
        with enable_grad():
            assert is_grad_enabled()
        assert not is_grad_enabled()
    assert is_grad_enabled()
```

验证 `no_grad` / `enable_grad` 的嵌套语义：
- 进入 `no_grad` 后 `is_grad_enabled() == False`，运算不建图。
- 退出后恢复到进入前的状态（最后 `assert is_grad_enabled()` 因为最外层默认开）。
- `enable_grad` 能在 `no_grad` 内嵌套打开。

### 3.10.10 `test_pow_backward`：Pow 反向

```python
def test_pow_backward():
    x = Tensor.from_numpy(np.array([2.0, 3.0])); x.requires_grad = True
    (x**3).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [12.0, 27.0])
```

验证 $\frac{d(x^3)}{dx} = 3x^2$：$3 \cdot 2^2 = 12$、$3 \cdot 3^2 = 27$。
测 `Pow.backward` 的 `grad_a = grad * exp * a^(exp-1)`。

---

## 3.11 优劣势总结

### 优势

- **正确性可证**：拓扑序保证梯度计算顺序正确，累加保证多后继/共享叶子正确。
- **内存可控**：默认 `retain_graph=False`，反向后图立即释放。
- **API 简洁**：`backward()` 一个方法搞定，标量默认梯度符合直觉。
- **与 PyTorch 一致**：行为对齐，便于教学迁移。

### 代价

- **单线程**：无并行加速，大模型反向慢。
- **递归拓扑**：深图栈溢出风险。
- **无原地优化**：每次累加新建 Tensor，频繁分配。
- **无 create_graph**：不支持高阶导数。
- **全局 `no_grad`**：非线程局部，多线程不安全（虽然 minitorch 不多线程）。

### 适用场景

- 教学：逻辑清晰，便于理解反向传播全过程。
- 小模型实验：性能足够，API 友好。
- 不适用：大模型训练（用 PyTorch 多线程引擎）、高阶导数（用 `torch.func`）、静态优化（用 `torch.compile`）。

---

## 3.12 下一章预告

本章我们让梯度"流"了一遍图，但图本身的**生命周期**还没讲清楚：

- 反向后图去哪了？为什么默认释放？
- 什么时候需要 `retain_graph=True`？RNN truncated BPTT 是什么场景？
- `retain_grad` 和 `retain_graph` 有什么区别？
- 为什么 backward 累加而非覆盖梯度？这和 SGD 的 `zero_grad()` 有什么关系？
- "Trying to backward through the graph a second time" 这个经典错误怎么来的？

第四章将聚焦**计算图机制本身**：它的表示、生命周期、释放策略，以及那些让初学者头疼的错误信息背后的设计。
