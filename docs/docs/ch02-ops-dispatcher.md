# 第二章 算子与分发：Function 模式与计算图建图

> 本章是 minitorch 自动微分系列的第二章。我们将从零讲清楚"一个加法算子是怎么被实现成可求导的"，
> 并解释 `Function.apply` 在前向计算时如何顺手把计算图"建"出来。
> 阅读本章前，请确保你已经理解第一章的 `Tensor`（Storage + shape + strides）。

---

## 2.1 本章目标

读完本章后，你应当能够：

1. 说出 `Function` 模式的三件套：`forward` / `backward` / `apply`，并解释它们各自的职责。
2. 用一句话讲清楚 `apply` 在前向时"做了什么额外的事"，为什么这一步是计算图建图的关键。
3. 区分 `Node` 与 `AccumulateGrad`：知道哪个是中间节点、哪个是叶子节点，以及它们的 `backward_fn` 行为差异。
4. 解释为什么 `Tensor` 上要同时存在 `_add`（底层）和 `add`（公开）两套方法，并能在新算子里正确选择。
5. 写出 `_reduce_grad` 的两个步骤，说明它在处理广播反向时为什么必须存在。
6. 把 minitorch 的 `_C.py` 路由表与 PyTorch 的 `DispatchKey / DispatchTable` 对应起来，说出"分发"二字的含义。
7. 在不查文档的情况下，给一个新算子（例如 `Sin`）写出完整的 `Function` 子类并接入 `Tensor`。
8. 读懂 `tests/test_ops.py` 里的每一行断言，知道它们在防御哪一类 bug。

---

## 2.2 原理铺垫：从"算子"到"可微算子"

### 2.2.1 什么是算子

在数值计算里，"算子"就是一个函数：把若干张量映射成一个张量。

$$
\text{Add}: (a, b) \mapsto a + b
$$

如果只关心前向计算，写一个 Python 函数就够了：

```python
def add(a, b):
    return a + b
```

但训练神经网络需要的是**梯度**。这就要求算子不仅会"算前向"，还要会"算反向"——
给定输出对下游的梯度 $\frac{\partial L}{\partial y}$，推出输入的梯度 $\frac{\partial L}{\partial a}$ 和 $\frac{\partial L}{\partial b}$。

### 2.2.2 链式法则的局部化

链式法则的本质是"局部可微即可全局可微"。对一个复合函数 $L = f(g(x))$：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

注意右边的两个因子：

- $\frac{\partial L}{\partial y}$ 是"上游传过来的梯度"，由调用方（反向引擎）提供；
- $\frac{\partial y}{\partial x}$ 是"本算子的局部导数"，只与本算子的前向有关。

**关键洞察**：每个算子只需要知道自己的局部导数，不需要知道整个网络长什么样。
这就是 `Function` 模式能 scale 到几百个算子的根本原因。

### 2.2.3 把算子拆成 forward / backward

于是每个算子被拆成两半：

| 方法     | 输入                       | 输出                       | 何时调用       |
| -------- | -------------------------- | -------------------------- | -------------- |
| forward  | 输入张量 + ctx             | 输出张量                   | 前向计算时     |
| backward | ctx + 上游梯度             | 各输入的梯度（元组）       | 反向传播时     |

`ctx`（Context）是一个"小信封"，forward 把反向要用到的中间量塞进去，
backward 再取出来。最典型的就是 `Mul`：反向需要原输入 $a, b$，所以 forward 要 `save_for_backward(a, b)`。

### 2.2.4 计算图是"顺手"建出来的

::: tip 心智模型
前向计算时，每经过一个可微算子，就在输出张量上挂一个"小标签" `grad_fn`，
记录"我是被哪个算子从哪些输入算出来的"。这些标签连起来就是计算图。
:::

用图示意一次 `z = (x * x) + (x * x)` 的前向：

```
   x (叶子, requires_grad=True)
   │
   ├──> Mul ──> y1 = x*x   (y1.grad_fn = Mul, next_edges=[AccumulateGrad(x), AccumulateGrad(x)])
   │
   └──> Mul ──> y2 = x*x   (y2.grad_fn = Mul, next_edges=[AccumulateGrad(x), AccumulateGrad(x)])

   y1, y2 ──> Add ──> z    (z.grad_fn = Add, next_edges=[y1.grad_fn, y2.grad_fn])
```

注意三件事：

1. **叶子节点 `x` 没有自己的 `grad_fn`**，但反向时仍要接收梯度——这就是 `AccumulateGrad` 存在的理由。
2. **同一个 `x` 被用了两次**，所以图里有两个 `AccumulateGrad(x)` 实例，反向时它们的梯度会被**累加**到同一个 `x.grad`。
3. **`z` 是输出**，它的 `grad_fn` 是整张图的入口，反向引擎从这里开始走。

### 2.2.5 类比：快递面单

把 `Tensor` 想象成快递包裹，`grad_fn` 是贴在包裹上的"面单"：

- 面单上写着"我是由 `Add` 这个工序，从包裹 A 和包裹 B 制造出来的"。
- 面单还附带一个"反向操作说明"（`backward_fn`），告诉分拣中心："如果有人要退货梯度，请按这个公式算给 A 和 B"。
- 如果一个包裹是用户直接创建的（叶子），它没有面单，但分拣中心会给它准备一个" AccumulateGrad 签收单"，专门用来累计别人退给它的梯度。

`Function.apply` 就是那个"贴面单"的工序。

---

## 2.3 设计决策与权衡

| 决策                                   | 选择                                    | 理由                                                                 | 代价                                              |
| -------------------------------------- | --------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------- |
| 算子用类还是函数                       | 类（`Function` 子类）                   | 需要在前向时携带 `ctx`，且要和 PyTorch API 形态一致                  | 多一层 `cls.apply`，比纯函数稍重                  |
| `forward`/`backward` 是静态还是实例方法 | 静态方法                                | 算子无状态，所有状态放进 `ctx`，避免实例化                          | 不能在 `backward` 里读 `self`，初学者易混淆       |
| 建图放在哪                             | `apply` 里                              | 算子的 `forward` 只关心数值，建图逻辑统一收敛到 `apply`              | `apply` 变得稍长，但所有算子共享同一份建图代码    |
| 是否对 `requires_grad=False` 也建图    | 不建                                    | 性能：推理时不应有图开销                                             | 用户必须显式 `requires_grad=True`，易忘           |
| 叶子节点如何收梯度                     | `AccumulateGrad` 子类                   | 叶子没有 `backward` 公式，只能"累加到 `.grad`"，行为与中间节点不同   | 反向引擎要 `isinstance` 判断，多一个分支          |
| 底层数值方法命名                       | `_add` / `_mul` 带下划线                | 公开 `add` 走 `apply` 建图，反向里调 `_add` 避免二次建图递归         | API 表面多一倍方法，需要文档说明                  |
| 广播反向如何处理                       | `_reduce_grad`                          | 前向广播把 `(3,)` 拉成 `(3,4)`，反向必须把梯度 sum 回 `(3,)`         | 反向代码每个算子都要手动调一次                    |
| 分发系统第一版用什么                   | Python `dict` 路由表                    | 教学优先，先跑通再优化；与 PyTorch `DispatchTable` 概念对齐          | 无多后端、无 kernel 选择策略，仅 CPU              |
| `save_for_backward` 存什么             | 只存 `Tensor`                           | 与 PyTorch 一致，避免误存大对象                                     | 标量参数（如 `dim`）要走 `ctx.meta`              |
| `next_edges` 里放 `None` 还是跳过      | 放 `None`                               | 与输入位置对齐，`zip(next_edges, grads)` 能正确配对                  | 引擎里要多判 `if edge is None`                    |

---

## 2.4 代码逐行实现

### 2.4.1 `Context`：forward 与 backward 之间的信封

```python
class Context:
    """forward 与 backward 之间传递信息的上下文。"""

    def __init__(self):
        self.saved_tensors: tuple = ()   # 反向要用的张量，初始为空
        self.meta: dict = {}             # 标量参数（dim/keepdim 等），用 dict 装

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors     # 直接覆盖，调用一次即可
```

为什么 `saved_tensors` 是 `tuple` 而不是 `list`？

- 语义上"保存完就不变"，用不可变类型更安全。
- 解包写法 `(a, b) = ctx.saved_tensors` 与元组天然契合。

为什么还要单独一个 `meta`？

- `Sum` 的 `dim` 是个 `int`，不是 `Tensor`，放进 `saved_tensors` 会让类型混乱。
- PyTorch 的做法是 `ctx.dim = dim` 直接挂属性；minitorch 用 `meta` dict 集中管理，便于教学展示。

### 2.4.2 `Node`：计算图的边与节点

```python
class Node:
    """计算图节点，持有 backward 函数与 next_edges。"""

    def __init__(self, backward_fn, next_edges: list, name: str = ""):
        self.backward_fn = backward_fn   # 反向时调用的可调用对象
        self.next_edges = next_edges     # 指向输入的 grad_fn 列表（邻接表）
        self.name = name                 # 算子名，仅用于调试/测试断言
        self.output: Tensor | None = None  # 本节点的输出张量，供 retain_grad 使用

    def __repr__(self) -> str:
        return f"Node({self.name})"
```

几个关键点：

- `backward_fn` 是一个**闭包**，在 `apply` 里用 `lambda *grads: cls.backward(ctx, *grads)` 生成。
  这样 `Node` 不需要再单独持有 `ctx` 和 `cls`，闭包自动捕获。
- `next_edges` 就是**邻接表**：`next_edges[i]` 是"第 i 个输入的 `grad_fn`"。
  反向引擎用它做拓扑遍历。
- `output` 字段在默认反向流程里**不参与**，只有 `retain_grad=True` 时才会把梯度写回 `output.grad`。
  这是 Ch4 的内容，这里先留个钩子。
- `name` 字段是教学辅助；真实 PyTorch 用 C++ RTTI，没有这个字段。

### 2.4.3 `AccumulateGrad`：叶子节点的特殊 Node

```python
class AccumulateGrad(Node):
    """叶子节点的 Node：反向时把梯度累加到 variable.grad。"""

    def __init__(self, variable: Tensor):
        self.variable = variable                       # 记住要累加到哪个 Tensor
        super().__init__(self._accumulate, [], name="AccumulateGrad")

    def _accumulate(self, grad: Tensor):
        if self.variable.grad is None:
            self.variable.grad = grad                  # 第一次：直接赋值
        else:
            self.variable.grad = self.variable.grad + grad  # 后续：累加
```

为什么叶子节点要单独一个子类？

1. **它没有 `backward` 公式**。叶子是用户创建的，不是某个算子的输出，不存在"局部导数"。
2. **它的"反向"动作就是"把梯度写进 `.grad`"**，这与中间节点"算出输入梯度并往下传"完全不同。
3. **累加语义**：同一个 `x` 在图里出现多次（如 `y = x * x`），会有多个 `AccumulateGrad(x)`，
   每次反向都往 `x.grad` 上加，最终得到全图对 `x` 的总梯度。

注意 `next_edges=[]`：叶子节点没有更上游了，反向到它就停。

### 2.4.4 `_reduce_grad`：广播反向的"还原器"

```python
def _reduce_grad(grad: Tensor, shape: tuple[int, ...]) -> Tensor:
    """把 grad 的 shape reduce 回 shape（处理前向广播的反向）。"""
    while grad.ndim > len(shape):
        grad = grad.sum(dim=0)              # 步骤 1：消掉前向多出来的高维
    for i in range(len(shape)):
        if grad.shape[i] != shape[i] and shape[i] == 1:
            grad = grad.sum(dim=i, keepdim=True)  # 步骤 2：把被广播的 1 维 sum 回去
    return grad
```

**为什么需要它？** 举个例子：

```
a.shape = (3, 4)      # requires_grad
b.shape = (4,)        # requires_grad
c = a + b             # 前向：b 被广播成 (3, 4)
```

反向时，`grad_output.shape = (3, 4)`。

- 对 `a`：`a.shape` 本来就是 `(3, 4)`，`_reduce_grad` 不动它。
- 对 `b`：`b.shape = (4,)`，但 `grad_output.shape = (3, 4)`，直接赋值会 shape 不匹配。
  必须把 `(3, 4)` 沿 dim 0 sum 成 `(4,)`，这正是"广播的逆操作"。

两步的逻辑：

1. **维度数对齐**：前向广播可能在前面补维度（`(4,) → (1, 4) → (3, 4)`），
   反向就沿这些补出来的维度 `sum(dim=0)`。
2. **大小为 1 的维度**：前向把 `1` 扩成 `3`，反向沿这个维度 `sum(keepdim=True)` 还原成 `1`。

::: warning 易错点
`_reduce_grad` 只处理"前向广播过的维度"。如果 `grad` 和 `shape` 完全一致，
函数原样返回，不会有任何副作用。所以**每个二元算子的反向都无脑调一次**是安全的。
:::

### 2.4.5 `Function` 基类与 `apply`

```python
class Function:
    """自动微分算子基类。子类实现 forward/backward 静态方法。"""

    @staticmethod
    def forward(ctx: Context, *args) -> Tensor:
        raise NotImplementedError          # 子类必须重写

    @staticmethod
    def backward(ctx: Context, *grad_outputs) -> tuple:
        raise NotImplementedError          # 子类必须重写

    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor:
        ctx = Context()                                       # 1. 建信封
        result = cls.forward(ctx, *args, **kwargs)            # 2. 算前向

        if not is_grad_enabled():                             # 3. no_grad 下不建图
            return result

        needs_grad = any(isinstance(a, Tensor) and a.requires_grad for a in args)
        if not needs_grad:                                    # 4. 全不需要梯度，不建图
            return result

        next_edges: list[Node | None] = []
        for a in args:                                        # 5. 为每个输入准备 edge
            if isinstance(a, Tensor) and a.requires_grad:
                if a.grad_fn is not None:
                    next_edges.append(a.grad_fn)              #    中间节点：直接用它的 grad_fn
                else:
                    next_edges.append(AccumulateGrad(a))      #    叶子节点：新建 AccumulateGrad
            else:
                next_edges.append(None)                       #    不求导的输入：占位 None

        node = Node(
            backward_fn=lambda *grads: cls.backward(ctx, *grads),  # 6. 闭包捕获 ctx
            next_edges=next_edges,
            name=cls.__name__,
        )
        node.output = result
        result.requires_grad = True                           # 7. 输出标记为需要梯度
        result.grad_fn = node                                 # 8. 把面单贴到输出上
        return result
```

逐段解读 `apply` 的 8 步：

**第 1-2 步：建信封、算前向。**
`ctx` 是给 `forward` 用的，`forward` 里会 `save_for_backward` 把反向需要的张量塞进去。
注意 `forward` **只做数值计算**，不碰 `grad_fn`，不碰 `requires_grad`。

**第 3 步：`no_grad` 短路。**
`is_grad_enabled()` 是个全局开关（见 `grad_mode.py`）。在 `with no_grad():` 块里，
前向只算数值不建图——这是推理和反向内部运算的共同需求。

**第 4 步：全不求导短路。**
如果所有输入都 `requires_grad=False`，输出也不需要梯度，没必要建图。
这是性能优化，也是正确性要求：否则会建出一堆无意义的 `Node`。

**第 5 步：构造 `next_edges`。**
这是建图的核心。对每个输入分三种情况：

- 是 `Tensor` 且 `requires_grad=True` 且 `grad_fn is not None`：它是某个算子的输出（中间节点），
  直接复用它的 `grad_fn` 作为 edge。**注意是复用，不是新建**——这就是"同一个 `x` 用两次会产生两条 edge 指向同一个 Node"的来源。
- 是 `Tensor` 且 `requires_grad=True` 但 `grad_fn is None`：它是叶子节点（用户创建的），
  新建一个 `AccumulateGrad(a)`。**这里每次都新建**，所以同一个 `x` 用两次会有两个 `AccumulateGrad` 实例。
- 不是 `Tensor` 或不需要梯度：放 `None` 占位，保持位置对齐。

**第 6 步：构造 `Node`。**
`backward_fn` 是个闭包，捕获了 `ctx` 和 `cls`。反向引擎调用 `node.backward_fn(grad)` 时，
等价于调用 `cls.backward(ctx, grad)`。

**第 7-8 步：标记输出、贴面单。**
`result.requires_grad = True` 让下游算子知道"我有梯度"。
`result.grad_fn = node` 把面单贴上，下游的 `apply` 会通过 `result.grad_fn` 找到这个 node。

### 2.4.6 一个具体算子：`Add`

```python
class Add(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)                 # 反向要用 a.shape, b.shape 做 reduce
        a, b = Tensor.broadcast_tensors(a, b)       # 广播到同形
        return a._add(b)                            # 调底层 _add，不走 apply，不建图

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b = ctx.saved_tensors
        # Add 的局部导数：∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
        # 所以 grad_a = grad_output * 1, grad_b = grad_output * 1
        # 但广播过的输入要 reduce 回原 shape
        return (_reduce_grad(grad_output, a.shape), _reduce_grad(grad_output, b.shape))
```

逐行解读：

- `ctx.save_for_backward(a, b)`：保存的是**广播前**的 `a, b`，因为反向要 reduce 回它们的原始 shape。
- `Tensor.broadcast_tensors(a, b)`：静态方法，返回广播后的两个新 Tensor（view，不拷贝数据）。
- `a._add(b)`：调底层方法，**不走 `Add.apply`**，所以不会再建图。这是 `_` 前缀方法存在的根本理由。
- `backward` 返回**元组**，长度等于 `forward` 的输入个数。即使某个输入不求导（如 `Pow` 的指数），
  对应位置也要返回 `None`，保持位置对齐。

### 2.4.7 `Mul`：反向需要原输入

```python
class Mul(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        a, b = Tensor.broadcast_tensors(a, b)
        return a._mul(b)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b = ctx.saved_tensors
        # ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
        grad_a = _reduce_grad(grad_output._mul(b), a.shape)
        grad_b = _reduce_grad(grad_output._mul(a), b.shape)
        return (grad_a, grad_b)
```

对比 `Add`：`Mul` 的反向不是简单返回 `grad_output`，而是要乘以另一个输入。
这就是 `save_for_backward` 必须存在的原因——反向时前向的输入已经"过去了"，必须提前存下来。

注意 `grad_output._mul(b)` 用的是 `_mul` 而不是 `*`：
反向发生在 `with no_grad():` 下（见 Ch3 引擎），但即便如此，用 `_mul` 更明确地表达"我不要建图"。

### 2.4.8 `Matmul`：没有广播，但有转置

```python
class Matmul(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a._matmul(b)                  # 不广播，要求 a.shape[-1] == b.shape[-2]

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b = ctx.saved_tensors
        # 对矩阵乘法 C = A @ B：
        #   dA = grad_output @ B^T
        #   dB = A^T @ grad_output
        grad_a = grad_output._matmul(b.transpose())
        grad_b = a.transpose()._matmul(grad_output)
        return (grad_a, grad_b)
```

注意 `b.transpose()` 调的是**公开** `transpose`（走 `Transpose.apply`）。
这在反向里是安全的吗？是的，因为引擎用 `with no_grad():` 包裹了整个反向过程，
`Transpose.apply` 里的 `is_grad_enabled()` 会返回 `False`，直接短路返回，不建图。

### 2.4.9 `Sum` / `Mean`：标量参数走 `ctx.meta`

```python
class Sum(Function):
    @staticmethod
    def forward(ctx, a: Tensor, dim=None, keepdim: bool = False) -> Tensor:
        ctx.save_for_backward(a)
        ctx.meta["dim"] = dim                # 标量参数走 meta，不走 saved_tensors
        ctx.meta["keepdim"] = keepdim
        return a._sum(dim=dim, keepdim=keepdim)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (a,) = ctx.saved_tensors
        grad = grad_output
        if ctx.meta["dim"] is not None and not ctx.meta["keepdim"]:
            grad = grad.unsqueeze(ctx.meta["dim"])   # 把消失的维度补回来
        grad = grad.broadcast_to(a.shape)            # 广播回 a 的形状
        return (grad,)
```

`Sum` 的反向逻辑：

- 前向 `sum(dim=1)` 把 shape `(3, 4)` 压成 `(3,)`。
- 反向要把 `(3,)` 的梯度"展开"回 `(3, 4)`：先 `unsqueeze(1)` 变 `(3, 1)`，再 `broadcast_to((3, 4))`。
- 如果 `keepdim=True`，前向输出是 `(3, 1)`，不需要 unsqueeze，直接 broadcast。

### 2.4.10 `Tensor` 上的双面方法：`_add` vs `add`

```python
# tensor.py 里
def _add(self, other) -> Tensor:
    return self._binary(other, lambda a, b: a + b)    # 纯数值，不建图

def add(self, other) -> Tensor:
    from .ops.arithmetic import Add
    return Add.apply(self, self._ensure_tensor(other))  # 走 apply，建图

def __add__(self, other): return self.add(other)       # 操作符重载走公开方法
```

**两套方法的本质分工**：

| 方法    | 调用方               | 是否建图 | 用途                       |
| ------- | -------------------- | -------- | -------------------------- |
| `_add`  | 算子的 `forward`/`backward` | 否       | 纯数值计算，避免递归建图   |
| `add`   | 用户代码 / `__add__` | 是       | 用户入口，需要自动微分     |

如果只有 `add` 没有 `_add`，会发生什么？

```python
# 假设 Add.forward 写成 return a + b（即 a.add(b)）
# 那么 Add.apply 调 forward → forward 调 add → add 调 Add.apply → ...
# 无限递归，栈溢出
```

所以 `_add` 是"逃生门"，让 `forward` 能做数值计算而不重新触发建图。

### 2.4.11 分发路由表 `_C.py`

```python
_dispatch_table: dict[tuple[str, str], Callable] = {}

def register(op_name: str, kernel: Callable, device: str = "cpu") -> None:
    _dispatch_table[(op_name, device)] = kernel

def dispatch(op_name: str, *args: Any, device: str = "cpu", **kwargs: Any) -> Any:
    key = (op_name, device)
    if key not in _dispatch_table:
        raise RuntimeError(f"no kernel registered for op '{op_name}' on device '{device}'")
    return _dispatch_table[key](*args, **kwargs)

def has_kernel(op_name: str, device: str = "cpu") -> bool:
    return (op_name, device) in _dispatch_table
```

当前 minitorch 的算子**并没有真的走 `dispatch`**——`Add.forward` 直接调 `a._add(b)`。
`_C.py` 是为后续章节（多后端、C++ kernel）预留的骨架，教学上用来对照 PyTorch 的分发机制。

关键设计：

- **键是 `(op_name, device)` 二元组**：同一个算子名在不同设备上有不同 kernel。
  这就是 PyTorch `DispatchKey` 的雏形。
- **`register` 注册、`dispatch` 查表调用**：与 PyTorch `torch::Library::def` + `Dispatcher::call` 对应。
- **缺失 kernel 抛 `RuntimeError`**：和 PyTorch 行为一致，便于上层统一捕获。

---

## 2.5 完整示例：从创建到运算

```python
import numpy as np
from minitorch import Tensor

# 1. 创建叶子张量
x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
x.requires_grad = True
print("x =", x.tolist(), "requires_grad =", x.requires_grad, "grad_fn =", x.grad_fn)
# x = [1.0, 2.0, 3.0] requires_grad = True grad_fn = None

# 2. 第一次运算：y = x * 2
y = x * 2
print("y =", y.tolist(), "requires_grad =", y.requires_grad)
print("y.grad_fn =", y.grad_fn, "name =", y.grad_fn.name)
# y = [2.0, 4.0, 6.0] requires_grad = True
# y.grad_fn = Node(Mul) name = Mul

# 3. 查看 next_edges：y 的输入是 x 和 2
#    x 是叶子 → AccumulateGrad(x)
#    2 不需要梯度 → None
print("edges:", y.grad_fn.next_edges)
# edges: [Node(AccumulateGrad), None]
print("edge[0].variable is x:", y.grad_fn.next_edges[0].variable is x)
# edge[0].variable is x: True

# 4. 第二次运算：z = y + y（同一个 y 用两次）
z = y + y
print("z.grad_fn.name =", z.grad_fn.name)             # Add
print("z.grad_fn.next_edges[0] is z.grad_fn.next_edges[1]:",
      z.grad_fn.next_edges[0] is z.grad_fn.next_edges[1])
# True —— 因为 y 用了两次，两条 edge 指向同一个 y.grad_fn

# 5. 不需要梯度的运算不建图
a = Tensor.from_numpy(np.array([1.0, 2.0]))
b = Tensor.from_numpy(np.array([3.0, 4.0]))
c = a + b
print("c.requires_grad =", c.requires_grad, "c.grad_fn =", c.grad_fn)
# c.requires_grad = False c.grad_fn = None

# 6. 广播运算
m = Tensor.from_numpy(np.ones((3, 4)))
m.requires_grad = True
v = Tensor.from_numpy(np.arange(4).astype(float))
v.requires_grad = True
out = m + v
print("out.shape =", out.shape)                       # (3, 4)
print("out[0] =", out[0].tolist())                   # [1, 2, 3, 4]
# 反向时 m 的梯度 shape 仍是 (3,4)，v 的梯度会被 _reduce_grad sum 成 (4,)
```

预期输出（精简）：

```
x = [1.0, 2.0, 3.0] requires_grad = True grad_fn = None
y = [2.0, 4.0, 6.0] requires_grad = True
y.grad_fn = Node(Mul) name = Mul
edges: [Node(AccumulateGrad), None]
edge[0].variable is x: True
z.grad_fn.name = Add
z.grad_fn.next_edges[0] is z.grad_fn.next_edges[1]: True
c.requires_grad = False c.grad_fn = None
out.shape = (3, 4)
out[0] = [1.0, 2.0, 3.0, 4.0]
```

---

## 2.6 常见陷阱

### 陷阱 1：忘记 `requires_grad = True`

```python
x = Tensor.from_numpy(np.array([1.0, 2.0]))
y = x * x
y.sum().backward()   # RuntimeError: backward() called on a tensor with no grad_fn
```

**原因**：`x.requires_grad` 默认 `False`，`apply` 第 4 步短路，没建图。
**解决**：创建后显式 `x.requires_grad = True`，或用未来的 `Tensor(..., requires_grad=True)` 构造参数。

### 陷阱 2：在 `forward` 里用了公开方法导致递归

```python
class BadAdd(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a + b   # BUG：a + b 调 a.add(b) 调 Add.apply 又调 forward → 无限递归
```

**解决**：`forward` / `backward` 里**只用 `_` 前缀方法**（`_add` / `_mul` / ...）。

### 陷阱 3：`backward` 返回的梯度顺序与 `forward` 输入不一致

```python
class BadMul(Function):
    @staticmethod
    def forward(ctx, a, b): ...
    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * a, grad_output * b)   # 如果把 a, b 顺序写反了呢？
```

`apply` 里 `zip(next_edges, grads)` 按位置配对，顺序反了会把 `a` 的梯度塞给 `b`。
**解决**：`backward` 返回元组的第 i 个元素**必须**对应 `forward` 的第 i 个输入。

### 陷阱 4：广播反向忘记 `_reduce_grad`

```python
class BadAdd(Function):
    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output, grad_output)   # 广播时会 shape 不匹配
```

当 `a.shape=(3,4)`、`b.shape=(4,)` 时，`grad_output.shape=(3,4)` 直接赋给 `b.grad` 会报错。
**解决**：每个二元算子的反向都要 `_reduce_grad(grad, original_input.shape)`。

### 陷阱 5：`save_for_backward` 存了不该存的大对象

```python
class Big(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b, a @ b)   # 把中间结果也存了，内存翻倍
        return a @ b
```

**原则**：只存反向**必需**的张量。能从前向输出反推的，不要重复存。

### 陷阱 6：在 `no_grad` 下修改 `requires_grad`

```python
with no_grad():
    x.requires_grad = True   # 不会报错，但 x 参与的运算仍不建图
    y = x + 1
    assert y.grad_fn is None   # 因为 is_grad_enabled() 是 False
```

`no_grad` 是**全局开关**，优先级高于单个 Tensor 的 `requires_grad`。
**解决**：要建图就退出 `no_grad` 块，或嵌套 `enable_grad`。

---

## 2.7 与真实 PyTorch 对照

| minitorch 概念                | PyTorch C++ 对应                                    | 关键差异                                                                 |
| ----------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------ |
| `Function` 基类               | `torch::autograd::Function`                         | PyTorch 用 CRTP 静态分发，minitorch 用 `@classmethod apply`             |
| `Function.apply`              | `Function::apply` (function.h)                      | PyTorch 在 C++ 里手工展开变参，minitorch 用 Python `*args`              |
| `Context`                     | `AutogradContext`                                   | 几乎一一对应；PyTorch 还支持 `mark_dirty` / `mark_non_differentiable`  |
| `Node`                        | `Node` (function.h)                                 | PyTorch 的 Node 是 C++ 类，`next_edges_` 是 `edge_list`                  |
| `AccumulateGrad`              | `AccumulateGrad` (function.h)                       | 完全对应；PyTorch 里它是模板类，针对不同 variable 类型特化              |
| `_reduce_grad`                | `at::sum_to_size` / `reduce_grad`                   | PyTorch 内联在 kernel 里，且处理更复杂的广播语义                        |
| `Tensor._add` vs `Tensor.add` | `at::_add` vs `at::add` (TensorBody.h)              | PyTorch 的 `at::` 命名空间就是"不建图"层，`torch::` 才走 autograd        |
| `_C.py` dict 路由表           | `c10::DispatchTable` + `DispatchKey`                | PyTorch 有几十个 DispatchKey（CPU/CUDA/Autograd/...），minitorch 只有 CPU |
| `register("add_cpu", kernel)` | `TORCH_LIBRARY_IMPL(..., CPU, m.impl("add", ...))`  | PyTorch 用宏注册，minitorch 用函数调用                                   |
| `dispatch("add_cpu", a, b)`   | `c10::Dispatcher::call(op, args)`                   | PyTorch 的 call 还要走 Autograd dispatch key（建图），minitorch 分两层   |
| `next_edges` 里放 `None`      | `Edge` 用 `isDefined()` 判断                        | PyTorch 用 C++ 指针，minitorch 用 Python None                            |
| `ctx.meta` dict               | `ctx.meta()` / 直接挂属性                           | PyTorch 允许任意属性，minitorch 集中到一个 dict                          |

::: tip 关键差异详解
PyTorch 的分发是**多层的**：一个 `torch::add` 调用会先走 `Autograd` key（建图），
再走 `CPU` key（数值）。minitorch 把这两层硬编码进了 `apply`（建图）和 `_add`（数值），
没有显式的 Autograd dispatch key。这种"两层硬编码"在教学上更清晰，
但牺牲了 PyTorch 分发系统的可扩展性（如 functorch 的 vmap 就靠插入新 dispatch key 实现）。
:::

---

## 2.8 历史背景

### 2.8.1 PyTorch 之前的自动微分

在 PyTorch 之前，主流深度学习框架的自动微分有两类做法：

- **Theano / TensorFlow 静态图**：先编译一张计算图，再反复执行。
  优点是能做全局优化（算子融合、内存复用），缺点是调试困难、控制流不友好。
- **autograd (HIPS/autograd) / Chainer**：动态建图，但用闭包链表表示，
  缺少对多输出、多后继的一等支持。

PyTorch（2016）选择了**动态图 + 反向模式 AD + tape-based**：
每次前向实时建图（tape），反向沿 tape 走一遍，然后丢弃。
这让控制流（if/while/for）天然可微，调试体验接近纯 Python。

### 2.8.2 `Function` 模式的演化

早期 PyTorch（0.1）的算子用 LuaTorch 风格的 C 函数表，没有统一的 `Function` 基类。
0.2 引入 `torch.autograd.Function`，统一了 forward/backward 接口。
0.4 把 `Variable` 和 `Tensor` 合并（之前 `Variable` 是包装类），`grad_fn` 直接挂到 `Tensor` 上。
1.0 引入 `c10::Dispatcher`，把"算子"和"实现"解耦，为多后端、functorch、torch.compile 铺路。

minitorch 的 `Function` 模式对应 PyTorch 1.x 的形态：`Tensor` 即 `Variable`，`grad_fn` 挂在 Tensor 上。

### 2.8.3 为什么是反向模式 AD

正向模式 AD 对每个输入要跑一次，反向模式 AD 对每个输出要跑一次。
神经网络通常是"百万输入、单输出（loss）"，反向模式远更高效。
代价是反向模式需要先做完前向、存好 tape，内存开销更大。
这就是为什么 PyTorch 默认反向，只在 `torch.func.jvp` 等场景才用正向模式。

### 2.8.4 分发系统的演化

PyTorch 1.0 之前，算子是 `TH` 库的 C 函数，没有分发。
1.0 引入 `c10::Dispatcher`，每个算子是一个 `OperatorHandle`，按 `DispatchKey` 路由到不同实现。
1.13 引入 `func` namespace 和 ` functorch`，靠新的 dispatch key 实现 vmap/vjp。
2.0 引入 `torch.compile`，又加了 `Dynamo` / `Inductor` 等编译期 dispatch key。

minitorch 的 `_C.py` 是 1.0 之前形态的简化：单一 dict，单后端。
后续章节会逐步引入 device key、autograd key，向真实 Dispatcher 演化。

---

## 2.9 练习题

### 练习 1：实现 `Sin` 算子

要求：`Sin.forward(ctx, a)` 返回 `sin(a)`；`Sin.backward(ctx, grad_output)` 返回 `(grad_output * cos(a),)`。
提示：`a._numpy_view()` 可以拿到 numpy 数组，`np.sin` / `np.cos` 可用。

??? 解答 ???

```python
class Sin(Function):
    @staticmethod
    def forward(ctx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return Tensor.from_numpy(np.sin(a.numpy()))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (a,) = ctx.saved_tensors
        cos_a = Tensor.from_numpy(np.cos(a.numpy()))
        return (grad_output._mul(cos_a),)
```

接入 `Tensor`：

```python
def sin(self) -> Tensor:
    from .ops.arithmetic import Sin
    return Sin.apply(self)
```

注意 `backward` 里用 `grad_output._mul(cos_a)` 而非 `grad_output * cos_a`，
避免在反向中二次建图（虽然引擎已用 `no_grad` 包裹，但 `_mul` 更明确）。

### 练习 2：解释 `y = x * x` 的图里为什么有两个 `AccumulateGrad(x)`

要求：画出 `y = x * x` 的计算图，标出所有 `Node` 和 `next_edges`，解释为什么 `x` 的 edge 不是同一个实例。

??? 解答 ???

```
x (叶子)
│
├── edge0: AccumulateGrad(x)  ──┐
│                                ├──> Mul ──> y
└── edge1: AccumulateGrad(x)  ──┘
```

`Mul.apply(x, x)` 的第 5 步遍历 `args = (x, x)`：
- 第一个 `x`：`grad_fn is None` → 新建 `AccumulateGrad(x)` 实例 A
- 第二个 `x`：`grad_fn is None` → 新建 `AccumulateGrad(x)` 实例 B

A 和 B 是**两个不同的 Python 对象**，但都指向同一个 `x`。
反向时 A 和 B 各自把梯度累加到 `x.grad`，最终 `x.grad = 2 * x * grad_output`，正是 $\frac{d(x^2)}{dx} = 2x$。

如果只建一个共享的 `AccumulateGrad`，反向引擎只会调用它一次，
`x.grad` 只会得到一份梯度，结果就错了。所以"每次出现都新建"是正确性要求。

### 练习 3：`Pow` 的 backward 为什么返回 `(grad_a, None)`

要求：解释 `None` 的含义，以及 `apply` 里 `zip(next_edges, grads)` 如何处理它。

??? 解答 ???

`Pow.forward(ctx, a, exponent)` 有两个输入：底数 `a` 和指数 `exponent`。
通常只对 `a` 求导，不对 `exponent` 求导（exponent 常是常数）。
所以 `backward` 返回 `(grad_a, None)`，第二个 `None` 表示"指数没有梯度"。

`apply` 第 5 步构造 `next_edges` 时，如果 `exponent` 是 `float`（不是 Tensor），
它会被归到 `else` 分支，`next_edges[1] = None`。
反向引擎里 `zip(next_edges, grads)` 得到 `(None, None)`，
`if edge is None or g is None: continue` 跳过，不会出错。

如果 `exponent` 是 `Tensor` 且 `requires_grad=True`，`next_edges[1]` 不是 None，
但 `backward` 返回的 `None` 仍会让引擎跳过——这意味着**当前实现不支持对指数求导**。
要支持需补全 `Pow.backward` 的第二项：$\frac{\partial a^e}{\partial e} = a^e \ln a$。

### 练习 4：手算 `_reduce_grad` 的结果

`grad` 形状 `(2, 3, 4)`，要 reduce 回 shape `(3, 4)` 和 `(1, 4)`，分别得到什么？

??? 解答 ???

**reduce 回 `(3, 4)`**：
- 步骤 1：`grad.ndim=3 > len((3,4))=2`，`grad = grad.sum(dim=0)` → shape `(3, 4)`
- 步骤 2：遍历 i=0,1，`grad.shape[i] == shape[i]`，不操作
- 结果：`grad.sum(dim=0)`，shape `(3, 4)`

**reduce 回 `(1, 4)`**：
- 步骤 1：`grad.ndim=3 > len((1,4))=2`，`grad = grad.sum(dim=0)` → shape `(3, 4)`
- 步骤 2：i=0，`grad.shape[0]=3 != shape[0]=1 且 shape[0]==1`，`grad = grad.sum(dim=0, keepdim=True)` → shape `(1, 4)`
- i=1，`grad.shape[1]=4 == shape[1]=4`，不操作
- 结果：`grad.sum(dim=0).sum(dim=0, keepdim=True)`，shape `(1, 4)`

### 练习 5：为什么 `Add.backward` 不需要 `a` 和 `b` 的值，却仍要 `save_for_backward`

要求：观察 `Add.backward` 只用了 `a.shape` 和 `b.shape`，没用到数值。能否不 save？

??? 解答 ???

当前实现里 `Add.backward` 确实只用 shape，理论上可以只存 shape：

```python
ctx.meta["a_shape"] = a.shape
ctx.meta["b_shape"] = b.shape
```

但 minitorch 选择 `save_for_backward(a, b)` 有两个理由：

1. **与 PyTorch 习惯一致**：PyTorch 的 AddBackward 也 save 输入，便于未来扩展（如检查 NaN）。
2. **教学统一**：所有二元算子都用同一套 `save_for_backward(a, b)` 模式，减少认知负担。

代价是多持有了两个 Tensor 引用。但 Tensor 是 view（不拷贝数据），引用开销很小。
在极端内存敏感场景，可以改成只存 shape——这就是工程权衡。

---

## 2.10 关键测试解读

### 2.10.1 `test_add_forward`：前向数值正确

```python
def test_add_forward():
    a = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    b = Tensor.from_numpy(np.array([4.0, 5.0, 6.0]))
    c = a + b
    assert c.tolist() == [5.0, 7.0, 9.0]
```

验证最基础的 `__add__` → `add` → `Add.apply` → `Add.forward` → `_add` 链路。
`a` / `b` 都没设 `requires_grad`，所以 `apply` 第 4 步短路，不建图，纯数值。
这行测试同时验证了"不建图时也能算对"。

### 2.10.2 `test_requires_grad_propagation`：requires_grad 传播

```python
def test_requires_grad_propagation():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    x.requires_grad = True
    y = x + x
    assert y.requires_grad          # 输入需要 → 输出需要
    z = y * y
    assert z.requires_grad          # 链式传播
```

验证 `apply` 第 7 步 `result.requires_grad = True` 在链式调用中正确传播。
如果某一步漏了这行，`z.requires_grad` 会是 `False`，下游就不会建图。

### 2.10.3 `test_grad_fn_built`：面单正确贴上

```python
def test_grad_fn_built():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    x.requires_grad = True
    y = x + x
    assert y.grad_fn is not None
    assert y.grad_fn.name == "Add"   # name 字段用于测试断言
```

验证 `apply` 第 8 步 `result.grad_fn = node`，且 `node.name` 是算子类名。
`name` 字段在生产代码里没用，纯粹是为了让测试可读。

### 2.10.4 `test_no_graph_when_not_required`：不建图短路

```python
def test_no_graph_when_not_required():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    y = x + x
    assert not y.requires_grad
    assert y.grad_fn is None
```

验证 `apply` 第 4 步：所有输入 `requires_grad=False` 时，输出不建图。
这是推理场景的性能保证。

### 2.10.5 `test_computation_graph_chain`：图的链式结构

```python
def test_computation_graph_chain():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    x.requires_grad = True
    y = x * x
    z = y + y
    assert z.grad_fn.name == "Add"
    assert z.grad_fn.next_edges[0] is y.grad_fn   # 边指向 y 的 grad_fn
    assert y.grad_fn.name == "Mul"
```

验证 `next_edges` 的正确性：`z = y + y` 的两条 edge 都指向 `y.grad_fn`，
且是**同一个对象**（`is` 而非 `==`）。
这验证了 `apply` 第 5 步"中间节点复用 `grad_fn`"的逻辑。

### 2.10.6 `test_broadcast_in_apply`：广播在前向生效

```python
def test_broadcast_in_apply():
    a = Tensor.from_numpy(np.ones((3, 4)))
    a.requires_grad = True
    b = Tensor.from_numpy(np.arange(4).astype(float))
    c = a + b
    assert c.shape == (3, 4)
    assert c[0].tolist() == [1, 2, 3, 4]
```

验证 `Add.forward` 里的 `broadcast_tensors` 把 `b` 从 `(4,)` 广播到 `(3, 4)`。
`c[0]` 是 `[1+0, 1+1, 1+2, 1+3] = [1, 2, 3, 4]`。

### 2.10.7 `test_leaf_accumulate_grad_node`：叶子节点用 AccumulateGrad

```python
def test_leaf_accumulate_grad_node():
    x = Tensor.from_numpy(np.array([1.0, 2.0]))
    x.requires_grad = True
    y = x * x
    edge = y.grad_fn.next_edges[0]
    assert edge.name == "AccumulateGrad"
    assert edge.variable is x
```

验证 `apply` 第 5 步：叶子节点（`grad_fn is None`）会新建 `AccumulateGrad`，
且 `variable` 字段指向原 `x`。这是反向时梯度能写回 `x.grad` 的关键。

### 2.10.8 `test_dispatcher_register_and_dispatch`：分发路由表

```python
def test_dispatcher_register_and_dispatch():
    register("add_cpu", lambda a, b: a + b)
    assert has_kernel("add_cpu")
    assert dispatch("add_cpu", 1, 2) == 3
```

验证 `_C.py` 的 register/dispatch 基本功能。
注意这里注册的是一个**纯 Python lambda**，与算子层无关——
`_C.py` 是底层骨架，目前还没被 `Add.forward` 真正使用。

### 2.10.9 `test_dispatcher_missing_kernel`：缺失 kernel 报错

```python
def test_dispatcher_missing_kernel():
    assert not has_kernel("nonexistent_op")
    with pytest.raises(RuntimeError):
        dispatch("nonexistent_op")
```

验证分发失败时的错误行为：抛 `RuntimeError`，与 PyTorch 一致。
这让上层能统一 `try/except RuntimeError` 处理"算子未实现"。

---

## 2.11 优劣势总结

### 优势

- **统一接口**：所有算子写法一致，新增算子只需 `forward` + `backward` 两个静态方法。
- **建图透明**：用户写 `y = a + b`，建图自动发生，无需显式 tape。
- **局部可微**：每个算子只管自己的局部导数，组合即可表达任意复杂网络。
- **与 PyTorch API 一致**：学完 minitorch 可无缝切换到 PyTorch。

### 代价

- **每个算子要写两遍**：forward 和 backward，工作量翻倍。
- **`_` 前缀方法污染 API**：`Tensor` 上同时有 `_add` 和 `add`，初学者易混淆。
- **广播反向要手动 reduce**：每个二元算子都要调 `_reduce_grad`，容易漏。
- **单后端**：`_C.py` 目前只有 CPU，多后端要扩 dispatch key。
- **无算子融合**：每个算子独立调度，无法把 `a + b * c` 融合成一个 kernel。

### 适用场景

- 教学与小规模实验：API 简洁，便于理解自动微分原理。
- 快速原型：新算子几十行就能接入。
- 不适用：高性能训练（应直接用 PyTorch）、静态图优化场景（应用 torch.compile）。

---

## 2.12 下一章预告

本章我们解决了"**算子怎么建图**"，但图建好之后一直没动它——`grad_fn` 挂在 Tensor 上，
`next_edges` 串成邻接表，然后呢？

第三章将回答：**反向传播引擎是如何沿这张图走一遍，把梯度算出来的？**

具体涉及：

1. 反向模式 AD 的数学原理（链式法则的矩阵形式、Jacobian）。
2. 拓扑排序：为什么 DFS 后序逆序是正确的反向顺序？
3. `run_backward` 的逐步执行：用一个具体计算图走一遍。
4. 梯度累加：多后继节点的梯度如何求和。
5. `no_grad` 上下文：为什么反向中的中间运算不能再建图。
6. 单线程 vs PyTorch 多线程引擎的对比。

读完第三章，你就能在脑内完整模拟一次 `loss.backward()` 的执行过程。
