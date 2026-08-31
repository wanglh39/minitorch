# 第六章 优化器系统：从 SGD 到 Adam 的数学推导与工程实现

> 上一章我们让 `nn.Module` 把成百上千个参数组织成一个整体。现在问题来了：
> 拿到参数和它们的梯度之后，**怎么更新参数才能让 loss 下降**？这就是优化器要回答的问题。
> 本章从最朴素的 `p -= lr * grad` 出发，一步步推导出动量、Nesterov、Adam，
> 讲清每个公式"为什么这么写"，再对照 minitorch 的 `optim/sgd.py` 和 `optim/adam.py` 逐行实现。
> 最后讲 `param_groups`、优化器 state、LR Scheduler 的工程设计。

---

## 6.1 本章目标

读完本章后，你应当能够：

1. 从泰勒展开推导出梯度下降更新公式 `p ← p - lr * ∇L`，并说出学习率过大时为什么会震荡发散。
2. 写出动量法的更新规则 `v ← μv + g; p ← p - lr * v`，解释动量如何"累积历史梯度方向"以加速收敛。
3. 区分 Nesterov 动量和普通动量：Nesterov 在"预测位置"算梯度，具有"前瞻"效果，推导出等价的 `g ← g + μ * v` 形式。
4. 推导 Adam 的完整更新规则：一阶矩 `m`、二阶矩 `v`、bias correction `m̂`、`v̂`、自适应步长 `lr * m̂ / (√v̂ + eps)`。
5. 解释 bias correction 为什么必要：初始几步 `m` 和 `v` 严重偏向 0，不修正会步长过小。
6. 用 `param_groups` 给不同参数组设不同学习率（如冻结 backbone、只调 head），并说出 `defaults` 与 group 的合并规则。
7. 说明优化器 `state` 为什么用 `id(p)` 做 key，以及这种设计的代价（参数被回收后 id 可能复用）。
8. 解释为什么优化器的更新是 in-place numpy 操作、不参与计算图（否则会建一张"更新图"导致内存泄漏）。
9. 推导 `weight_decay`（L2 正则）的梯度贡献 `g ← g + wd * p`，并说出它和 AdamW 的区别。
10. 画出 `LambdaLR`、`StepLR`、`CosineAnnealingLR` 三种调度器的 lr 曲线，说出各自适用场景。

---

## 6.2 原理铺垫：从泰勒展开到梯度下降

### 6.2.1 一维泰勒展开与下降方向

假设损失 `L(p)` 是一维可导函数，当前参数 `p`，我们想找一个新点 `p' = p + Δ` 让 `L` 下降。一阶泰勒展开：

```
L(p + Δ) ≈ L(p) + L'(p) * Δ
```

要 `L(p + Δ) < L(p)`，需要 `L'(p) * Δ < 0`。即 `Δ` 与 `L'(p)` 异号。最简单的选择：

```
Δ = -lr * L'(p)     (lr > 0)
```

于是：

```
p ← p - lr * L'(p)
```

这就是**梯度下降**。`lr`（learning rate）控制步长。

多维情况完全一样，`L'(p)` 换成梯度向量 `∇L(p)`：

```
p ← p - lr * ∇L(p)
```

### 6.2.2 学习率过大为什么会发散

二阶泰勒展开：

```
L(p + Δ) ≈ L(p) + L'(p) * Δ + 0.5 * L''(p) * Δ²
```

代入 `Δ = -lr * L'(p)`：

```
L(p + Δ) - L(p) ≈ -lr * L'(p)² + 0.5 * lr² * L''(p) * L'(p)²
                = -lr * L'(p)² * (1 - 0.5 * lr * L''(p))
```

要下降，需要 `1 - 0.5 * lr * L''(p) > 0`，即：

```
lr < 2 / L''(p)
```

`L''(p)` 是曲率。曲率越大，能容许的 `lr` 越小。如果 `lr` 超过 `2/L''`，每步"跨过"最优解越来越远，**震荡发散**。

::: tip 直觉
把损失函数想象成一个碗。曲率大=碗窄陡。学习率太大=步子太大，在碗壁之间来回撞，越撞越远。学习率太小=步子太小，半天走不到碗底。
:::

### 6.2.3 随机梯度下降（SGD）

全批量梯度下降每次用全部数据算梯度，太贵。**SGD** 每次抽一个小批量（mini-batch）算梯度近似：

```
g_batch ≈ ∇L(p)    （无偏估计，但方差大）
p ← p - lr * g_batch
```

小批量的好处：① 算得快；② 随机性帮助跳出鞍点。代价：梯度有噪声，收敛轨迹抖动。

### 6.2.4 动量法：累积历史方向

SGD 在"窄而长"的损失面（条件数大）上走 zigzag：一个方向震荡，另一个方向缓慢。**动量**的思路：把历史梯度方向指数加权平均，震荡方向正负抵消，稳定方向累加。

```
v ← μ * v + g          # v 是动量缓冲，μ 是衰减系数（典型 0.9）
p ← p - lr * v
```

展开看 `v` 是历史梯度的加权和：

```
v_t = g_t + μ * g_{t-1} + μ² * g_{t-2} + ...
```

震荡方向：`g_t` 和 `g_{t-1}` 反号，`g_t + μ * g_{t-1} ≈ (1 - μ) * g_t`，被削弱。
稳定方向：`g_t` 和 `g_{t-1}` 同号，`g_t + μ * g_{t-1} ≈ (1 + μ) * g_t`，被放大。

效果：在稳定方向上"加速"，在震荡方向上"减速"。

### 6.2.5 Nesterov 动量：前瞻梯度

普通动量在**当前位置**算梯度。Nesterov 在**预测的下一位置**算梯度：

```
v ← μ * v + ∇L(p - lr * μ * v)     # 在 p - lr*μ*v 处算梯度
p ← p - lr * v
```

直觉：既然下一步要走到 `p - lr * v`，不如先去那里看看梯度，"前瞻"一下再决定方向。

这个形式要算 `∇L(p - lr * μ * v)`，得在前向时改参数位置，工程上不便。Sutskever et al. (2013) 给出等价改写：

```
v ← μ * v + g                       # g 是在当前位置算的梯度
g ← g + μ * v                       # 修正梯度
p ← p - lr * g
```

这就是 minitorch `sgd.py` 里 `if nesterov: grad = grad + momentum * buf` 的来源。

### 6.2.6 Adam：自适应步长

SGD/动量对所有参数用同一个 `lr`。但不同参数的曲率可能差几个数量级（稀疏特征 vs 密集特征）。**Adam** 给每个参数单独算"有效步长"。

核心思想：维护梯度的**一阶矩**（均值）和**二阶矩**（未中心化方差）的指数滑动平均：

```
m ← β₁ * m + (1 - β₁) * g           # 一阶矩（梯度均值）
v ← β₂ * v + (1 - β₂) * g²          # 二阶矩（梯度平方均值）
```

更新方向用 `m`（去噪的梯度），步长除以 `√v`（梯度大的参数步长小，梯度小的参数步长大）：

```
p ← p - lr * m / (√v + eps)
```

### 6.2.7 Bias Correction：为什么初始几步要修正

初始 `m = 0, v = 0`。第一步：

```
m₁ = (1 - β₁) * g₁        # 严重偏向 0
v₁ = (1 - β₂) * g₁²       # 严重偏向 0
```

`m₁ / √v₁ = (1-β₁)/(1-β₂) * sign(g₁)`，但量级被 `(1-β₁)` 缩小。典型 `β₁=0.9, β₂=0.999`：`m₁` 只有真实值的 10%，`v₁` 只有 0.1%。步长会被严重低估，初期几乎不更新。

**修正**：除以 `(1 - βᵗ)` 抵消初始化偏置：

```
m̂ = m / (1 - β₁ᵗ)
v̂ = v / (1 - β₂ᵗ)
p ← p - lr * m̂ / (√v̂ + eps)
```

`t` 是步数。当 `t → ∞`，`βᵗ → 0`，修正因子 → 1，不再影响。只在初期起作用。

### 6.2.8 化简：PyTorch 的实际写法

把 `m̂ / √v̂` 展开：

```
m̂ / √v̂ = [m / (1 - β₁ᵗ)] / √[v / (1 - β₂ᵗ)]
        = m / (1 - β₁ᵗ) * √(1 - β₂ᵗ) / √v
        = [lr / (1 - β₁ᵗ)] * m / [√v / √(1 - β₂ᵗ) + eps]
```

定义：

```
step_size = lr / (1 - β₁ᵗ)
denom = √v / √(1 - β₂ᵗ) + eps
p ← p - step_size * m / denom
```

这就是 minitorch `adam.py` 最后几行的写法。`eps` 放在 `denom` 里而非外面，是为了避免 `v` 很小时 `√v` 接近 0 导致除以 0。`eps` 典型 1e-8。

### 6.2.9 weight_decay：L2 正则

在损失上加 `0.5 * wd * ||p||²`，梯度变成：

```
∇(L + 0.5 * wd * ||p||²) = ∇L + wd * p
```

所以更新前先修正梯度：

```
g ← g + wd * p
p ← p - lr * g
```

这就是 `weight_decay` 的实现。它让参数趋向小值，防止过拟合。

::: warning Adam vs AdamW
对 SGD，L2 正则等价于在更新里加 `wd * p`。但对 Adam，因为 `v` 累积了 `(g + wd*p)²`，L2 正则和"权重衰减"（直接 `p ← (1 - lr*wd) * p`）不再等价。**AdamW**（Loshchilov & Hutter 2017）把权重衰减解耦出来：

```
g ← g + wd * p    # Adam 的 L2 正则（耦合）
# vs
p ← (1 - lr * wd) * p    # AdamW 的解耦权重衰减
```

minitorch 的 Adam 用前者（耦合 L2），与 PyTorch 的 `Adam` 一致。要用 AdamW 得另写。
:::

### 6.2.10 param_groups：不同参数不同配置

迁移学习常见场景：backbone 用小 lr，新加的 head 用大 lr。`param_groups` 让优化器持有多个参数组，每组独立 `lr`/`weight_decay` 等：

```python
opt = SGD(
    [
        {"params": backbone.parameters(), "lr": 1e-4},
        {"params": head.parameters(), "lr": 1e-3},
    ],
    lr=1e-2,   # defaults，被组里的 lr 覆盖
)
```

`step()` 时遍历每个 group，用 group 自己的 `lr` 更新该组的参数。

### 6.2.11 优化器 state：为什么用 `id(p)` 做 key

动量缓冲 `v`、Adam 的 `m`/`v`/`step` 是**每个参数一份**的。存哪里？

- **存在参数上**：`p._momentum_buffer = ...`。污染 Tensor，且不同优化器字段冲突。
- **存在优化器上，用 `id(p)` 做 key**：`self.state[id(p)] = {"momentum_buffer": ...}`。Tensor 保持干净，优化器各自管自己的 state。

minitorch 用后者。代价：① 参数对象被 GC 回收后 `id` 可能被新对象复用，state 错乱（实践中参数生命周期通常长于优化器，问题不大）；② 不能直接 `pickle` 优化器（要存参数指针）。

PyTorch 也是 `id(p)` 做 key，但额外维护 `param_to_id` 映射解决部分问题。

### 6.2.12 为什么优化器不参与计算图

更新 `p -= lr * g` 如果走 autograd，会建一张"更新图"，每次 step 都把参数变成"非叶子"，`p.grad_fn` 不为 None，下次 backward 行为错乱。所以优化器必须**绕过 autograd**，直接 in-place 改 storage：

```python
param = p._numpy_view()       # 拿到 numpy 视图
param -= lr * grad            # in-place 改底层 storage
```

PyTorch 用 `torch.no_grad()` 上下文。minitorch 直接操作 numpy 视图，等价于 no_grad。

### 6.2.13 LR Scheduler：在训练中调 lr

学习率不是常数。常见策略：

- **LambdaLR**：`lr = base * lambda(epoch)`，最灵活。
- **StepLR**：每 `step_size` 个 epoch 乘 `gamma`（如每 30 epoch 乘 0.1）。
- **CosineAnnealingLR**：`lr = eta_min + (base - eta_min) * (1 + cos(π * t / T)) / 2`，余弦衰减到 `eta_min`。

调度器持有 optimizer 引用，每次 `scheduler.step()` 改 optimizer 的 `param_groups[i]["lr"]`。

---

## 6.3 设计决策与权衡

| 决策                              | 选择                              | 理由                                                | 代价                                              |
| ----------------------------------- | --------------------------------- | --------------------------------------------------- | ------------------------------------------------- |
| 优化器持有参数方式                  | `param_groups` 列表，每组含 params | 支持不同组不同配置                                  | 结构嵌套，遍历要两层循环                          |
| `defaults` 与 group 合并            | `{**defaults, **group}`           | group 里没指定的用 defaults                         | 用户要理解覆盖优先级                              |
| state 的 key                        | `id(p)`                           | Tensor 保持干净，多优化器不冲突                     | id 复用风险；不能直接 pickle                      |
| 更新方式                            | in-place numpy `param -= lr * grad`| 绕过 autograd，不建更新图                           | 不能用 autograd 做 meta-learning（要 `torch.no_grad` 配合）|
| Adam 的 `eps` 位置                  | `denom = √v/√bc2 + eps`           | 与 PyTorch 一致，数值稳定                           | 与原论文 `√v + eps` 略不同，差异可忽略            |
| bias correction 实现                | `step_size = lr / (1 - β₁ᵗ)`      | 化简后少算一次除法                                  | 公式不直观，要推导才能看懂                        |
| `weight_decay` 位置                 | 在算 `m`/`v` 之前修正 `g`         | 与 PyTorch `Adam` 一致（耦合 L2）                   | 不是 AdamW 的解耦形式                             |
| `zero_grad` 设 `None`               | 与 Module.zero_grad 一致          | 省内存，语义清晰                                    | 优化器和模块都有 zero_grad，可能混淆              |
| Nesterov 等价改写                   | `g ← g + μ * v`                   | 不用在前向时改参数位置                              | 公式不直观，要 Sutskever 推导才懂                 |
| Scheduler 持有 optimizer 引用       | `self.optimizer = optimizer`      | 改 lr 直接改 optimizer.param_groups                 | 循环引用（optimizer 不持 scheduler，所以没问题）  |
| `last_epoch` 从 -1 开始              | `__init__` 末尾调 `self.step()`   | 让 epoch=0 时 lr 就是 `get_lr(0)`                   | 初学者困惑为什么构造时就 step 一次                 |
| `StepLR` 用 `//` 整除                | `gamma ** (epoch // step_size)`   | 简单直观                                            | 衰减点不连续                                      |
| `CosineAnnealingLR` 的 `eta_min`    | 默认 0                            | 衰减到 0                                            | 训练后期 lr 过小，可能不收敛                      |
| 无 `foreach` 向量化                 | 逐参数循环                        | 实现简单                                            | 大模型慢（PyTorch 用 foreach 批量更新）           |

---

## 6.4 代码逐行实现

### 6.4.1 `optimizer.py`：Optimizer 基类

```python
class Optimizer:
    def __init__(self, params: Iterable, defaults: dict):
        self.defaults = defaults                  # 默认配置（lr/momentum/...）
        self.param_groups: list[dict] = []        # 参数组列表
        params = list(params)
        if len(params) == 0:
            raise ValueError("optimizer got empty param list")
        if isinstance(params[0], dict):
            # 用户传的是 [{"params": ..., "lr": ...}, ...]
            for group in params:
                self.param_groups.append({**defaults, **group})
        else:
            # 用户传的是 [p1, p2, ...]
            self.param_groups = [{"params": params, **defaults}]
        self.state: dict = {}                     # 优化器状态，key=id(p)
```

逐行解读：

- **`defaults`**：存默认配置。子类 `SGD.__init__` 把 `lr/momentum/...` 打包成 `defaults` 传上来。
- **`isinstance(params[0], dict)`**：判断用户传的是"参数列表"还是"参数组列表"。如果第一个元素是 dict，说明是 `[{"params": [...], "lr": ...}, ...]` 形式。
- **`{**defaults, **group}`**：合并。group 里的 key 覆盖 defaults 里的同名 key。所以组里指定 `lr` 就用组的，没指定就用 defaults 的。
- **`self.state = {}`**：优化器状态字典，key 是 `id(p)`，value 是该参数的状态（如 `{"momentum_buffer": ...}`）。

### 6.4.2 `optimizer.py`：`zero_grad` 与 `step`

```python
def zero_grad(self) -> None:
    for group in self.param_groups:
        for p in group["params"]:
            p.grad = None                         # 设 None 而非 0

def step(self) -> None:
    raise NotImplementedError                      # 子类实现
```

逐行解读：

- **`zero_grad` 遍历所有组所有参数**：不能只清一组。设 `None` 而非零张量，理由见 5.4.8。
- **`step` 抽象方法**：子类必须实现。这遵循模板方法模式。

### 6.4.3 `sgd.py`：SGD 构造与参数校验

```python
class SGD(Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)
```

逐行解读：

- **参数校验**：`lr < 0` 无意义；`momentum` 要在 `[0, 1)`；Nesterov 要求 `momentum > 0` 且 `dampening == 0`。这些和 PyTorch 完全一致。
- **`dampening`**：minitorch 支持但默认 0。dampening 让动量更新变成 `v ← μ*v + (1-d)*g`，d=0 就是标准动量。Nesterov 要求 d=0。
- **`defaults` 打包**：所有配置塞进 dict，传给 `Optimizer.__init__`。

### 6.4.4 `sgd.py`：`step` 逐参数更新

```python
def step(self) -> None:
    for group in self.param_groups:               # ① 遍历参数组
        lr = group["lr"]
        momentum = group["momentum"]
        dampening = group["dampening"]
        weight_decay = group["weight_decay"]
        nesterov = group["nesterov"]

        for p in group["params"]:                 # ② 遍历组内参数
            if p.grad is None:
                continue                          # 没梯度的跳过（冻结参数）
            grad = p.grad._numpy_view()           # ③ 拿梯度的 numpy 视图
            param = p._numpy_view()               # ④ 拿参数的 numpy 视图

            if weight_decay != 0:
                grad = grad + weight_decay * param   # ⑤ L2 正则修正梯度

            if momentum != 0:
                state = self.state.setdefault(id(p), {})   # ⑥ 取/建该参数的 state
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = np.zeros_like(param)
                buf = state["momentum_buffer"]
                buf *= momentum                   # ⑦ v ← μ * v
                buf += grad if momentum == 0 else (1 - dampening) * grad  # ⑧ v += (1-d)*g

                if nesterov:
                    grad = grad + momentum * buf  # ⑨ Nesterov: g ← g + μ*v
                else:
                    grad = buf                    # ⑩ 普通动量: 用 v 当梯度

            param -= lr * grad                    # ⑪ p ← p - lr * g
```

逐行解读：

- **① 遍历 param_groups**：每组用各自的 `lr`/`momentum` 等。
- **② 遍历组内参数**：逐个更新。
- **③④ `_numpy_view()`**：拿到底层 numpy 数组的视图，不拷贝。后续 in-place 操作直接改底层 storage。
- **⑤ weight_decay**：`g ← g + wd * p`。注意这里 `grad` 重新绑定到一个新数组（`grad + wd*param` 是新数组），不污染原始 `p.grad`。
- **⑥ `setdefault(id(p), {})`**：如果 state 里没有该参数的条目，建空 dict。key 是 `id(p)`。
- **⑦ `buf *= momentum`**：in-place 乘，等价于 `v = μ * v`。
- **⑧ `buf += (1-d)*g`**：in-place 加。`dampening=0` 时就是 `buf += g`。注意条件 `grad if momentum == 0 else (1-d)*grad`——其实 `momentum != 0` 已经进了这个分支，所以 `momentum == 0` 永远为 False，这行等价于 `buf += (1-dampening)*grad`。这是 PyTorch 源码的历史遗留写法。
- **⑨ Nesterov**：`g ← g + μ * v`。注意用的是**当前 grad**（已含 weight_decay 修正）加 `μ * buf`。
- **⑩ 普通动量**：直接用 `buf`（即 `v`）当更新方向。
- **⑪ `param -= lr * grad`**：in-place 减。`param` 是 numpy 视图，改它就是改 `p` 的底层 storage。**不经过 autograd**。

::: tip 为什么 `grad = grad + weight_decay * param` 不用 in-place？
如果写 `grad += weight_decay * param`，会**污染 `p.grad` 的底层 storage**。下次 backward 累加梯度时，基础值变成了被 weight_decay 修正过的，错误。所以用 `grad = grad + ...` 重新绑定到新数组，原 `p.grad` 不动。
:::

### 6.4.5 `adam.py`：Adam 构造

```python
class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
```

逐行解读：

- **默认 `lr=1e-3`**：Adam 的典型学习率，比 SGD 的 `1e-2` 小，因为 Adam 有自适应步长。
- **`betas=(0.9, 0.999)`**：β₁ 控制一阶矩衰减（梯度均值），β₂ 控制二阶矩衰减（梯度方差）。β₂ 比 β₁ 接近 1，因为二阶矩要更长时间才稳定。
- **`eps=1e-8`**：防除零。
- **校验 `0 <= beta < 1`**：保证矩衰减是稳定的指数滑动平均。

### 6.4.6 `adam.py`：`step` 逐参数更新

```python
def step(self) -> None:
    for group in self.param_groups:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad._numpy_view()
            param = p._numpy_view()

            if weight_decay != 0:
                grad = grad + weight_decay * param    # ① L2 正则

            state = self.state.setdefault(id(p), {})
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = np.zeros_like(param)      # ② 一阶矩 m，初始 0
                state["exp_avg_sq"] = np.zeros_like(param)   # ③ 二阶矩 v，初始 0

            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]                # m
            exp_avg_sq = state["exp_avg_sq"]          # v

            exp_avg *= beta1                          # ④ m ← β₁ * m
            exp_avg += (1 - beta1) * grad             # ⑤ m += (1-β₁) * g
            exp_avg_sq *= beta2                       # ⑥ v ← β₂ * v
            exp_avg_sq += (1 - beta2) * (grad * grad) # ⑦ v += (1-β₂) * g²

            bias_correction1 = 1 - beta1**step        # ⑧ 1 - β₁ᵗ
            bias_correction2 = 1 - beta2**step        # ⑨ 1 - β₂ᵗ
            step_size = lr / bias_correction1         # ⑩ lr / (1 - β₁ᵗ)
            bias_correction2_sqrt = np.sqrt(bias_correction2)

            denom = np.sqrt(exp_avg_sq) / bias_correction2_sqrt + eps  # ⑪ √v/√(1-β₂ᵗ) + eps
            param -= step_size * exp_avg / denom      # ⑫ p ← p - step_size * m / denom
```

逐行解读：

- **① weight_decay**：与 SGD 一样，在算矩之前修正梯度。这是耦合 L2 正则。
- **②③ 初始化 `exp_avg`/`exp_avg_sq`**：Adam 的两个矩缓冲，形状与参数一致，初始 0。命名 `exp_avg`（指数加权平均）是 PyTorch 的命名习惯。
- **④⑤ 一阶矩更新**：`m ← β₁*m + (1-β₁)*g`。分两步 in-place：先乘 β₁，再加 `(1-β₁)*g`。
- **⑥⑦ 二阶矩更新**：`v ← β₂*v + (1-β₂)*g²`。注意是 `grad * grad`（逐元素平方）。
- **⑧⑨ bias correction 因子**：`1 - βᵗ`。`step` 从 1 开始（前面 `+= 1`），所以第一步 `1 - β¹`，不是 `1 - β⁰ = 0`。
- **⑩ `step_size`**：`lr / (1 - β₁ᵗ)`。把 bias correction 的除法合并进 step_size。
- **⑪ `denom`**：`√v / √(1 - β₂ᵗ) + eps`。注意 `eps` 在**加完 bias correction 之后**才加，与 PyTorch 一致。原论文是 `√v̂ + eps`，等价。
- **⑫ 更新**：`p -= step_size * m / denom`。`step_size * exp_avg / denom` 就是 `lr * m̂ / (√v̂ + eps)`。

::: warning 为什么 `exp_avg *= beta1` 用 in-place？
`exp_avg` 是 `state["exp_avg"]` 引用的 numpy 数组。in-place 修改它，state 里的也跟着变（同一对象）。如果写 `exp_avg = beta1 * exp_avg`，`exp_avg` 重新绑定到新数组，state 里的还是旧的，下次 step 拿到的还是初始 0——bug。所以矩更新必须 in-place。
:::

### 6.4.7 `lr_scheduler.py`：基类

```python
class LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self._initial_lr = [g["lr"] for g in optimizer.param_groups]   # 保存初始 lr
        self.step()                    # ① 立即 step 一次，让 epoch=0 的 lr 生效

    def get_lr(self) -> list[float]:
        raise NotImplementedError

    def step(self) -> None:
        self.last_epoch += 1
        values = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, values, strict=True):
            group["lr"] = lr           # ② 改 optimizer 的 lr
```

逐行解读：

- **`_initial_lr`**：保存每个 group 的初始 lr。调度器基于初始 lr 算当前 lr，而非基于"上一步的 lr"——避免累积误差。
- **① 构造时立即 step**：`last_epoch=-1`，`step()` 后变成 0，`get_lr(0)` 算出 epoch 0 的 lr。这样用户构造完调度器，optimizer 的 lr 就是 epoch 0 的值，无需手动调一次。
- **② `group["lr"] = lr`**：直接改 optimizer 的 param_groups。下次 `optimizer.step()` 就用新 lr。
- **`strict=True`**：Python 3.10+ 的 zip 参数，要求两个迭代器长度一致，否则报错。防御 param_groups 数量变化。

### 6.4.8 `lr_scheduler.py`：LambdaLR

```python
class LambdaLR(LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        epoch = self.last_epoch
        return [base * self.lr_lambda(epoch) for base in self._initial_lr]
```

逐行解读：

- **`lr_lambda`**：用户传一个函数 `epoch -> factor`，lr = base * factor。
- **每个 group 算一次**：`_initial_lr` 是列表，每个 group 一个 base。所以返回列表。

用法：

```python
sched = LambdaLR(opt, lr_lambda=lambda e: 0.95 ** e)
# epoch 0: lr = base * 1.0
# epoch 1: lr = base * 0.95
# epoch 2: lr = base * 0.9025
```

### 6.4.9 `lr_scheduler.py`：StepLR

```python
class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base * factor for base in self._initial_lr]
```

逐行解读：

- **`//` 整除**：`epoch // step_size` 算出"已经衰减了几次"。如 `step_size=30`，epoch 0-29 算 0 次，30-59 算 1 次。
- **`gamma ** 次数`**：每次衰减乘 gamma。典型 `gamma=0.1`，每 30 epoch lr 乘 0.1。

曲线：

```
epoch:   0   29  30  59  60  89  90
lr:     1.0  1.0 0.1 0.1 0.01 0.01 0.001
```

阶梯下降。

### 6.4.10 `lr_scheduler.py`：CosineAnnealingLR

```python
class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch == 0:
            return self._initial_lr
        return [
            self.eta_min + (base - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base in self._initial_lr
        ]
```

逐行解读：

- **`T_max`**：半个周期的长度。epoch `T_max` 时 cos(π) = -1，lr = `eta_min`。
- **`eta_min`**：lr 衰减的下界。默认 0。
- **公式**：`eta_min + (base - eta_min) * (1 + cos(π * t / T)) / 2`。当 t=0，cos(0)=1，lr=base；t=T，cos(π)=-1，lr=eta_min；t=2T，cos(2π)=1，lr=base（回升）。所以这是**半周期**衰减，t=T 到 2T 会回升。通常只用 0 到 T 这一段。
- **`if last_epoch == 0: return _initial_lr`**：避免数值误差。epoch 0 时公式给 `eta_min + (base-eta_min)*1 = base`，但浮点可能有微小误差，直接返回 initial_lr 更精确。

曲线（T_max=10, eta_min=0, base=1）：

```
epoch:  0    1    2    3    4    5    6    7    8    9   10
lr:    1.0 0.98 0.90 0.79 0.65 0.50 0.35 0.21 0.10 0.02 0.0
```

平滑余弦下降，比 StepLR 更平滑，常用于训练后期精细调优。

---

## 6.5 完整示例

### 6.5.1 SGD 基本更新

```python
import numpy as np
from minitorch import Tensor
from minitorch.optim import SGD

p = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
p.requires_grad = True
p.grad = Tensor.from_numpy(np.array([0.1, 0.2, 0.3]))
opt = SGD([p], lr=0.5)
opt.step()
print(p.numpy())   # [0.95 1.9  2.85]   ← p - 0.5 * grad
```

### 6.5.2 SGD + 动量

```python
p = Tensor.from_numpy(np.array([1.0]))
p.requires_grad = True
p.grad = Tensor.from_numpy(np.array([1.0]))
opt = SGD([p], lr=0.1, momentum=0.9)
opt.step()
print(p.numpy())   # [0.9]   ← v=1.0, p=1-0.1*1=0.9

p.grad = Tensor.from_numpy(np.array([1.0]))
opt.step()
print(p.numpy())   # [0.71]  ← v=0.9*1+1=1.9, p=0.9-0.1*1.9=0.71
```

### 6.5.3 SGD + Nesterov

```python
p = Tensor.from_numpy(np.array([1.0]))
p.requires_grad = True
p.grad = Tensor.from_numpy(np.array([1.0]))
opt = SGD([p], lr=0.1, momentum=0.9, nesterov=True)
opt.step()
# buf = 1.0, nesterov_grad = 1.0 + 0.9*1.0 = 1.9, p = 1 - 0.1*1.9 = 0.81
print(p.numpy())   # [0.81]
```

### 6.5.4 SGD + weight_decay

```python
p = Tensor.from_numpy(np.array([2.0]))
p.requires_grad = True
p.grad = Tensor.from_numpy(np.array([1.0]))
opt = SGD([p], lr=0.1, weight_decay=0.5)
opt.step()
# effective_grad = 1.0 + 0.5*2.0 = 2.0, p = 2 - 0.1*2.0 = 1.8
print(p.numpy())   # [1.8]
```

### 6.5.5 Adam 收敛二次函数

```python
p = Tensor.from_numpy(np.array([5.0, -3.0]))
p.requires_grad = True
opt = Adam([p], lr=0.1)

for _ in range(500):
    p.grad = Tensor.from_numpy(2.0 * p.numpy())   # ∇(p²) = 2p
    opt.step()
print(p.numpy())   # [~0.0, ~0.0]   ← 收敛到最小值点 0
```

### 6.5.6 Adam 第一步的 bias correction

```python
p = Tensor.from_numpy(np.array([0.0]))
p.requires_grad = True
p.grad = Tensor.from_numpy(np.array([1.0]))
opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
opt.step()
# m=0.1, v=0.001, m̂=0.1/0.1=1.0, v̂=0.001/0.001=1.0
# p = 0 - 0.1 * 1.0 / (1.0 + 1e-8) ≈ -0.1
print(p.numpy())   # [-0.1]
```

### 6.5.7 param_groups 独立 lr

```python
p1 = Tensor.from_numpy(np.array([1.0])); p1.requires_grad = True
p2 = Tensor.from_numpy(np.array([1.0])); p2.requires_grad = True
p1.grad = Tensor.from_numpy(np.array([1.0]))
p2.grad = Tensor.from_numpy(np.array([1.0]))
opt = SGD(
    [{"params": [p1], "lr": 0.1}, {"params": [p2], "lr": 0.01}],
    lr=0.05,   # defaults，被组的 lr 覆盖
)
opt.step()
print(p1.numpy())   # [0.9]    ← 用组 lr=0.1
print(p2.numpy())   # [0.99]   ← 用组 lr=0.01
```

### 6.5.8 StepLR 调度

```python
p = Tensor.from_numpy(np.array([1.0])); p.requires_grad = True
opt = SGD([p], lr=1.0)
sched = StepLR(opt, step_size=2, gamma=0.1)
print(opt.param_groups[0]["lr"])   # 1.0
sched.step()                       # epoch 1
print(opt.param_groups[0]["lr"])   # 1.0
sched.step()                       # epoch 2 → 衰减
print(opt.param_groups[0]["lr"])   # 0.1
sched.step()                       # epoch 3
print(opt.param_groups[0]["lr"])   # 0.1
sched.step()                       # epoch 4 → 再衰减
print(opt.param_groups[0]["lr"])   # 0.01
```

### 6.5.9 CosineAnnealingLR 曲线

```python
p = Tensor.from_numpy(np.array([1.0])); p.requires_grad = True
opt = SGD([p], lr=1.0)
T_max = 10
sched = CosineAnnealingLR(opt, T_max=T_max, eta_min=0.0)
lrs = [opt.param_groups[0]["lr"]]
for _ in range(T_max):
    sched.step()
    lrs.append(opt.param_groups[0]["lr"])
print(lrs)
# [1.0, 0.9755, 0.9045, 0.7939, 0.6545, 0.5, 0.3455, 0.2061, 0.0955, 0.0245, 0.0]
```

### 6.5.10 端到端：训练 + 调度

```python
from minitorch.nn import Linear, MSELoss

model = ...  # 某个 MLP
opt = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
sched = CosineAnnealingLR(opt, T_max=100)

for epoch in range(100):
    pred = model(X)
    loss = MSELoss()(pred, Y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()                    # 每个 epoch 末尾调
    if epoch % 20 == 0:
        print(f"epoch {epoch}, lr {opt.param_groups[0]['lr']:.4f}, loss {loss.item():.4f}")
```

---

## 6.6 常见陷阱

### 陷阱 1：忘记 `zero_grad`，梯度累加导致发散

```python
for epoch in range(100):
    loss = crit(model(x), y)
    loss.backward()      # ← 梯度累加到上一步的梯度上！
    opt.step()           # 用的是累加梯度，方向错
```

**症状**：loss 不降反升，或震荡剧烈。

**解决**：每次 `backward` 前调 `opt.zero_grad()`（或 `model.zero_grad()`）。

### 陷阱 2：`scheduler.step()` 顺序错

PyTorch 1.1+ 改了语义：`scheduler.step()` 应在 `optimizer.step()` **之后**调。如果在之前调，第一个 epoch 的 lr 就被跳过了。

```python
# 正确
opt.step()
sched.step()

# 错误（会跳过 epoch 0 的 lr）
sched.step()
opt.step()
```

### 陷阱 3：Adam 的 `eps` 放错位置

原论文写 `lr * m̂ / (√v̂ + eps)`，PyTorch 写 `lr * m̂ / (√v / √(1-β₂ᵗ) + eps)`。两者在 `eps` 很小时等价，但如果 `eps` 较大（如 1e-3），差异显现。minitorch 跟 PyTorch。

### 陷阱 4：`weight_decay` 在 Adam 里是耦合 L2

```python
opt = Adam([p], lr=0.1, weight_decay=0.01)   # 这是 L2 正则，不是 AdamW
```

如果想要解耦的权重衰减（AdamW），minitorch 没有现成的，得自己写或扩展。

### 陷阱 5：param_groups 里的 `params` 必须是列表

```python
opt = SGD(model.parameters(), lr=0.1)   # model.parameters() 是生成器
```

`Optimizer.__init__` 里 `params = list(params)` 物化了，所以这没问题。但如果用户后续 `list(model.parameters())` 改了顺序，优化器里的顺序是**构造时的快照**，不会跟着变。

### 陷阱 6：`id(p)` 复用导致 state 错乱

```python
p = Tensor.from_numpy(np.array([1.0])); p.requires_grad = True
opt = SGD([p], lr=0.1, momentum=0.9)
p.grad = Tensor.from_numpy(np.array([1.0])); opt.step()
old_id = id(p)

del p                            # p 被回收
p2 = Tensor.from_numpy(np.array([2.0]))   # 可能复用 old_id
# 如果 p2 进了某个新优化器，且 id(p2) == old_id，会拿到旧 state
```

实践中参数生命周期长于优化器，问题不大。但要避免在训练中途换参数。

### 陷阱 7：Nesterov 要求 `dampening=0`

```python
SGD([p], lr=0.1, momentum=0.9, nesterov=True, dampening=0.5)
# ValueError: Nesterov momentum requires a momentum and zero dampening
```

构造时会校验。Nesterov 的等价改写 `g ← g + μ*v` 只在 `dampening=0` 时成立。

### 陷阱 8：Adam 的 `step` 从 0 开始但 bias correction 用 `step` 从 1

minitorch 里 `state["step"] += 1` 在算 bias correction 之前，所以第一步 `step=1`，`1 - β¹`。如果改成先算再 `+= 1`，第一步 `step=0`，`1 - β⁰ = 0`，除零。**顺序很重要**。

### 陷阱 9：调度器构造时立即 step 一次

```python
opt = SGD([p], lr=1.0)
sched = StepLR(opt, step_size=2, gamma=0.1)
print(opt.param_groups[0]["lr"])   # 1.0 ← 已经是 epoch 0 的 lr
```

调度器构造函数末尾调了一次 `step()`，`last_epoch` 从 -1 变 0。如果用户不知道，再手动调一次 `sched.step()`，会跳到 epoch 1。

### 陷阱 10：`CosineAnnealingLR` 超过 `T_max` 会回升

```python
sched = CosineAnnealingLR(opt, T_max=10)
for _ in range(20):
    sched.step()
# epoch 10: lr=0, epoch 11-20: lr 回升到 1.0
```

如果不想回升，用 `CosineAnnealingWarmRestarts` 或在 `T_max` 处停。minitorch 没实现 WarmRestarts。

---

## 6.7 与真实 PyTorch 对照

| minitorch                                       | PyTorch                                                | 差异说明                                  |
| ----------------------------------------------- | ------------------------------------------------------ | ----------------------------------------- |
| `Optimizer` 持有 `param_groups`/`state`/`defaults` | 同                                                     | 一致                                      |
| `param_groups` 合并 `{**defaults, **group}`      | 同                                                     | 一致                                      |
| `state` 用 `id(p)` 做 key                        | 同                                                     | 一致                                      |
| `SGD` 更新规则（动量/Nesterov/weight_decay）     | 同                                                     | 一致                                      |
| `Adam` 更新规则（bias correction/eps 位置）      | 同                                                     | 一致                                      |
| `zero_grad` 设 `None`                            | 同（0.4+）                                             | 一致                                      |
| `LRScheduler` 基类 + `get_lr`/`step`             | 同                                                     | 一致                                      |
| `LambdaLR`/`StepLR`/`CosineAnnealingLR`          | 同                                                     | 一致                                      |
| 无 `foreach` 向量化                              | 1.13+ 支持 foreach 批量更新                            | minitorch 逐参数循环，慢                 |
| 无 CUDA fused kernel                             | 有 `_fused_adam` 等                                    | minitorch 纯 numpy                        |
| 无 `AdamW`                                       | 有                                                     | minitorch 未实现解耦权重衰减             |
| 无 `Optimizer.add_param_group`                  | 有，运行时加参数组                                     | minitorch 未实现                          |
| 无 `optimizer.state_dict()`/`load_state_dict()` | 有，保存优化器状态                                     | minitorch 未实现（重启训练会丢动量）     |
| 无 `gradient_ascent_view`                        | 有，调试用                                             | minitorch 未实现                          |
| 无 `differentiable` 参数                         | 有，支持 meta-learning                                  | minitorch 不支持可微优化器               |
| `dampening` 条件写法有历史遗留                   | 同（PyTorch 也这样写）                                 | 一致                                      |
| `Scheduler` 构造时 step 一次                     | 同（1.1+）                                             | 一致                                      |
| 无 `ReduceLROnPlateau`                           | 有，按指标降 lr                                        | minitorch 未实现                          |
| 无 `OneCycleLR`                                  | 有，super-convergence                                  | minitorch 未实现                          |

### 6.7.1 关键差异详解：`foreach` 向量化

PyTorch 1.13+ 引入 `foreach` 选项：把多个参数的更新**批量**做成一个向量化操作，而非逐参数循环。对大模型（几千个参数张量），循环开销显著。minitorch 逐参数循环，简单但慢。

### 6.7.2 关键差异详解：优化器 state 的保存

PyTorch 的 `optimizer.state_dict()` 把 `state` 序列化（key 从 `id(p)` 转成参数索引），可以保存到磁盘。重启训练时 `optimizer.load_state_dict()` 恢复动量/Adam 矩。minitorch 没实现，重启训练会丢 state，前几步表现像冷启动。

---

## 6.8 历史背景

### 6.8.1 梯度下降的起源

梯度下降最早可追溯到 Cauchy (1847)。机器学习里用 SGD 训练神经网络可追溯到 1980s 的 backprop（Rumelhart, Hinton, Williams 1986）。早期全批量，后来 mini-batch 成为主流。

### 6.8.2 动量法的引入

Polyak (1964) 提出重球法（heavy ball method），即动量法。Sutskever et al. (2013) 在深度学习语境下系统研究动量，发现 Nesterov 动量在训练 RNN 时显著更好，并把 Nesterov 改写成工程上便利的形式（即 minitorch 用的 `g ← g + μ*v`）。

### 6.8.3 Adam 的提出与争议

Kingma & Ba (2014) 提出 Adam，结合了 AdaGrad（自适应步长）和 RMSProp（指数滑动平均）的思想。Adam 一度是深度学习最流行的优化器。

但 Reddi et al. (2018) 指出 Adam 在某些凸问题上不收敛，提出 **AMSGrad**（取 `v̂` 的历史最大值）。后续又有 AdaBound、RAdam 等修正。PyTorch 的 `Adam` 是原版，`AdamW` 是解耦权重衰减版（Loshchilov & Hutter 2017）。

### 6.8.4 LR Scheduler 的演化

早期训练用固定 lr。AlexNet (2012) 开始用 step decay（每若干 epoch 乘 0.1）。ResNet (2015) 引入 warmup + cosine。Transformer (2017) 强调 warmup 对 Adam 的必要性（否则初期二阶矩不稳）。OneCycleLR (Smith 2017) 提出 super-convergence：lr 先升后降，配合动量反向变化。

### 6.8.5 PyTorch 优化器 API 的稳定

PyTorch 0.1 确立了 `Optimizer`/`param_groups`/`state` 的设计，沿用至今。1.1 改了 `scheduler.step()` 的语义（从 `optimizer.step()` 之前调改成之后）。1.13 加 `foreach`。2.0 加 fused kernel。API 基本向后兼容。

---

## 6.9 练习题

### 练习 1：手动推导 Adam 第一步

设 `lr=0.1, β₁=0.9, β₂=0.999, eps=1e-8`，初始 `p=0, g=1`。手算 Adam 第一步后 `p` 的值。

??? 解答
    `m₁ = 0.1*1 = 0.1`，`v₁ = 0.001*1 = 0.001`。
    `step=1`，`bc1 = 1-0.9 = 0.1`，`bc2 = 1-0.999 = 0.001`。
    `step_size = 0.1/0.1 = 1.0`。
    `denom = √0.001/√0.001 + 1e-8 = 1 + 1e-8 ≈ 1`。
    `p = 0 - 1.0 * 0.1 / 1 = -0.1`。
    所以 `p ≈ -0.1`。bias correction 让第一步的更新量等于 `lr * sign(g)`，不被初始偏置缩小。
???

### 练习 2：实现 AdamW

AdamW 把权重衰减解耦：先 `p ← (1 - lr*wd) * p`，再做 Adam 更新（不把 `wd*p` 加进梯度）。实现一个。

??? 解答
    ```python
    class AdamW(Adam):
        def step(self):
            for group in self.param_groups:
                lr, wd = group["lr"], group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None: continue
                    param = p._numpy_view()
                    if wd != 0:
                        param *= (1 - lr * wd)        # 解耦权重衰减
                    # 然后调用 Adam 的更新（但不做 weight_decay 修正梯度）
                    # ... 复用 Adam 的矩更新逻辑，跳过 grad += wd*param
    ```
    关键区别：`weight_decay` 作用在参数上（乘性衰减），不作用在梯度上（加性修正）。
???

### 练习 3：为什么 `step_size = lr / (1 - β₁ᵗ)` 而非分开除

把 `m̂ / √v̂` 写成 `m / (1-β₁ᵗ) / √(v / (1-β₂ᵗ))` 和写成 `step_size * m / denom` 比，哪个省计算？

??? 解答
    分开除要两次除法（`m / bc1` 和 `√v / √bc2`），合并后 `step_size = lr/bc1` 是标量除法（一次），`denom = √v/√bc2 + eps` 是逐元素除法（一次），总共一次标量除 + 一次逐元素除。原写法要两次逐元素除。所以合并写法省一次逐元素除法，对大张量有意义。
???

### 练习 4：`StepLR` vs `CosineAnnealingLR` 选哪个

训练一个 CNN 100 epoch，前 50 快速下降，后 50 精细调优。用 StepLR 还是 Cosine？

??? 解答
    用 CosineAnnealingLR(T_max=100)。Cosine 曲线前段下降慢（lr 高，快速探索），中段下降快，后段下降慢（lr 低，精细调优）。StepLR 是阶梯，过渡生硬。Cosine 的平滑过渡通常更好。如果要在某 epoch 大幅降 lr（如 plateau），用 ReduceLROnPlateau。
???

### 练习 5：`id(p)` 做 key 的替代方案

设计一个不用 `id(p)` 做 key 的 state 存储方案，并说出它的优缺点。

??? 解答
    方案：在 `Parameter` 上加一个 `_optimizer_state` 字典，`p._optimizer_state["momentum_buffer"] = ...`。
    优点：① 没有 id 复用问题；② 参数带着 state 走，pickle 参数就 pickle 了 state。
    缺点：① 污染 Parameter 类；② 多个优化器对同一参数，state 字典冲突（要再加一层 `id(opt)` 做 key）；③ Parameter 类要预先声明这个字段。
    PyTorch 选 `id(p)` 是因为参数对象通常长期存活，id 复用风险低，且保持 Parameter 类干净。
???

---

## 6.10 关键测试解读

`tests/test_optim.py` 的每个测试都在防御一类 bug：

### `test_sgd_basic_update`

```python
p = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
p.grad = Tensor.from_numpy(np.array([0.1, 0.2, 0.3]))
opt = SGD([p], lr=0.5); opt.step()
assert np.allclose(p.numpy(), [0.95, 1.9, 2.85])
```

**防御**：基本更新 `p -= lr * grad`。如果 `param -= lr * grad` 写成 `param += lr * grad`，会挂。

### `test_sgd_momentum`

```python
opt = SGD([p], lr=0.1, momentum=0.9); opt.step()
assert np.allclose(p.numpy(), [0.9])
p.grad = ...; opt.step()
assert np.allclose(p.numpy(), [0.9 - 0.1 * (0.9 + 1.0)])
```

**防御**：动量缓冲跨 step 持久化。第二步 `v = 0.9*1 + 1 = 1.9`。如果 state 没存住（每步重建），第二步 `v` 会是 1.0，挂。

### `test_sgd_nesterov`

```python
opt = SGD([p], lr=0.1, momentum=0.9, nesterov=True); opt.step()
buf1 = 1.0
nesterov_grad1 = 1.0 + 0.9 * buf1
assert np.allclose(p.numpy(), [1.0 - 0.1 * nesterov_grad1])
```

**防御**：Nesterov 的 `g ← g + μ*v` 修正。如果漏了 `+ momentum * buf`，会用普通动量的更新，挂。

### `test_sgd_weight_decay`

```python
opt = SGD([p], lr=0.1, weight_decay=0.5); opt.step()
effective_grad = 1.0 + 0.5 * 2.0
assert np.allclose(p.numpy(), [2.0 - 0.1 * effective_grad])
```

**防御**：weight_decay 修正梯度。如果漏了 `grad += wd * param`，会用原梯度，挂。

### `test_adam_converges_on_quadratic`

```python
opt = Adam([p], lr=0.1)
for _ in range(500):
    p.grad = Tensor.from_numpy(2.0 * p.numpy())
    opt.step()
assert np.allclose(p.numpy(), [0.0, 0.0], atol=1e-5)
```

**防御**：Adam 在 `L = p²` 上收敛到 0。这是端到端测试，覆盖矩更新、bias correction、自适应步长的整体正确性。如果 bias correction 漏了，初期步长过小，500 步收敛不到 1e-5。

### `test_adam_bias_correction_first_step`

```python
opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8); opt.step()
m_hat = 1.0; v_hat = 1.0
expected = 0.0 - 0.1 * m_hat / (math.sqrt(v_hat) + 1e-8)
assert np.allclose(p.numpy(), [expected])
```

**防御**：第一步 bias correction 后 `m̂=1, v̂=1`。如果漏了 correction，`m=0.1, v=0.001`，`p = -0.1*0.1/√0.001 ≈ -0.316`，与 expected=-0.1 不符。

### `test_adam_state_persists`

```python
opt.step()
state = opt.state[id(p)]
assert state["step"] == 1
p.grad = ...; opt.step()
assert state["step"] == 2
```

**防御**：state 跨 step 持久化，`step` 递增。如果每步重建 state，`step` 永远是 1。

### `test_param_group_independent_lr`

```python
opt = SGD([{"params": [p1], "lr": 0.1}, {"params": [p2], "lr": 0.01}], lr=0.05)
opt.step()
assert np.allclose(p1.numpy(), [0.9])    # 用 0.1
assert np.allclose(p2.numpy(), [0.99])   # 用 0.01
```

**防御**：不同组用不同 lr。如果 `{**defaults, **group}` 合并错（如反过来），组的 lr 会被 defaults 覆盖，两个都用 0.05。

### `test_lambda_lr` / `test_step_lr` / `test_cosine_annealing_lr_curve`

**防御**：各调度器的公式正确。`test_cosine_annealing_lr_curve` 特别检查 `lrs[0]==1.0`、`lrs[T_max]==0.0`、中间每点符合余弦公式。

### `test_scheduler_with_param_groups`

```python
opt = SGD([{"params": [p1], "lr": 1.0}, {"params": [p2], "lr": 2.0}], lr=1.0)
sched = StepLR(opt, step_size=1, gamma=0.5); sched.step()
assert np.isclose(opt.param_groups[0]["lr"], 0.5)   # 1.0 * 0.5
assert np.isclose(opt.param_groups[1]["lr"], 1.0)   # 2.0 * 0.5
```

**防御**：调度器同时调所有 group 的 lr，按各自初始 lr 比例衰减。如果调度器只调第一个 group，第二个 lr 不变。

---

## 6.11 优劣势总结

### 优势

1. **数学严谨**：SGD/Adam 的实现严格遵循推导，bias correction、Nesterov 改写都正确。
2. **与 PyTorch API 一致**：`param_groups`/`state`/`zero_grad`/调度器 API 完全对齐，迁移成本低。
3. **不污染 Tensor**：state 存优化器，Parameter 类保持干净。
4. **绕过 autograd**：in-place numpy 更新，不建更新图，内存安全。
5. **调度器解耦**：调度器独立类，持 optimizer 引用，可组合多种策略。

### 劣势

1. **逐参数循环**：无 `foreach` 向量化，大模型慢。
2. **无优化器 state 保存**：重启训练丢动量/矩，前几步像冷启动。
3. **无 AdamW**：只有耦合 L2 正则的 Adam，不符合现代推荐。
4. **`id(p)` 复用风险**：参数被回收后 id 可能复用，state 错乱（实践罕见）。
5. **无 fused kernel**：纯 numpy，无 GPU 加速。
6. **Nesterov 要求 dampening=0**：不能灵活组合，与 PyTorch 一致的限制。

---

## 6.12 下一章预告

本章我们解决了"参数怎么更新"。下一章 **第七章 损失与训练循环** 将回答：

- 有了模型和优化器，怎么算 loss？`MSELoss` 和 `CrossEntropyLoss` 的数学推导是什么？
- 为什么 `CrossEntropyLoss = LogSoftmax + NLLLoss`？拆开有什么好处？
- `log_softmax` 怎么用"减 max"技巧保证数值稳定？不这么做会怎样？
- `ReLU` 在 `x=0` 处梯度是什么？minitorch 怎么处理？
- `functional` API 和 `Module` 包装两种风格有什么区别？什么时候用哪个？
- 完整训练循环 `forward → loss → zero_grad → backward → step` 每一步做什么？
- 端到端跑一个 MLP 回归和分类，loss 怎么下降？

我们将从损失函数的数学推导开始，对照 minitorch 的 `nn/functional.py` 和 `nn/loss.py` 逐行实现，最后跑通完整训练循环。
