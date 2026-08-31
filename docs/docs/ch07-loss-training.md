# 第七章 损失与训练循环：从 MSELoss 到端到端训练

> 前两章我们解决了"参数怎么组织"和"参数怎么更新"。现在还差最后一块拼图：
> **怎么衡量模型的好坏**？这就是损失函数。本章从最简单的 MSE 推导到分类用的 CrossEntropy，
> 讲清 `LogSoftmax + NLLLoss` 拆分的数值稳定动机，再串起完整的训练循环
> `forward → loss → zero_grad → backward → step`，最后端到端跑一个 MLP 回归和分类。

---

## 7.1 本章目标

读完本章后，你应当能够：

1. 推导 `MSELoss = mean((pred - target)²)` 的梯度 `∂L/∂pred = 2*(pred - target)/N`，并说出为什么回归任务用 MSE。
2. 解释分类任务为什么用 CrossEntropy 而非 MSE：MSE 对 logits 的梯度在预测正确时趋于 0（梯度消失），CrossEntropy 不会。
3. 推导 `CrossEntropyLoss = LogSoftmax + NLLLoss` 的等价性，并说出拆开实现的原因：暴露"减 max"数值稳定技巧。
4. 手写数值稳定的 `log_softmax`：`x - max(x) - log(sum(exp(x - max(x))))`，并解释不减 max 时 `exp(1000)` 会 inf。
5. 推导 `Softmax` 的雅可比，写出 `backward` 公式 `grad_x = softmax * (grad - sum(grad * softmax, axis=dim, keepdim))`。
6. 说明 `ReLU` 在 `x=0` 处梯度未定义，minitorch 取 `x > 0`（即 0 处梯度为 0），与 PyTorch 一致。
7. 区分 `functional` API（`F.relu(x)`，无状态）和 `Module` 包装（`nn.ReLU()`，可注册、可有 hooks）两种风格，说出各自适用场景。
8. 写出完整训练循环的五个步骤，解释每一步为什么在那个顺序：`forward → loss → zero_grad → backward → step`。
9. 用 minitorch 端到端训练一个 MLP 回归（MSE + SGD）和分类（CrossEntropy + Adam），观察 loss 下降。
10. 解释 `model.eval()` 与 `no_grad` 的区别：前者切模式（影响 Dropout/BN），后者关 autograd（不建图），eval 时两者都要。

---

## 7.2 原理铺垫：损失函数是什么

### 7.2.1 损失函数的角色

损失函数 `L(pred, target)` 衡量"预测与目标的差距"。训练就是找参数 `θ` 让 `L(f_θ(x), y)` 在训练集上最小。

对损失函数的要求：

1. **非负**：`L ≥ 0`，预测完全正确时 `L = 0`。
2. **可微**：能算梯度才能用梯度下降（不可微的要用次梯度或平滑近似）。
3. **凸性（可选）**：凸损失保证全局最优。神经网络里损失对参数通常非凸，但对预测可以凸（如 MSE 对 pred 凸）。

### 7.2.2 MSELoss：回归任务的默认

均方误差：

```
L = (1/N) * Σ (pred_i - target_i)²
```

梯度：

```
∂L/∂pred_i = 2 * (pred_i - target_i) / N
```

为什么回归用 MSE：

- **凸**：对 pred 是凸的，优化友好。
- **可微**：处处可微。
- **惩罚大误差**：平方放大大的偏差，模型更关注难样本。
- **高斯似然**：如果噪声是高斯的，最小化 MSE 等价于最大似然。

代价：**对离群点敏感**（平方放大）。如果数据有重尾噪声，用 MAE 或 Huber 更稳。

### 7.2.3 为什么分类不用 MSE

考虑二分类，target=1，pred=0（logit，未过 sigmoid）。用 MSE：

```
L = (sigmoid(0) - 1)² = (0.5 - 1)² = 0.25
∂L/∂pred = 2 * (sigmoid(pred) - 1) * sigmoid'(pred)
         = 2 * (-0.5) * 0.25 = -0.25
```

当 pred 很负（如 -10），`sigmoid(-10) ≈ 0`，`sigmoid'(-10) ≈ 0`：

```
∂L/∂pred = 2 * (0 - 1) * 0 = 0   ← 梯度消失！
```

预测严重错误（pred=-10, target=1），但梯度为 0，学不动。这是 MSE 用于分类的根本问题。

CrossEntropy 解决这个：

```
L = -log(sigmoid(pred))    ← target=1 时
∂L/∂pred = sigmoid(pred) - 1 = -1   ← pred=-10 时梯度≈-1，不消失
```

### 7.2.4 CrossEntropy 的推导

多分类：`pred` 是 logits 向量（未归一化），`target` 是类别索引。CrossEntropy：

```
L = -log(softmax(pred)[target])
  = -log(exp(pred[target]) / Σ_j exp(pred[j]))
  = -pred[target] + log(Σ_j exp(pred[j]))
```

梯度：

```
∂L/∂pred_j = softmax(pred)_j - (1 if j == target else 0)
```

即 `softmax - onehot(target)`。预测正确时（softmax 接近 onehot），梯度趋于 0；预测错误时，梯度大。**梯度自然反映错误程度**。

### 7.2.5 为什么拆成 LogSoftmax + NLLLoss

直接实现 `L = -pred[target] + log(Σ exp(pred))` 的问题：`log(Σ exp(pred))` 当 pred 大时（如 1000）`exp(1000)` 溢出 inf。

**LogSoftmax** 用"减 max"技巧：

```
log_softmax(x)_i = x_i - max(x) - log(Σ_j exp(x_j - max(x)))
```

令 `m = max(x)`：

```
log(Σ exp(x)) = log(Σ exp(x - m) * exp(m)) = m + log(Σ exp(x - m))
```

`x - m` 的最大元素是 0，`exp(0) = 1`，其余 ≤ 1，`Σ exp(x-m)` 在 `[1, C]`（C 是类别数），`log` 在 `[0, log C]`，**数值稳定**。

所以拆成两步：

```
log_probs = LogSoftmax(logits)    # 数值稳定的 log softmax
loss = NLLLoss(log_probs, target) # -mean(log_probs[target])
```

`NLLLoss`（Negative Log Likelihood）只是取 `log_probs[target]` 取负求均值，不涉及 exp，不会溢出。

::: tip 拆分的教学价值
拆开不是为了性能，是为了**暴露数值稳定技巧**。学生能看到 `LogSoftmax` 里减 max 的代码，理解为什么。如果合成一个 `cross_entropy` 黑盒，技巧藏在内部，教学不友好。PyTorch 也这样拆，`F.cross_entropy` 内部调 `log_softmax` + `nll_loss`。
:::

### 7.2.6 LogSoftmax 的 backward

`y = log_softmax(x)`，即 `y_i = x_i - log(Σ exp(x))`。求雅可比：

```
∂y_i/∂x_j = δ_ij - softmax(x)_j
```

给上游梯度 `g`：

```
grad_x_j = Σ_i g_i * (δ_ij - softmax_j)
         = g_j - softmax_j * Σ_i g_i
         = g_j - softmax_j * sum(g)
```

向量化：

```
grad_x = g - softmax * sum(g, axis=dim, keepdim=True)
```

这就是 minitorch `LogSoftmax.backward` 的公式。`softmax` 在 forward 时算好存起来（`ctx.save_for_backward`）。

### 7.2.7 Softmax 的 backward

`y = softmax(x)`，`y_i = exp(x_i) / Σ exp(x)`。雅可比：

```
∂y_i/∂x_j = y_i * (δ_ij - y_j)
```

给上游梯度 `g`：

```
grad_x_j = Σ_i g_i * y_i * (δ_ij - y_j)
         = y_j * g_j - y_j * Σ_i g_i * y_i
         = y_j * (g_j - Σ_i g_i * y_i)
         = y_j * (g_j - dot(g, y))
```

向量化：

```
grad_x = softmax * (g - sum(g * softmax, axis=dim, keepdim=True))
```

这就是 minitorch `Softmax.backward`。注意 `Softmax` 和 `LogSoftmax` 的 backward 不同，别混。

### 7.2.8 NLLLoss 的 backward

`L = -(1/N) * Σ log_probs[i, target_i]`。对 `log_probs[i, j]` 求梯度：

```
∂L/∂log_probs[i, j] = -1/N   if j == target_i
                    = 0      otherwise
```

即一个稀疏矩阵，只有 `target` 位置是 `-1/N`。给上游梯度 `g`（标量，因为 loss 是标量）：

```
grad_log_probs[i, j] = -g/N   if j == target_i
                     = 0      otherwise
```

minitorch 实现：

```python
grad = np.zeros((n, C))
grad[np.arange(n), target] = -1.0 / n
return Tensor.from_numpy(grad * grad_output.item()), None
```

`grad_output.item()` 是上游梯度（标量）。返回 `(g, None)`：对 `log_probs` 的梯度和对 `target` 的梯度（target 是整数索引，不算梯度，返回 None）。

### 7.2.9 ReLU 的分段梯度

`ReLU(x) = max(0, x)`。分段：

```
x > 0:  ReLU(x) = x,  ReLU'(x) = 1
x < 0:  ReLU(x) = 0,  ReLU'(x) = 0
x = 0:  不可导
```

`x = 0` 处不可导，但实践中几乎不会正好是 0（浮点）。约定 `ReLU'(0) = 0`（minitorch 用 `x > 0` 严格大于，所以 0 处梯度 0）或 `= 1`（用 `x >= 0`）。PyTorch 用 `x > 0`，minitorch 一致。

backward：

```python
mask = (x._numpy_view() > 0).astype(np.float64)
return Tensor.from_numpy(grad_output._numpy_view() * mask)
```

`mask` 是 0/1 矩阵，逐元素乘上游梯度。

### 7.2.10 functional API vs Module 包装

两种风格：

```python
# functional API（无状态）
out = F.relu(x)
out = F.mse_loss(pred, target)
out = F.cross_entropy(logits, target)

# Module 包装（有状态，可注册到模型）
relu = nn.ReLU()
out = relu(x)
crit = nn.MSELoss()
loss = crit(pred, target)
```

区别：

- **functional**：纯函数，无状态，不继承 `Module`，不能 `model.parameters()` 收集（因为没有参数）。适合无参数层（ReLU、loss）。
- **Module**：继承 `Module`，可以注册到模型，可以有 hooks，可以有参数（如 `PReLU` 的 `alpha`）。适合有参数层、需要统一接口的场景。

minitorch 的 `loss.py` 把 functional 包成 Module：

```python
class MSELoss(Module):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)
```

这样用户可以 `crit = MSELoss(); loss = crit(pred, target)`，与 `model(x)` 风格一致。

### 7.2.11 完整训练循环

```
for epoch in range(epochs):
    for x, y in dataloader:
        pred = model(x)              # ① 前向：建计算图
        loss = criterion(pred, y)    # ② 算损失：图上加一个 loss 节点
        optimizer.zero_grad()        # ③ 清梯度：把所有 p.grad 设 None
        loss.backward()              # ④ 反向：沿图走，填 p.grad
        optimizer.step()             # ⑤ 更新：用 p.grad 改 p
```

每步详解：

- **① forward**：`model(x)` 触发 `__call__` → `forward`，逐层算并建 autograd 图（每个算子调 `Function.apply` 新建 `Node`）。
- **② loss**：`criterion(pred, y)` 也是前向，把 loss 节点挂到图上。loss 是标量。
- **③ zero_grad**：清上一步的梯度。必须**在 backward 之前**清，否则梯度累加。可以放 ② 之前或 ③ 位置，习惯放 ③。
- **④ backward**：从 loss 开始沿图反向走，把每个叶子参数的 `grad` 填上。详见第三、四章。
- **⑤ step**：优化器用 `p.grad` 更新 `p`。in-place 改 storage，不建图。详见第六章。

顺序不能乱：

- `zero_grad` 必须在 `backward` 前（否则清掉刚算的）。
- `backward` 必须在 `step` 前（否则没梯度怎么更新）。
- `step` 必须在 `backward` 后。

### 7.2.12 eval 模式与 no_grad

测试/推理时：

```python
model.eval()           # 切 eval 模式：Dropout 恒等，BN 用 running_mean
with torch.no_grad():  # 关 autograd：不建图，省内存省时间
    pred = model(x)
```

两者作用不同：

- **`model.eval()`**：递归设 `self.training = False`。影响**层的行为**（Dropout、BN）。不关 autograd。
- **`no_grad()`**：关 autograd 引擎，不建计算图。影响**梯度计算**。不切模式。

eval 时两者都要：`eval()` 让层行为正确，`no_grad()` 省内存。minitorch 没实现 `no_grad` 上下文（简化），但 eval 时直接 `model(x)` 也能跑，只是会建图（浪费内存）。

---

## 7.3 设计决策与权衡

| 决策                                | 选择                              | 理由                                                | 代价                                              |
| ----------------------------------- | --------------------------------- | --------------------------------------------------- | ------------------------------------------------- |
| CrossEntropy 实现                    | 拆 `LogSoftmax` + `NLLLoss`       | 暴露数值稳定技巧，可复用                            | 两个算子，API 表面大                              |
| `log_softmax` 数值稳定               | 减 max                            | 防 exp 溢出                                          | 多算一次 max                                      |
| `Softmax` 也减 max                   | 同                                | 一致性，也防溢出                                    | 多算一次 max                                      |
| `NLLLoss` 接受 log_probs 而非 probs  | 语义清晰                          | 名字就叫 NLL（负对数似然）                          | 用户要懂先 log_softmax 再 nll                     |
| `cross_entropy` functional           | 内部调 LogSoftmax + NLLLoss       | 复用，不重复实现                                    | 多一次函数调用                                    |
| `MSELoss` 实现                       | 用现有算子组合 `(diff**2).mean()` | 自动建图，无需自定义 backward                       | 不能特殊优化（如 fused kernel）                   |
| `ReLU` 在 0 处梯度                   | `x > 0`（0 处梯度 0）             | 与 PyTorch 一致                                     | 0 处不可导，约定                                  |
| functional vs Module                 | 两者都提供                        | 灵活：无状态用 functional，要注册用 Module          | 代码重复（Module 只是包装）                       |
| `target` 类型                        | 接受 Tensor 或 list/ndarray        | 用户方便                                            | 内部要转                                          |
| `NLLLoss.backward` 返回 `(g, None)`  | 对 target 返回 None               | target 是整数索引，不算梯度                         | 要理解 None 语义                                  |
| `loss` 是标量                        | `Tensor.from_numpy(np.array(loss))`| 统一类型                                            | 标量 Tensor 略重                                  |
| 无 `reduction` 参数                  | 默认 mean                         | 简化                                                | 不能选 sum/none                                   |
| 无 `ignore_index`                    | -                                 | 简化                                                | 不能跳过 padding                                  |
| 无 `label_smoothing`                 | -                                 | 简化                                                | 现代分类常用                                      |
| 训练循环不封装                       | 用户手写五步                      | 教学透明                                            | 工程上重复                                        |

---

## 7.4 代码逐行实现

### 7.4.1 `functional.py`：Relu

```python
class Relu(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)                              # ① 存输入，backward 要用
        return Tensor.from_numpy(np.maximum(0, x._numpy_view()))  # ② max(0, x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]                              # ③ 取出存的 x
        mask = (x._numpy_view() > 0).astype(np.float64)       # ④ 0/1 掩码
        return Tensor.from_numpy(grad_output._numpy_view() * mask)  # ⑤ 逐元素乘
```

逐行解读：

- **`Function` 子类**：自定义算子，实现 `forward`/`backward`。详见第三章。
- **① `save_for_backward(x)`**：把 `x` 存进 `ctx`，反向时能取到。ReLU 的 backward 需要知道哪里是正哪里是负。
- **② `np.maximum(0, x)`**：逐元素 max。`maximum` 是 numpy 函数，不是 Python 内置。
- **③ `ctx.saved_tensors[0]`**：取出存的 `x`。`saved_tensors` 是 tuple。
- **④ `(x > 0).astype(float)`**：布尔数组转 float64，True→1.0，False→0.0。`x > 0` 严格大于，所以 `x=0` 处 mask=0，梯度 0。
- **⑤ `grad_output * mask`**：链式法则，上游梯度乘局部梯度。

### 7.4.2 `functional.py`：LogSoftmax

```python
class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        arr = x._numpy_view()
        max_val = np.max(arr, axis=dim, keepdims=True)        # ① 减 max 技巧
        shifted = arr - max_val
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))  # ② log(sum(exp))
        result = shifted - log_sum_exp                        # ③ log_softmax = x - max - logsumexp
        ctx.save_for_backward(Tensor.from_numpy(np.exp(result)))  # ④ 存 softmax（=exp(log_softmax)）
        ctx.dim = dim
        return Tensor.from_numpy(result)

    @staticmethod
    def backward(ctx, grad_output):
        softmax = ctx.saved_tensors[0]
        dim = ctx.dim
        sum_grad = np.sum(grad_output._numpy_view(), axis=dim, keepdims=True)  # ⑤ sum(g)
        grad_x = grad_output._numpy_view() - softmax._numpy_view() * sum_grad  # ⑥ g - softmax * sum(g)
        return Tensor.from_numpy(grad_x)
```

逐行解读：

- **① `max_val`**：沿 `dim` 取 max，`keepdims=True` 保持形状以便广播。这是数值稳定的关键。
- **② `log_sum_exp`**：`log(Σ exp(x - max))`。因为减了 max，`exp` 的最大值是 `exp(0)=1`，不会溢出。
- **③ `result = shifted - log_sum_exp`**：即 `x - max - log(Σ exp(x-max))` = `x - log(Σ exp(x))` = `log_softmax(x)`。数学等价，数值稳定。
- **④ `save exp(result)`**：存 softmax。因为 `exp(log_softmax) = softmax`，backward 要用。直接存 `result` 再 exp 也行，但存 softmax 省一次 exp。
- **⑤ `sum_grad`**：`Σ_i g_i`，沿 `dim` 求和，`keepdims` 保持形状广播。
- **⑥ `grad_x = g - softmax * sum(g)`**：LogSoftmax 的 backward 公式（见 7.2.6）。

::: warning 为什么不直接 `np.log(np.sum(np.exp(arr)))`？
如果 `arr = [1000, 1001, 1002]`，`np.exp(1000)` = inf，`sum` = inf，`log(inf)` = inf。结果全是 inf 或 nan。减 max 后 `arr - max = [-2, -1, 0]`，`exp` 都在 `[exp(-2), 1]`，安全。
:::

### 7.4.3 `functional.py`：Softmax

```python
class Softmax(Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        arr = x._numpy_view()
        max_val = np.max(arr, axis=dim, keepdims=True)        # ① 减 max
        exp_arr = np.exp(arr - max_val)
        result = exp_arr / np.sum(exp_arr, axis=dim, keepdims=True)  # ② 归一化
        ctx.save_for_backward(Tensor.from_numpy(result))      # ③ 存 softmax
        ctx.dim = dim
        return Tensor.from_numpy(result)

    @staticmethod
    def backward(ctx, grad_output):
        softmax = ctx.saved_tensors[0]
        dim = ctx.dim
        grad = grad_output._numpy_view()
        sm = softmax._numpy_view()
        dot = np.sum(grad * sm, axis=dim, keepdims=True)      # ④ dot(g, softmax)
        grad_x = sm * (grad - dot)                            # ⑤ softmax * (g - dot)
        return Tensor.from_numpy(grad_x)
```

逐行解读：

- **①②**：标准数值稳定的 softmax。减 max 后 exp 再归一化。
- **③**：存 softmax 结果，backward 用。
- **④ `dot = sum(g * softmax)`**：`Σ_i g_i * y_i`，沿 `dim` 求和。
- **⑤ `grad_x = softmax * (g - dot)`**：Softmax 的 backward 公式（见 7.2.7）。注意与 LogSoftmax 不同：这里是 `softmax * (g - dot)`，LogSoftmax 是 `g - softmax * sum(g)`。

### 7.4.4 `functional.py`：NLLLoss

```python
class NLLLoss(Function):
    @staticmethod
    def forward(ctx, log_probs, target):
        n = log_probs.shape[0]                                # ① batch size
        target_arr = target._numpy_view().astype(int)         # ② target 转整数索引
        lp = log_probs._numpy_view()
        loss = -np.mean(lp[np.arange(n), target_arr])         # ③ -mean(log_probs[range, target])
        ctx.n = n
        ctx.num_classes = log_probs.shape[1]
        ctx.target = target_arr
        return Tensor.from_numpy(np.array(loss))              # ④ 标量 Tensor

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.n
        target = ctx.target
        C = ctx.num_classes
        grad = np.zeros((n, C))                               # ⑤ 全零梯度
        grad[np.arange(n), target] = -1.0 / n                 # ⑥ target 位置填 -1/n
        g = Tensor.from_numpy(grad * grad_output.item())      # ⑦ 乘上游梯度
        return g, None                                        # ⑧ 对 target 返回 None
```

逐行解读：

- **① `n = shape[0]`**：batch size。假设 `log_probs` 形状 `(N, C)`。
- **② `target.astype(int)`**：target 是类别索引，转整数。
- **③ `lp[np.arange(n), target_arr]`**：花式索引。`np.arange(n)` 是 `[0,1,...,n-1]`，`target_arr` 是 `[t_0, t_1, ...]`，索引出 `[lp[0,t_0], lp[1,t_1], ...]`，即每个样本的 log_prob 在正确类别处的值。`-np.mean` 即 NLL。
- **④ 标量 Tensor**：loss 是标量，包成 0 维 Tensor。
- **⑤⑥ 稀疏梯度**：只有 `target` 位置是 `-1/n`，其余 0。`grad[range, target] = -1/n` 是批量赋值。
- **⑦ `grad_output.item()`**：上游梯度是标量，`.item()` 取出 Python float。乘到 grad 上。
- **⑧ `(g, None)`**：返回对两个输入的梯度。`log_probs` 有梯度，`target` 是整数索引不算梯度，返回 None。

### 7.4.5 `functional.py`：组合函数

```python
def relu(x: Tensor) -> Tensor:
    return Relu.apply(x)                                      # 调 Function.apply 建图

def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    return LogSoftmax.apply(x, dim=dim)

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return Softmax.apply(x, dim=dim)

def nll_loss(log_probs: Tensor, target: Tensor) -> Tensor:
    if not isinstance(target, Tensor):
        target = Tensor.from_numpy(np.asarray(target))       # target 转 Tensor
    return NLLLoss.apply(log_probs, target)

def cross_entropy(logits: Tensor, target: Tensor, dim: int = -1) -> Tensor:
    if not isinstance(target, Tensor):
        target = Tensor.from_numpy(np.asarray(target))
    log_probs = LogSoftmax.apply(logits, dim=dim)            # ① log_softmax
    return NLLLoss.apply(log_probs, target)                  # ② nll_loss

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    if not isinstance(target, Tensor):
        target = Tensor.from_numpy(np.asarray(target))
    diff = pred - target                                      # ① 用现有算子
    return (diff**2).mean()                                   # ② mean(diff²)
```

逐行解读：

- **`Relu.apply(x)`**：`Function.apply` 是类方法，建 `Node`、调 `forward`、把 `Node` 挂到输出。详见第三章。
- **`cross_entropy`**：组合 `LogSoftmax` + `NLLLoss`。两步都建图，autograd 自动链式。
- **`mse_loss`**：用现有算子 `-`、`**2`、`.mean()` 组合。无需自定义 backward，autograd 自动推。这是"用现有算子组合"的风格，与 `Relu`（自定义 backward）对比。

::: tip 两种实现损失的风格
- **自定义 backward**（`NLLLoss`、`LogSoftmax`）：手写 forward 和 backward。性能好，但要小心 backward 公式正确。
- **组合现有算子**（`mse_loss`）：用 `+`、`-`、`*`、`**`、`.mean()` 等。无需写 backward，autograd 自动。但建图节点多，性能略差。

minitorch 的 `mse_loss` 用组合，`cross_entropy` 用自定义（因为 log_softmax 的数值稳定要手动处理）。
:::

### 7.4.6 `loss.py`：Module 包装

```python
class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(pred, target)                       # 委托给 functional

class NLLLoss(Module):
    def forward(self, log_probs: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(log_probs, target)

class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(logits, target)
```

逐行解读：

- **继承 `Module`**：可以 `crit = MSELoss()`，`crit` 有 `__call__`、hooks 等。
- **`forward` 委托**：只是调 `F.xxx`。Module 包装本身无参数、无 state，纯粹为接口统一。
- **无 `__init__`**：不需要，继承 `Module.__init__` 即可（但 `Module.__init__` 不会自动调，所以严格说应该加 `def __init__(self): super().__init__()`。minitorch 这里省略了，依赖 `Module.__init__` 在首次 `__setattr__` 时 `setdefault` 建空字典——能跑但不严谨）。

::: warning 严格说应该调 super().__init__()
`MSELoss()` 没调 `super().__init__()`，`_parameters` 等不存在。但因为 `MSELoss` 不注册任何参数/子模块，且 `__getattr__` 用 `.get(..., {})` 防御了缺失，所以能跑。如果调 `crit.parameters()` 会触发 `__getattr__("_parameters")` 返回 `{}`，`yield from {}` 不 yield，正常。但这是脆弱的，建议加 `super().__init__()`。
:::

---

## 7.5 完整示例

### 7.5.1 MSELoss 前向与反向

```python
import numpy as np
from minitorch import Tensor
from minitorch.nn import functional as F

pred = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
pred.requires_grad = True
target = Tensor.from_numpy(np.array([0.0, 0.0, 0.0]))
loss = F.mse_loss(pred, target)
print(loss.item())   # (1+4+9)/3 = 4.6667

loss.backward()
print(pred.grad.numpy())   # 2*(pred-target)/3 = [0.667, 1.333, 2.0]
```

### 7.5.2 ReLU 前向与反向

```python
x = Tensor.from_numpy(np.array([-1.0, 0.5, 1.0]))
x.requires_grad = True
out = F.relu(x)
print(out.numpy())   # [0, 0.5, 1.0]

out.backward(Tensor.from_numpy(np.ones(3)))
print(x.grad.numpy())   # [0, 1, 1]   ← x<=0 处梯度 0
```

### 7.5.3 Softmax 和为 1

```python
x = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
out = F.softmax(x, dim=-1)
print(out.numpy().sum(axis=-1))   # [1.0, 1.0]
```

### 7.5.4 LogSoftmax 数值稳定

```python
x = Tensor.from_numpy(np.array([1000.0, 1001.0, 1002.0]))
out = F.log_softmax(x, dim=-1)
print(out.numpy())   # [-2.407, -1.407, -0.407]   全有限，不 inf
# 不减 max 的话：exp(1000) = inf，log(inf) = inf，结果全是 inf/nan
```

### 7.5.5 CrossEntropy 前向

```python
logits = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]))
target = Tensor.from_numpy(np.array([2, 0]))
loss = F.cross_entropy(logits, target)
print(loss.item())   # -mean([log_softmax(logits[0])[2], log_softmax(logits[1])[0]])
```

### 7.5.6 CrossEntropy 大 logit 不 NaN

```python
logits = Tensor.from_numpy(np.array([[1000.0, 1001.0, 1002.0]]))
target = Tensor.from_numpy(np.array([0]))
loss = F.cross_entropy(logits, target)
print(np.isfinite(loss.item()))   # True   ← 数值稳定生效
```

### 7.5.7 CrossEntropy 反向

```python
logits = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]))
logits.requires_grad = True
target = Tensor.from_numpy(np.array([2, 0]))
loss = F.cross_entropy(logits, target)
loss.backward()
# 梯度 = (softmax - onehot) / N
print(logits.grad.numpy())
# [[-0.245, -0.066, 0.311],
#  [ 0.245, -0.033, -0.212]]   ← 大致
```

### 7.5.8 Module 包装

```python
from minitorch.nn import MSELoss, CrossEntropyLoss

crit = CrossEntropyLoss()
logits = Tensor.from_numpy(np.array([[0.0, 0.0, 0.0]]))
target = Tensor.from_numpy(np.array([1]))
loss = crit(logits, target)
print(loss.item())   # log(3) ≈ 1.0986   ← 均匀 logits，正确类别概率 1/3
```

### 7.5.9 端到端：MLP 回归

```python
import numpy as np
from minitorch import Tensor
from minitorch.nn import Linear, Module, MSELoss
from minitorch.nn import functional as F
from minitorch.optim import SGD

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

np.random.seed(42)
X = np.random.randn(32, 4)
W_true = np.random.randn(4, 1)
Y = X @ W_true + 0.1 * np.random.randn(32, 1)

model = MLP()
opt = SGD(model.parameters(), lr=0.01)
crit = MSELoss()

losses = []
for epoch in range(200):
    pred = model(Tensor.from_numpy(X))               # ① forward
    loss = crit(pred, Tensor.from_numpy(Y))          # ② loss
    opt.zero_grad()                                  # ③ zero_grad
    loss.backward()                                  # ④ backward
    opt.step()                                       # ⑤ step
    losses.append(loss.item())

print(f"initial loss: {losses[0]:.4f}")   # ~1.2
print(f"final loss:   {losses[-1]:.4f}")  # ~0.05   ← 下降 > 50%
```

### 7.5.10 端到端：MLP 分类

```python
from minitorch.nn import Sequential, CrossEntropyLoss
from minitorch.optim import Adam

np.random.seed(42)
n = 40
X = np.random.randn(n, 4)
labels = (X[:, 0] + X[:, 1] > 0).astype(int)        # 线性可分

model = Sequential(Linear(4, 8), Linear(8, 2))      # 简单两层（无 ReLU，线性分类器）
opt = Adam(model.parameters(), lr=0.01)
crit = CrossEntropyLoss()

losses = []
for epoch in range(200):
    logits = model(Tensor.from_numpy(X))
    loss = crit(logits, Tensor.from_numpy(labels))
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())

print(f"initial loss: {losses[0]:.4f}")   # ~0.7
print(f"final loss:   {losses[-1]:.4f}")  # ~0.3   ← 下降
```

### 7.5.11 过拟合小批量验证容量

```python
np.random.seed(0)
X = np.random.randn(4, 4)                            # 只有 4 个样本
Y = np.random.randn(4, 1)

model = MLP()
opt = Adam(model.parameters(), lr=0.01)
crit = MSELoss()

initial_loss = None
for _ in range(500):
    pred = model(Tensor.from_numpy(X))
    loss = crit(pred, Tensor.from_numpy(Y))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if initial_loss is None:
        initial_loss = loss.item()

final_loss = loss.item()
print(f"initial: {initial_loss:.4f}, final: {final_loss:.4f}")
# initial: ~1.5, final: ~0.001   ← 过拟合到几乎 0
```

模型容量够大（4→8→1 有 4*8+8+8*1+1=49 参数），4 个样本能完美拟合。这验证了**优化器正确**（能下降）和**模型容量足够**。

---

## 7.6 常见陷阱

### 陷阱 1：CrossEntropy 传错输入

```python
# 错误：传了 softmax 后的值
probs = F.softmax(logits, dim=-1)
loss = F.cross_entropy(probs, target)   # ← 又 log_softmax 一次，错！

# 正确：传 logits
loss = F.cross_entropy(logits, target)
```

`cross_entropy` 内部会做 `log_softmax`，所以输入应该是 **logits**（未归一化）。

### 陷阱 2：NLLLoss 传了 logits 而非 log_probs

```python
# 错误：NLLLoss 期望 log_probs，传了 logits
loss = F.nll_loss(logits, target)   # ← 不会报错但结果错

# 正确
log_probs = F.log_softmax(logits, dim=-1)
loss = F.nll_loss(log_probs, target)
```

`NLLLoss` 不做 softmax，它只是 `-mean(log_probs[target])`。传 logits 会得到无意义的结果。

### 陷阱 3：target 形状不对

```python
# 错误：target 是 one-hot
target = np.array([[0, 1, 0], [1, 0, 0]])   # (2, 3)
loss = F.cross_entropy(logits, target)       # ← 花式索引错乱

# 正确：target 是类别索引
target = np.array([1, 0])                    # (2,)
loss = F.cross_entropy(logits, target)
```

minitorch 的 `NLLLoss` 假设 target 是 `(N,)` 的整数索引，不是 one-hot。

### 陷阱 4：忘记 `zero_grad` 梯度累加

```python
for epoch in range(100):
    loss = crit(model(x), y)
    loss.backward()      # ← 梯度累加
    opt.step()           # 用累加梯度，错
```

**症状**：loss 不降或震荡。

**解决**：`opt.zero_grad()` 在 `backward` 前。

### 陷阱 5：`backward` 前没设 `requires_grad`

```python
x = Tensor.from_numpy(np.array([1.0, 2.0]))   # 默认 requires_grad=False
out = F.relu(x)
out.backward()   # ← x.grad 是 None，因为 x 不要求梯度
```

**解决**：`x.requires_grad = True`。或者用 `Parameter`（默认 True）。

### 陷阱 6：对非标量 backward

```python
out = F.relu(x)   # out 是向量
out.backward()    # ← 报错或行为错，backward 要标量
```

**解决**：`out.sum().backward()` 或 `out.backward(Tensor.from_numpy(np.ones_like(out)))`。loss 是标量不会有这个问题。

### 陷阱 7：`model.eval()` 后忘记 `model.train()`

```python
model.eval()
# 在测试集评估
for x, y in test_loader:
    ...
# 回到训练
for x, y in train_loader:
    pred = model(x)   # ← model 还在 eval 模式！Dropout 恒等，BN 用 running_mean
    ...
```

**解决**：评估完调 `model.train()` 切回。

### 陷阱 8：logits 和 target 的 dtype 不匹配

```python
logits = Tensor.from_numpy(np.array([[1.0, 2.0]]))           # float64
target = Tensor.from_numpy(np.array([0]))                    # int64
loss = F.cross_entropy(logits, target)                        # OK，内部 astype(int)
```

minitorch 在 `NLLLoss.forward` 里 `target_arr = target._numpy_view().astype(int)`，所以 target 是 float 也能跑。但严格说 target 应该是整数。

### 陷阱 9：MSELoss 的 target 形状

```python
pred = model(x)              # (N, 1)
target = Tensor.from_numpy(Y)  # Y 是 (N,) 而非 (N, 1)
loss = F.mse_loss(pred, target)  # ← 广播可能错
```

**解决**：确保 `pred` 和 `target` 形状一致。`Y = Y.reshape(-1, 1)` 或 `pred = pred.squeeze()`。

### 陷阱 10：ReLU 在 0 处的梯度

```python
x = Tensor.from_numpy(np.array([0.0]))
x.requires_grad = True
F.relu(x).backward(Tensor.from_numpy(np.array([1.0])))
print(x.grad.numpy())   # [0.]   ← minitorch 用 x > 0，0 处梯度 0
```

如果依赖 0 处梯度为 1，会出错。但实践中 0 处几乎不出现。

---

## 7.7 与真实 PyTorch 对照

| minitorch                                       | PyTorch                                                | 差异说明                                  |
| ----------------------------------------------- | ------------------------------------------------------ | ----------------------------------------- |
| `F.relu` 自定义 backward                         | 同（但 PyTorch 用 `at::relu` C++ kernel）              | minitorch 纯 numpy                        |
| `F.log_softmax` 减 max                           | 同                                                     | 一致                                      |
| `F.softmax` 减 max                               | 同                                                     | 一致                                      |
| `F.nll_loss` 花式索引                             | 同                                                     | 一致                                      |
| `F.cross_entropy` = log_softmax + nll            | 同                                                     | 一致                                      |
| `F.mse_loss` 用算子组合                           | 同（PyTorch 也有组合实现）                              | 一致                                      |
| `MSELoss`/`CrossEntropyLoss` Module 包装         | 同                                                     | 一致                                      |
| 无 `reduction` 参数（默认 mean）                  | 有 `reduction='mean'/'sum'/'none'`                     | minitorch 简化                             |
| 无 `ignore_index`                                | 有，跳过 padding 类                                    | minitorch 简化                             |
| 无 `label_smoothing`                             | 有（0.10+）                                            | minitorch 简化                             |
| 无 `weight`（类别权重）                          | 有                                                     | minitorch 简化                             |
| target 只支持整数索引                            | 还支持 one-hot / 概率分布（0.10+）                     | minitorch 简化                             |
| `NLLLoss.backward` 返回 `(g, None)`              | 同                                                     | 一致                                      |
| `Relu` 在 0 处梯度 0                              | 同                                                     | 一致                                      |
| 无 `LeakyReLU`/`ELU`/`GELU` 等                   | 有                                                     | minitorch 只实现 ReLU                     |
| 无 `BCELoss`/`BCEWithLogitsLoss`                 | 有，二分类专用                                         | minitorch 简化                             |
| 无 `HuberLoss`/`SmoothL1Loss`                    | 有                                                     | minitorch 简化                             |
| 无 `KLDivLoss`                                   | 有                                                     | minitorch 简化                             |
| 训练循环用户手写                                  | 同（PyTorch 也不提供内置循环）                          | 一致                                      |
| 无 `no_grad` 上下文                              | 有，关 autograd                                        | minitorch 简化                             |
| 无 `GradScaler`（AMP）                           | 有，混合精度                                           | minitorch 简化                             |

### 7.7.1 关键差异详解：`reduction` 参数

PyTorch 的 `F.mse_loss(pred, target, reduction='none')` 返回逐元素 `(pred-target)²`，不 reduce。`reduction='sum'` 返回和，`'mean'` 返回均值。minitorch 只有 mean。如果要 sum，用户 `(diff**2).sum()` 自己写。

### 7.7.2 关键差异详解：`BCEWithLogitsLoss`

二分类时 PyTorch 推荐 `BCEWithLogitsLoss`（把 sigmoid + BCE 合并，数值稳定）而非 `BCELoss`（先 sigmoid 再 BCE，不稳）。minitorch 没实现，二分类用 `CrossEntropyLoss`（二类版）即可。

---

## 7.8 历史背景

### 7.8.1 损失函数的演化

- **MSE**：最古老的损失，统计里用几百年。
- **CrossEntropy**：信息论概念，由 Bridle (1990) 引入神经网络（"softmax loss"）。
- **NLL + LogSoftmax 拆分**：PyTorch 早期就有，动机是数值稳定。TF 也类似（`tf.nn.softmax_cross_entropy_with_logits` 内部减 max）。
- **Focal Loss**（Lin et al. 2017）：`(1 - p)^γ * CE`，对难样本加权。minitorch 未实现。
- **Label Smoothing**（Szegedy et al. 2016）：把 onehot 平滑成 `[ε/C, ..., 1-ε+ε/C, ..., ε/C]`，防过自信。PyTorch 0.10+ 支持。

### 7.8.2 ReLU 的崛起

早期神经网络用 sigmoid/tanh，容易梯度消失。Nair & Hinton (2010) 提出 ReLU，Krizhevsky 在 AlexNet (2012) 大量使用，ReLU 成为默认。ReLU 的优点：① 计算简单；② 正区梯度恒 1，缓解消失；③ 生物学合理性。缺点：负区"死"（Dead ReLU）。后续有 LeakyReLU、ELU、GELU 等修正。

### 7.8.3 训练循环的标准化

早期 PyTorch 训练循环全靠用户手写，没有标准。后来出现 `ignite`、`lightning` 等高层库封装循环。PyTorch 1.0+ 加了 `torch.nn.parallel.DistributedDataParallel` 但循环仍手写。PyTorch 2.0+ 的 `torch.compile` 优化循环但不改 API。minitorch 保持手写循环，教学透明。

### 7.8.4 `no_grad` 与 `eval` 的分离

早期 PyTorch 没有明确区分 `eval` 和 `no_grad`，用户经常混淆。0.4 之后明确：`eval` 切模式，`no_grad` 关 autograd。官方文档强调 eval 时两者都要。minitorch 没实现 `no_grad`（简化），但概念要讲清。

---

## 7.9 练习题

### 练习 1：手算 CrossEntropy 梯度

logits = `[1, 2, 3]`，target = `2`。手算 `∂L/∂logits`。

??? 解答
    softmax = `[exp(1), exp(2), exp(3)] / (exp(1)+exp(2)+exp(3))` = `[0.09, 0.24, 0.67]`。
    onehot(2) = `[0, 0, 1]`。
    梯度 = softmax - onehot = `[0.09, 0.24, -0.33]`。
    注意 N=1（单样本），所以不除 N。批量时除 N。
???

### 练习 2：实现 `BCEWithLogitsLoss`

二分类：`L = -mean(target * log(sigmoid(x)) + (1-target) * log(1-sigmoid(x)))`。数值稳定写法？

??? 解答
    直接算 `log(sigmoid(x))` 在 x 很负时不稳。改写：
    `log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x))`
    `log(1 - sigmoid(x)) = -softplus(x) = -log(1 + exp(x))`
    softplus 也要减 max 稳定：`softplus(x) = max(x, 0) + log(1 + exp(-|x|))`。
    实现：
    ```python
    def bce_with_logits(x, target):
        # 稳定 softplus
        abs_x = np.abs(x._numpy_view())
        softplus = np.maximum(x._numpy_view(), 0) + np.log1p(np.exp(-abs_x))
        # L = (1 - target) * x + softplus   (化简后)
        loss = (1 - target._numpy_view()) * x._numpy_view() + softplus
        return Tensor.from_numpy(np.mean(loss))
    ```
    `np.log1p(y)` = `log(1+y)`，y 接近 0 时比直接 `log(1+y)` 精确。
???

### 练习 3：为什么 `cross_entropy` 不直接实现

为什么不写一个 `cross_entropy` 自定义 Function，而要拆成 `LogSoftmax + NLLLoss`？

??? 解答
    ① **数值稳定技巧暴露**：学生能看到减 max 的代码，理解为什么。合成黑盒则技巧藏在内部。
    ② **复用**：`LogSoftmax` 和 `NLLLoss` 各自可独立用。如要算 log probabilities 就用 `LogSoftmax`；要自定义 loss 就用 `NLLLoss`。
    ③ **backward 简单**：拆开后每个的 backward 简单（LogSoftmax 的 backward 用 softmax，NLLLoss 的 backward 是稀疏索引）。合成的 backward 要合并推导，公式复杂。
    ④ **与 PyTorch 一致**：PyTorch 也这样拆，迁移友好。
???

### 练习 4：训练循环顺序

为什么是 `forward → loss → zero_grad → backward → step`？能不能 `zero_grad → forward → loss → backward → step`？

??? 解答
    能。`zero_grad` 只清 `p.grad`，不依赖 forward/loss 的结果。放最前或 loss 后都行。习惯放 loss 后是为了"算完 loss 立刻清旧梯度再算新梯度"的紧凑感。绝对不能放 `backward` 后（会清掉刚算的）或 `step` 后（无意义）。
???

### 练习 5：过拟合小批量的意义

`test_overfit_small_batch` 用 4 个样本训 500 步，要求 loss 降到初始的 1%。这测的是什么？

??? 解答
    测的是**优化器和 autograd 的整体正确性**，而非泛化。如果优化器有 bug（如梯度方向错、动量没存住），loss 降不下去。如果 autograd 有 bug（如 backward 公式错），梯度错，也降不下去。模型容量够大（49 参数 >> 4 样本），理论上能完美拟合（loss→0）。所以"能过拟合"证明"能下降"。这是 Andrew Ng 推荐的 ML 调试第一步："先确保模型能过拟合一个小批量"。
???

---

## 7.10 关键测试解读

`tests/test_loss.py` 和 `tests/test_train.py` 的每个测试都在防御一类 bug：

### `test_mse_loss_forward`

```python
pred = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
target = Tensor.from_numpy(np.array([1.0, 0.0, 4.0]))
loss = F.mse_loss(pred, target)
expected = np.mean((np.array([1, 2, 3]) - np.array([1, 0, 4])) ** 2)
assert np.isclose(loss.item(), expected)
```

**防御**：MSE 前向公式正确。如果 `(diff**2).mean()` 写成 `(diff**2).sum()`，会差 N 倍。

### `test_mse_loss_backward`

```python
pred = Tensor.from_numpy(np.array([1.0, 2.0, 3.0])); pred.requires_grad = True
target = Tensor.from_numpy(np.array([0.0, 0.0, 0.0]))
loss = F.mse_loss(pred, target); loss.backward()
expected = 2 * (np.array([1, 2, 3]) - np.array([0, 0, 0])) / 3
assert np.allclose(pred.grad.numpy(), expected)
```

**防御**：MSE 反向梯度 `2*(pred-target)/N` 正确。如果 `mean()` 的 backward 漏了 `/N`，会差 N 倍。

### `test_relu_forward` / `test_relu_backward`

```python
x = Tensor.from_numpy(np.array([-1.0, 0.0, 1.0, 2.0]))
assert np.allclose(F.relu(x).numpy(), [0, 0, 1, 2])

x = Tensor.from_numpy(np.array([-1.0, 0.5, 1.0])); x.requires_grad = True
F.relu(x).backward(Tensor.from_numpy(np.ones(3)))
assert np.allclose(x.grad.numpy(), [0, 1, 1])
```

**防御**：ReLU 前向 `max(0,x)`、反向 `x>0` 掩码。如果 mask 用 `x >= 0`，0 处梯度会是 1，第二个测试 `[0, 1, 1]` 不变（因为没有 0），但边界情况会差异。

### `test_softmax_sums_to_one`

```python
x = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
out = F.softmax(x, dim=-1)
assert np.allclose(out.numpy().sum(axis=-1), [1.0, 1.0])
```

**防御**：softmax 归一化正确，每行和为 1。如果除以 `sum` 漏了或 dim 错，和不为 1。

### `test_log_softmax_numerical_stability`

```python
x = Tensor.from_numpy(np.array([1000.0, 1001.0, 1002.0]))
out = F.log_softmax(x, dim=-1)
assert np.all(np.isfinite(out.numpy()))
```

**防御**：大 logit 不溢出。如果漏了减 max，`exp(1000)=inf`，结果 inf/nan，`isfinite` 挂。

### `test_log_softmax_backward`

```python
x = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]])); x.requires_grad = True
F.log_softmax(x, dim=-1).backward(Tensor.from_numpy(np.ones((1, 3))))
assert np.allclose(x.grad.numpy().sum(), 0.0, atol=1e-10)
```

**防御**：LogSoftmax 的 backward 性质——当上游梯度全 1 时，输入梯度之和为 0。因为 `log_softmax` 的每行和为负的 logsumexp，对 x 平移不变，所以 `Σ ∂/∂x_i = 0`。这是 backward 公式正确性的一个不变量检验。

### `test_cross_entropy_forward`

```python
logits = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]))
target = Tensor.from_numpy(np.array([2, 0]))
loss = F.cross_entropy(logits, target)
# 手算对照
x = np.array([[1, 2, 3], [1, 1, 1]])
log_probs = x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
manual = -np.mean([log_probs[0, 2], log_probs[1, 0]])
assert np.isclose(loss.item(), manual)
```

**防御**：CrossEntropy 前向与手算一致。注意手算用 `x - log(sum(exp(x)))`（不减 max），因为这里 x 小不溢出。大 logit 由 `test_cross_entropy_numerical_stability` 覆盖。

### `test_cross_entropy_backward`

```python
logits = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]))
logits.requires_grad = True
target = Tensor.from_numpy(np.array([2, 0]))
F.cross_entropy(logits, target).backward()
sm = np.exp(np.array([[1, 2, 3], [0, 1, 2]]))
sm = sm / sm.sum(axis=-1, keepdims=True)
onehot = np.zeros((2, 3)); onehot[0, 2] = 1; onehot[1, 0] = 1
expected = (sm - onehot) / 2
assert np.allclose(logits.grad.numpy(), expected)
```

**防御**：CrossEntropy 反向梯度 = `(softmax - onehot) / N`。这是端到端 backward 测试，覆盖 LogSoftmax 和 NLLLoss 的 backward 链式。

### `test_cross_entropy_module`

```python
crit = CrossEntropyLoss()
logits = Tensor.from_numpy(np.array([[0.0, 0.0, 0.0]]))
target = Tensor.from_numpy(np.array([1]))
loss = crit(logits, target)
assert np.isclose(loss.item(), np.log(3))
```

**防御**：均匀 logits 的 cross entropy = `log(C)`。因为 `softmax([0,0,0]) = [1/3, 1/3, 1/3]`，`-log(1/3) = log(3)`。这是一个**解析解**检验，比数值对照更可靠。

### `test_regression_loss_decreases`

```python
# 训练 200 步
assert losses[-1] < losses[0] * 0.5
```

**防御**：回归 loss 下降到初始的 50% 以下。这是端到端测试，覆盖 model + loss + autograd + optimizer 全链路。如果任何一环有 bug（如 backward 错、optimizer 更新方向错），loss 降不下去。

### `test_overfit_small_batch`

```python
# 4 样本训 500 步
assert final_loss < initial_loss * 0.01
```

**防御**：能过拟合小批量。证明优化器和 autograd 正确（详见练习 5）。

### `test_classification_loss_decreases`

```python
# 40 样本训 200 步
assert losses[-1] < losses[0]
```

**防御**：分类 loss 下降。比回归弱（只要求降，不要求降多少），因为线性分类器（无 ReLU）在线性不可分数据上收敛慢。这里数据线性可分（`labels = (X[:,0]+X[:,1] > 0)`），所以能降。

---

## 7.11 优劣势总结

### 优势

1. **数值稳定**：`log_softmax`/`softmax` 都减 max，大 logit 不溢出。
2. **拆分清晰**：`CrossEntropy = LogSoftmax + NLLLoss`，暴露稳定技巧，可复用。
3. **两种风格**：functional 和 Module 都提供，灵活。
4. **与 PyTorch 一致**：API 和语义对齐，迁移成本低。
5. **MSE 用组合**：无需手写 backward，autograd 自动，减少 bug 面。
6. **端到端可跑**：训练循环透明，初学者能看清每一步。

### 劣势

1. **功能精简**：无 `reduction`/`ignore_index`/`label_smoothing`/`weight`，生产代码要补。
2. **只有 ReLU**：无 LeakyReLU/ELU/GELU/SiLU 等，现代模型常用。
3. **无 BCE**：二分类要借用 CrossEntropy（二类版），不便。
4. **无 `no_grad`**：eval 时仍建图，浪费内存。
5. **训练循环手写**：工程上重复，无标准封装。
6. **无 AMP**：无混合精度，大模型慢。
7. **`MSELoss` Module 没调 `super().__init__()`**：依赖 `setdefault` 防御，脆弱。

---

## 7.12 下一章预告

本章我们完成了"损失 + 训练循环"，minitorch 的 Python 部分到此就**端到端可跑**了：张量 → 算子 → autograd → 计算图 → Module → 优化器 → 损失 → 训练。

下一章 **第八章 C++ 核心计算层** 将回答：

- Python + numpy 跑深度学习为什么慢？瓶颈在哪？
- 怎么用 C++ 实现核心算子（matmul、conv2d、softmax）以加速？
- pybind11 或 CPython C-API 怎么把 C++ 函数暴露给 Python？
- minitorch 的 dispatch 机制怎么从 Python 函数切换到 C++ kernel？
- 内存管理：C++ 端怎么持有 numpy 数组的内存而不拷贝？
- 怎么编译成 `.so`/`.pyd` 让 Python import？

我们将从 Python 的性能瓶颈分析开始，用 pybind11 桥接 C++，逐步把热点算子迁到 C++，并讨论 dispatch key 机制如何选择 Python 或 C++ 实现。
