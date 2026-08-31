# �?12 章：持久化与混合精度

> 训练一个模型动辄几小时甚至几天，中途断电、OOM、bug 重启是常态——必须能**把模型存下来、再读回去继续训**�?
> 同时，现�?GPU �?fp32 太慢，fp16 能快 2~4 倍但容易数值爆�?下溢�?
> 本章讲两�?工程必修"事：**序列�?*（save/load）和**自动混合精度**（AMP）�?
> 它们看似无关，其实都围绕同一个主题：**让训练既快又稳，且能中断恢复**�?

---

## 12.1 本章目标

读完这一章，你应当能够：

1. 解释 `state_dict` 为什么采�?扁平点号路径"键（�?`"layer1.weight"`），而不是嵌套字典�?
2. 写出 `save`/`load` �?pickle + `__tensor__` 标记序列�?Tensor 的完整流程�?
3. 说清 pickle 序列化与 PyTorch 真实 zip 格式的差异，以及各自的优缺点�?
4. 解释 fp16 的数值范围（±65504，最小正规数 ~6e-8），以及为什么反向传播时梯度容易下溢�?
5. 描述 `GradScaler` 的完整工作流：scale �?backward �?unscale �?check inf �?step �?update�?
6. 推导动�?scale 调整的数学原理（growth_interval / backoff_factor 的作用）�?
7. 解释 `Autocast` 按算子分类策略：matmul/conv �?fp16，reduction 保留 fp32�?
8. 写出一个完整的 AMP 训练循环�?

---

## 12.2 原理铺垫

### 12.2.1 为什么需�?state_dict

一�?`nn.Module` 持有：参数（Parameter）、子模块（Module）、buffer（如 BatchNorm �?running_mean）、hooks、training 标志……其中只�?*参数�?buffer 是模�?学到的知�?**，其他都是结�?状态信息�?

序列化时我们只想�?知识"，不想存结构——结构由代码定义。所�?PyTorch 把所有参数和 buffer 抽出来放进一�?dict，这就是 `state_dict`�?

```python
class Sequential:
    def __init__(self, l1, l2):
        self.l1 = l1   # Linear(4, 3)，有 weight[3,4] bias[3]
        self.l2 = l2   # Linear(3, 2)，有 weight[2,3] bias[2]

model = Sequential(Linear(4,3), Linear(3,2))
model.state_dict()
# {
#   "l1.weight": Tensor[3,4],
#   "l1.bias":   Tensor[3],
#   "l2.weight": Tensor[2,3],
#   "l2.bias":   Tensor[2],
# }
```

**为什么用点号路径而不是嵌�?dict�?*

- 扁平键好序列化：一�?`dict[str, Tensor]` 直接 pickle 就行，不用递归处理嵌套�?
- 扁平键好 diff：两�?state_dict 的键一比就知道结构是否一致�?
- 扁平键好部分加载：想只加�?`l1.*`，过滤键就行；嵌�?dict 要深拷贝再切片�?
- 扁平键好�?checkpoint 分片：大模型按键前缀切到多张卡，`shard_0.l1.weight` 这种�?

代价：键名耦合了模块层级结构，重命名子模块会让�?checkpoint 加载失败。PyTorch �?`strict=False` 缓解�?

### 12.2.2 pickle 的能与不�?

Python 自带�?`pickle` 能序列化几乎所有对象：dict、list、自定义类（只要�?import 到）。但它有几个坑：

1. **不跨版本兼容**：Python 3.8 pickle 的对象在 3.7 读可能挂�?
2. **安全风险**：`pickle.load` 会执行任意代码，**永远不要 load 不信任的文件**�?
3. **�?Tensor �?*：pickle �?ndarray 当字节流 dump，没有压缩、没有分块、没有随机访问�?

教学版用 pickle 是为了简单——核心逻辑就是"递归�?Tensor 转成 dict 标记，再 pickle"�?

真实 PyTorch �?**zip 格式**：每�?Tensor 存成 zip 里的一个独立文件（`.npy` 或自定义二进制），外加一�?`data.pkl` 存元数据。好处：

- 可以**懒加�?*：只读需要的 Tensor，不必把整个文件读进内存�?
- 可以**分块存储**：超�?Tensor 分多�?shard�?
- 可以**版本兼容**：zip 里带 `version` 元数据，新版�?PyTorch 能识别旧格式�?

### 12.2.3 `__tensor__` 标记技�?

我们想把 `{"l1.weight": Tensor(...)}` 存进 pickle。但 pickle 不认识我们的 `Tensor` 类（其实认识，但跨版�?跨实现会出问题）。所以序列化时把每个 Tensor 转成一�?*带标记的普�?dict**�?

```python
Tensor([1, 2, 3])  �? {"__tensor__": True, "data": ndarray, "requires_grad": False}
```

反序列化时检�?dict 里有没有 `__tensor__` 键，有就重建 Tensor，没有就当普�?dict 递归处理�?

这个技巧叫** tagged union**，是序列化里非常通用的模式：用一个小标记字段区分"这是个普�?dict"还是"这是个被伪装的对�?。PyTorch 真实代码里用 `__torch_save__`、`__rebuild__` 等类似标记�?

### 12.2.4 fp16 的数值范围：又窄又稀�?

IEEE 754 半精度（fp16）：
- 1 位符�?+ 5 位指�?+ 10 位尾数�?
- 最大�?�?65504，超过就 inf�?*上溢**）�?
- 最小正规数 �?6.1e-5，小于就变成 subnormal 甚至 0�?*下溢**）�?
- 精度：相邻可表示数之间的相对间隔�?2^-10 �?0.1%�?

fp32�?
- 1 + 8 + 23 位�?
- 最�?�?3.4e38，最小正规数 �?1.2e-38�?
- 相对间隔�?2^-23 �?1e-7�?

对比�?

| �?            | fp16           | fp32           |
| -------------- | -------------- | -------------- |
| 最大�?         | 6.5e4          | 3.4e38         |
| 最小正规数      | 6.1e-5         | 1.2e-38        |
| 精度（相对）    | ~0.1%          | ~1e-7          |

训练里的问题�?
- **前向**：激活值通常�?0~100 量级，fp16 够用。但某些 loss（如 log-likelihood）可能产生极小值，下溢�?0�?
- **反向**：梯度比前向值小几个数量级。比�?loss=1，链式法则乘一�?<1 的偏导，梯度可能 1e-7。fp16 直接下溢�?0，参数不更新—�?*模型假装在训练，其实没学**�?

### 12.2.5 GradScaler 的核心思想：放�?loss

既然梯度太小会下溢，那就**�?loss 乘一个大常数 S**再反向。链式法则下，所有梯度都乘了 S，从 1e-7 变成 1e-7 * 65536 �?6e-3，安全落�?fp16 表示范围�?

反向后、优化器 step 前，**把梯度再除回 S**，恢复真实值。这就是 `scale �?backward �?unscale �?step`�?

但放大后可能**上溢**�?inf。所�?unscale 时检查梯度里有没�?inf/NaN，有�?*跳过这一�?*（不更新参数），并把 S 缩小一半，下次试试更保守的放大倍数�?

如果连续很多步都没溢出，说明 S 太保守了，可以慢慢增大，让梯度更精确（S 越大，越能保留小梯度信息）�?

这就�?**动�?scale 调整**：溢�?�?缩小；长期不溢出 �?增大�?

### 12.2.6 动态调整的数学

设当�?scale = S�?

- **每一�?*：unscale 后检�?inf�?
- **�?inf**：S �?S × backoff_factor（默�?0.5），重置连续成功计数�?`_growth_tracker = 0`�?
- **若无 inf**：`_growth_tracker += 1`；若达到 `growth_interval`（默�?2000），S �?S × growth_factor（默�?2.0），重置计数器�?

为什么默�?interval=2000？经验值：训练初期容易溢出，稳定后想精确表示小梯度�?000 步无溢出说明 S 安全，可以试探性翻倍。翻倍后若立刻溢出，立刻 backoff，不会反复横跳太多�?

数学上，scale 在对数空间做"试探性随机游�?+ 反向回拉"，长期会收敛到一个使溢出概率�?1/growth_interval 的稳态�?

### 12.2.7 Autocast：哪些算子转 fp16，哪些保�?fp32

不是所有算子都适合 fp16�?

| 算子类型      | 例子                | 推荐 dtype | 理由                              |
| ------------ | ------------------- | ---------- | --------------------------------- |
| 矩阵�?卷积   | matmul, conv2d      | **fp16**   | 计算密集，fp16 �?2~4x；中间值范围可�?|
| �?reduction | sum, mean, softmax  | **fp32**   | 累加很多数，fp16 精度不够会误差累�?  |
| 跨数量级      | exp, log, division  | **fp32**   | 输入小变化输出大变化，fp16 精度不够   |
| 激�?         | relu, gelu          | 跟输�?    | 不改变量级，保持�?dtype 即可        |

`Autocast` 是一个上下文管理器，进入时设全局开关，离开时恢复。算子内部检查开关，决定是否把输�?cast �?fp16�?

教学版简化：只提供一个全局开�?+ 一�?`autocast_tensor` 辅助函数，由算子自己决定调不调用。真�?PyTorch �?C++ 层按 op 名查表自�?cast�?

### 12.2.8 AMP 训练循环全貌

把上面拼起来，一�?AMP 训练 step 长这样：

```python
optimizer.zero_grad()
with Autocast():
    pred = model(x)
    loss = loss_fn(pred, y)
scaled_loss = scaler.scale(loss)           # 1. 放大 loss
scaled_loss.backward()                     # 2. 反向（梯度也被放大）
scaler.unscale_(optimizer)                 # 3. 梯度除回 S，检�?inf
scaler.step(optimizer)                     # 4. �?inf �?step
scaler.update()                            # 5. 动态调�?S
```

注意 `with Autocast()` 只包**前向**。反向在 fp32 下做（梯度已经是放大后的 fp32 值，存进 .grad）。这是关键：**前向�?fp16 省时间，反向�?fp32 保精�?*�?

### 12.2.9 数值下溢的直观演示

光说"梯度下溢"可能不直观，我们用数字看�?

假设一�?4 层网络，每层权重 ~0.5，激�?~0.5。前向：

```
x0 = 1.0
x1 = 0.5 * x0 = 0.5
x2 = 0.5 * x1 = 0.25
x3 = 0.5 * x2 = 0.125
x4 = 0.5 * x3 = 0.0625   �?前向还安�?
```

反向时链式法则乘偏导（~0.5），梯度�?loss=1 往回传�?

```
grad_x4 = 1.0
grad_x3 = 0.5 * 1.0 = 0.5
grad_x2 = 0.5 * 0.5 = 0.25
grad_x1 = 0.5 * 0.25 = 0.125
grad_x0 = 0.5 * 0.125 = 0.0625
```

4 层还好。但真实网络 50 层、权�?0.1�?

```
grad_x0 = 0.1^50 �?1e-50   �?fp32 最小正规数 1.2e-38，下溢成 0�?
```

fp16 更惨，最小正规数 6e-5，`0.5^20 �?1e-6` 就下溢了。这就是为什么深层网�?+ fp16 必须�?GradScaler�?

放大 loss 65536 倍后，所有梯度乘 65536�?

```
grad_x0 = 1e-50 * 65536 �?6.5e-46   �?还是下溢
```

这说�?65536 不够。实�?GradScaler 会动态增长到 2^32 甚至更大，直到梯度脱离下溢区�?*这就是动态调整的必要�?*——静�?scale 没法适应不同深度、不同阶段的网络�?

### 12.2.10 序列化的"什么不该存"

state_dict 只存参数�?buffer�?*不存**�?

- 模型结构（层数、隐藏维度）——由代码定义�?
- 优化器状态（动量、Adam �?m/v）——单独存 `optimizer.state_dict()`�?
- 训练进度（epoch、step）——用户自己存�?
- 随机数状态——`torch.get_rng_state()` 单独存�?
- hooks、training 标志——运行时状态�?

这个切分哲学是：**代码定义结构，state_dict 定义"知识"，其他状态各存各�?*。好处是 checkpoint 跨架构——同一�?state_dict 能灌进任何结构匹配的模型。代价是恢复训练要手动恢�?optimizer state、rng state 等�?

完整的训练中断恢复其实要存：

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "rng_state": np.random.get_state(),
}
save(checkpoint, "ckpt.pkl")
```

教学版只实现�?model state_dict 的存取，但这个完整模式是生产训练的标配�?

---

## 12.3 设计决策与权�?

| 决策                          | 我们的选择                              | 理由                                            | 代价                                       |
| --------------------------- | ---------------------------------- | --------------------------------------------- | ---------------------------------------- |
| state_dict 键格�?             | 扁平点号路径                             | 序列化简单、好 diff、好分片                              | 重命名子模块会让�?checkpoint 失效                |
| 序列化后�?                      | pickle                             | 标准库自带，零依�?                                    | 无压缩、无懒加载、跨版本兼容�?                        |
| Tensor 标记                   | `{"__tensor__": True, ...}`        | tagged union 模式，通用且自解释                        | 键名 `__tensor__` 占用，用�?state_dict 不能用这�? |
| 保存内容                        | data ndarray + requires_grad       | 足够覆盖参数；教学版不存 stride/offset                    | 非连�?Tensor 会被 materialize 成连�?          |
| load_state_dict 严格�?         | 静默跳过缺失�?                           | 容易做迁移学习（部分加载�?                                | 真实 PyTorch 默认 strict=True 报错；教学版太宽�?    |
| AMP dtype                   | numpy float16                      | 教学�?numpy 模拟，无需 GPU                          | 真实 fp16 �?GPU 上；CPU fp16 极慢�?numpy 支持�? |
| Autocast 范围                 | 全局开�?+ 算子自查                        | 实现极简                                          | 真实版按 op 名分类清单；教学版没强制 reduction �?fp32   |
| GradScaler 默认 init_scale    | 2^16 = 65536                       | PyTorch 默认值，经验上对大多数模型安�?                       | 某些模型需要更大或更小                              |
| growth_interval             | 2000                               | PyTorch 默认，平衡试探频率和稳定�?                         | 训练步数 < 2000 时永远不会增�?                    |
| backoff_factor              | 0.5                                | 减半是保守选择，避免连续溢�?                               | 减得太慢可能多步溢出                              |
| unscale_ 检查时�?              | step 前显式调�?                        | 让用户能拿到 unscale 后的真实梯度做梯度裁�?                   | 用户忘调�?step 会用放大梯度更新（错误）                  |

---

## 12.4 代码逐行实现

### 12.4.1 `serialization.py`：save / load

```python
"""serialization：模型持久化（Ch12）�?

state_dict 递归收集 Parameter + buffer�?
save/load �?pickle 序列化（教学用；真实 PyTorch �?zip 格式 + 版本兼容）�?
对应真实 PyTorch �?serialization.py�?
"""

from __future__ import annotations

import pickle

from .tensor import Tensor


def save(obj, path: str) -> None:
    """序列�?state_dict 或任�?pickle 可序列化对象到文件�?""
    # 先把 obj 里的 Tensor 都转成带 __tensor__ 标记的普�?dict
    # 这一步是"预处�?，让 pickle 只看到原生类型（dict/list/ndarray/数字�?
    serializable = _to_serializable(obj)
    with open(path, "wb") as f:
        pickle.dump(serializable, f)


def load(path: str):
    """从文件反序列化�?""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # 反过来：把带 __tensor__ 标记�?dict 重建�?Tensor
    return _from_serializable(obj)


def _to_serializable(obj):
    # 递归处理 dict：每个值再递归一次（值可能是 Tensor 或嵌�?dict�?
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    # 碰到 Tensor：转成标�?dict
    # �?.numpy() 拿到 ndarray（会拷贝一次，保证数据独立�?
    # requires_grad 也要存，否则加载后参数默认不求导
    if isinstance(obj, Tensor):
        return {"__tensor__": True, "data": obj.numpy(), "requires_grad": obj.requires_grad}
    # 其他类型（int/str/list/ndarray）pickle 自己能处理，原样返回
    return obj


def _from_serializable(obj):
    if isinstance(obj, dict):
        # 检查是不是 Tensor 标记
        if obj.get("__tensor__"):
            t = Tensor.from_numpy(obj["data"])
            t.requires_grad = obj["requires_grad"]
            return t
        # 普�?dict：递归每个�?
        return {k: _from_serializable(v) for k, v in obj.items()}
    # �?dict 原样返回
    return obj


def save_state_dict(model, path: str) -> None:
    # 便利函数：直接存模型�?state_dict
    save(model.state_dict(), path)


def load_state_dict(model, path: str) -> None:
    # 便利函数：读�?state_dict 灌进模型
    state = load(path)
    model.load_state_dict(state)
```

**逐行要点�?*

- `save` �?`load` 是对称的：`save` 多了一�?`_to_serializable`，`load` 多了一�?`_from_serializable`。这两步互为逆操作�?
- `_to_serializable` �?dict **递归**，所以嵌�?state_dict（虽然我们用扁平键，但用户可能存别的嵌套结构）也能处理�?
- `obj.get("__tensor__")` �?`.get` 而不�?`[]`，因为普�?dict 没这个键，`[]` �?KeyError。`.get` 返回 None（falsy），触发 else 分支�?
- `Tensor.from_numpy` �?classmethod，从 ndarray 重建 Tensor。这�?Tensor 类提供的标准入口�?
- `save_state_dict` / `load_state_dict` 是便利包装，�?�?state_dict"�?序列�?两步合一�?

### 12.4.2 `amp/autocast.py`：自动转换上下文

```python
"""autocast：自动混合精度上下文（Ch12）�?

上下文内前向自动�?fp16。matmul/conv �?fp16，reduction 保留 fp32�?
教学版用 numpy float16 模拟，主要讲原理�?
对应真实 PyTorch �?amp/autocast_mode.py�?
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor


class Autocast:
    def __init__(self, enabled: bool = True, dtype=np.float16):
        self.enabled = enabled
        self.dtype = dtype
        # _prev_enabled 保存进入前的全局状态，离开时恢�?
        # 这是为了支持嵌套 autocast：内层退出时不应该关掉外�?
        self._prev_enabled = False

    def __enter__(self):
        # 记入前先存旧�?
        self._prev_enabled = _autocast_enabled.global_enabled
        if self.enabled:
            _autocast_enabled.global_enabled = True
            _autocast_enabled.global_dtype = self.dtype
        return self

    def __exit__(self, *args):
        # 离开时恢复旧值（而不是简单设 False�?
        _autocast_enabled.global_enabled = self._prev_enabled


# 用一个类的属性当"全局变量"，比�?global 好测试、好重置
class _AutocastState:
    global_enabled: bool = False
    global_dtype = np.float16


_autocast_enabled = _AutocastState()


def is_autocast_enabled() -> bool:
    return _autocast_enabled.global_enabled


def get_autocast_dtype():
    return _autocast_enabled.global_dtype


def autocast_tensor(t: Tensor) -> Tensor:
    """如果 autocast 开启，�?tensor 转为 autocast dtype�?""
    if not is_autocast_enabled():
        return t        # 没开就原样返回，零成�?
    # astype 转换 dtype；_numpy_view 拿到底层 ndarray 视图
    arr = t._numpy_view().astype(get_autocast_dtype())
    return Tensor.from_numpy(arr)
```

**逐行要点�?*

- 用一�?`_AutocastState` 类的属性当全局状态，而不�?`global` 关键字。好处：好测试（可以 `AutocastState.global_enabled = False` 重置）、好命名空间（不会污染模块其他变量）�?
- `__enter__` 存旧值、`__exit__` 恢复旧值，这是上下文管理器处理**嵌套**的标准模式。如�?`__exit__` 直接�?False，内�?with 退出会把外层也关掉�?
- `autocast_tensor` 是给算子调用的辅助函数：算子内部 `if is_autocast_enabled(): x = autocast_tensor(x)`。教学版没强制每个算子都调，由用�?算子自觉�?
- `_numpy_view()` 拿视图不拷贝，`astype` 才真拷贝。这样未开 autocast 时零开销�?

### 12.4.3 `amp/grad_scaler.py`：梯度缩放器

```python
"""GradScaler：梯度缩放器（Ch12）�?

放大 loss 防小梯度下溢。反向后检�?inf/NaN 决定是否 skip step�?
动态调�?scale。对应真�?PyTorch �?amp/grad_scaler.py�?
"""

from __future__ import annotations

import numpy as np


class GradScaler:
    def __init__(
        self,
        init_scale: float = 2.0**16,        # 65536，PyTorch 默认
        growth_factor: float = 2.0,          # 长期无溢出时翻�?
        backoff_factor: float = 0.5,         # 溢出时减�?
        growth_interval: int = 2000,         # 连续 2000 步无溢出才增�?
    ):
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._found_inf = False              # 本次 unscale 是否发现 inf
        self._growth_tracker = 0             # 连续无溢出步�?

    def get_scale(self) -> float:
        return self._scale

    def scale(self, loss):
        """放大 loss�?""
        from ..tensor import Tensor

        # 延迟 import 避免循环依赖
        if isinstance(loss, Tensor):
            # Tensor 路径：拿 ndarray 乘标量，再包�?Tensor
            return Tensor.from_numpy(loss._numpy_view() * self._scale)
        # 标量路径：直接乘
        return loss * self._scale

    def unscale_(self, optimizer) -> None:
        """把梯度除�?scale，并检�?inf/NaN�?""
        self._found_inf = False
        # 遍历所有参数的梯度
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad._numpy_view()
                # 检�?inf �?NaN（NaN 可能�?0/0 产生�?
                if np.any(np.isinf(grad)) or np.any(np.isnan(grad)):
                    self._found_inf = True
                # 原地除回 scale：用 [:] 触发 ndarray 的原地赋�?
                # 注意：即�?found_inf=True 也要除，让用户能看到 unscale 后的�?
                p.grad._numpy_view()[:] = grad / self._scale

    def step(self, optimizer) -> None:
        """如果�?inf/NaN，执�?optimizer.step()�?""
        if not self._found_inf:
            optimizer.step()
        # 否则跳过——参数不更新，等下一步用更小�?scale 重试

    def update(self) -> None:
        """动态调�?scale�?""
        if self._found_inf:
            # 溢出：缩�?scale，重置计数器
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            # 未溢出：累加计数�?
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                # 达到阈值：增大 scale，重置计数器
                self._scale *= self._growth_factor
                self._growth_tracker = 0
```

**逐行要点�?*

- `scale` 方法�?Tensor 和标量分两条路。Tensor 路径要保证返回的还是 Tensor，否则后�?`.backward()` 没有�?
- `unscale_` 名字带下划线，表�?*原地**操作（修改梯度本身）。PyTorch 还有非原地的 `unscale`，但教学版只提供原地版�?
- `p.grad._numpy_view()[:] = ...` 这个切片赋值是 ndarray 原地修改的关键。直�?`p.grad = ...` 会替�?Tensor 引用，optimizer 持有的旧引用就失效了�?
- 即使 `found_inf=True` 也执行除法：让用户能 `print(p.grad)` 看到 unscale 后的值（虽然这些值可能含 inf）。PyTorch 真实版也是这么做的�?
- `step` 只在�?inf 时调 `optimizer.step()`�?*注意 scaler 没调 `zero_grad`**——那是用户的责任，通常在循环开头调�?
- `update` 的两个分支互斥：要么 backoff 要么增长。`_growth_tracker` 在两种情况下都重置为 0�?

---

## 12.5 完整示例

### 12.5.1 序列化往�?

```python
import numpy as np
from minitorch import Tensor
from minitorch.nn import Linear, Sequential
from minitorch.serialization import save, load, save_state_dict, load_state_dict

# 建一个两�?MLP
model = Sequential(Linear(4, 3), Linear(3, 2))
sd = model.state_dict()
print("state_dict keys:", list(sd.keys()))
# state_dict keys: ['0.weight', '0.bias', '1.weight', '1.bias']

# �?
save(sd, "_demo.pkl")

# �?
loaded = load("_demo.pkl")
for k in sd:
    assert np.allclose(sd[k].numpy(), loaded[k].numpy())
print("roundtrip OK")

# 直接存模�?state_dict
save_state_dict(model, "_demo_model.pkl")

# 建一个新模型，加�?
model2 = Sequential(Linear(4, 3), Linear(3, 2))
load_state_dict(model2, "_demo_model.pkl")
for k in model.state_dict():
    assert np.allclose(model.state_dict()[k].numpy(),
                       model2.state_dict()[k].numpy())
print("model load OK")

# 单独存一�?Tensor
t = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
save(t, "_demo_tensor.pkl")
t2 = load("_demo_tensor.pkl")
print("tensor:", t2.numpy())
```

### 12.5.2 AMP 训练循环

```python
import numpy as np
from minitorch import Tensor
from minitorch.nn import Linear, Sequential
from minitorch.optim import SGD
from minitorch.amp import Autocast, GradScaler

np.random.seed(0)
model = Sequential(Linear(4, 8), Linear(8, 1))
opt = SGD(model.parameters(), lr=0.01)
scaler = GradScaler(init_scale=1024.0, growth_interval=10)

# 假数�?
X = np.random.randn(16, 4)
Y = (X.sum(axis=1, keepdims=True) > 0).astype(np.float64)

def loss_fn(pred, y):
    return ((pred - y) ** 2).mean()

for step in range(20):
    # 一�?batch（这里全量）
    x = Tensor.from_numpy(X)
    y = Tensor.from_numpy(Y)

    opt.zero_grad()
    with Autocast(enabled=True):
        pred = model(x)
        loss = loss_fn(pred, y)

    scaled = scaler.scale(loss)
    scaled.backward()
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()

    print(f"step {step:2d}  loss={loss.item():.4f}  "
          f"scale={scaler.get_scale():.1f}  "
          f"found_inf={scaler._found_inf}")
```

预期输出（节选）�?

```
state_dict keys: ['0.weight', '0.bias', '1.weight', '1.bias']
roundtrip OK
model load OK
tensor: [1. 2. 3.]
step  0  loss=0.5234  scale=1024.0  found_inf=False
step  1  loss=0.4981  scale=1024.0  found_inf=False
...
step 10  loss=0.2134  scale=2048.0  found_inf=False   # growth_interval=10 触发增长
```

### 12.5.3 模拟溢出

```python
from minitorch import Tensor
from minitorch.optim import SGD
from minitorch.amp import GradScaler

p = Tensor.from_numpy(np.array([1.0]))
p.requires_grad = True
# 故意把梯度设�?inf，模拟反向时溢出
p.grad = Tensor.from_numpy(np.array([float("inf")]))
opt = SGD([p], lr=0.1)

scaler = GradScaler(init_scale=128.0)
scaler.unscale_(opt)
print("found_inf:", scaler._found_inf)        # True
original = p.numpy().copy()
scaler.step(opt)
print("param unchanged:", np.allclose(p.numpy(), original))   # True，跳过了 step
scaler.update()
print("scale backoff:", scaler.get_scale())   # 64.0
```

### 12.5.4 完整 checkpoint 保存与恢�?

生产训练里中断恢复要存的不只是模型。演示完整模式：

```python
import numpy as np
from minitorch import Tensor
from minitorch.nn import Linear, Sequential
from minitorch.optim import SGD
from minitorch.serialization import save, load

np.random.seed(0)
model = Sequential(Linear(4, 8), Linear(8, 1))
opt = SGD(model.parameters(), lr=0.01)

# 假装训了 3 �?epoch
for epoch in range(3):
    x = Tensor.from_numpy(np.random.randn(16, 4))
    y = Tensor.from_numpy(np.random.randn(16, 1))
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

# 存完�?checkpoint
checkpoint = {
    "epoch": 3,
    "model_state": model.state_dict(),
    "rng_state": np.random.get_state(),
}
save(checkpoint, "_full_ckpt.pkl")
print("saved at epoch 3")

# 恢复
ckpt = load("_full_ckpt.pkl")
np.random.set_state(ckpt["rng_state"])       # 恢复随机状�?
# 假装新建模型（模拟重启）
model2 = Sequential(Linear(4, 8), Linear(8, 1))
model2.load_state_dict(ckpt["model_state"])
start_epoch = ckpt["epoch"]
print(f"resumed from epoch {start_epoch}")

# 验证参数一�?
sd1, sd2 = model.state_dict(), model2.state_dict()
for k in sd1:
    assert np.allclose(sd1[k].numpy(), sd2[k].numpy())
print("resume OK")
```

这个模式是生产训练的标配：模型、优化器、epoch、rng 状态全存一�?dict。重启时全恢复，训练完全无缝继续。教学版 serialization 直接支持这种 dict 存取（值里�?Tensor 会自动用 `__tensor__` 标记）�?

---

## 12.6 常见陷阱

### 陷阱 1：忘�?`zero_grad`

`scaler.step()` 不会清梯度。如果循环开头不 `opt.zero_grad()`，梯度会累加，训练崩�?

**解决**：永远在循环开�?`opt.zero_grad()`�?scaler 不管这事�?

### 陷阱 2：`with Autocast()` 包了反向

```python
with Autocast():
    loss = model(x)
    loss.backward()      # 错！反向不该�?autocast �?
```

反向应该�?fp32 下做。autocast 包反向会让梯度也�?fp16，下溢更严重�?

**解决**：`with Autocast()` 只包前向 + loss 计算，`backward()` �?with 外面�?

### 陷阱 3：用�?scaler 但没�?`update`

`scaler.step()` 不会更新 scale。忘�?`update()`，scale 永远不变，溢出后永远用同一个会溢出�?scale�?

**解决**：每步结尾必�?`scaler.update()`�?

### 陷阱 4：load 不信任的 pickle

`pickle.load` 会执行任意代码。下载别人发�?`.pkl` 直接 load 等于运行未知代码�?

**解决**：只 load 自己存的；从外部拿的模型�?safetensors �?ONNX 等安全格式�?

### 陷阱 5：state_dict 键对不上

模型改了结构（重命名子模块、加了一层），旧 checkpoint 键对不上，`load_state_dict` 静默跳过（教学版）或报错（真�?PyTorch strict=True）�?

**解决**：迁移学习时�?`strict=False` 并手动检查缺失键；改结构时写一个键映射函数�?

### 陷阱 6：fp16 �?CPU 上反而更�?

numpy �?fp16 操作�?CPU 上没有向量化加速，�?fp64 慢。教学版�?numpy fp16 只是演示原理�?*真实训练必须�?GPU �?*�?

### 陷阱 7：`unscale_` 后又 `backward`

`unscale_` 把梯度除�?S。如果之后又�?`backward`（比如做二次反向），新梯度没�?S，混在一起就错了�?

**解决**：`unscale_` �?`step` 前调一次，之后别再 `backward`。需要二次反向的场景�?PyTorch 文档专门处理�?

### 陷阱 8：存了模型但没存优化器状�?

恢复训练时只 `load_state_dict(model)`，优化器从零开始。Adam 的动�?m/v 丢了，等于重�?warmup，loss 突然跳�?

**解决**：checkpoint 连优化器一起存�?

```python
save({"model": model.state_dict(),
      "optim": optimizer.state_dict()}, "ckpt.pkl")
```

### 陷阱 9：fp16 �?`1e-3` 不是你以为的 `1e-3`

```python
np.float16(1e-3)   # = 0.0009990...，相对误�?~0.1%
```

fp16 精度只有 11 位有效位�? 符号 + 10 尾数），`1e-3` 这种数无法精确表示。累加很多次误差会放大�?

**解决**：对精度敏感的标量（learning rate、loss 值）�?fp32 存，�?cast �?fp16�?

### 陷阱 10：Autocast 嵌套�?dtype 丢失

```python
with Autocast(dtype=np.float16):
    with Autocast(dtype=np.bfloat16):   # 内层�?bf16
        ...                              # 内层�?bf16
    # �?这里外层应该恢复 fp16，但教学�?__exit__ 只恢�?enabled，没恢复 dtype
```

教学�?`__exit__` 只恢�?`global_enabled`，没恢复 `global_dtype`。嵌套不�?dtype 会出 bug�?

**解决**：`__enter__` 也存 `_prev_dtype`，`__exit__` 恢复。这是教学版的一个已知简化缺陷，真实 PyTorch 处理正确�?

### 陷阱 11：`scaler.step` 后忘�?`optimizer.zero_grad`

```python
for ...:
    with Autocast():
        loss = model(x)
    scaled = scaler.scale(loss)
    scaled.backward()
    scaler.unscale_(opt)
    scaler.step(opt)       # �?更新了参�?
    scaler.update()
    # �?忘了 zero_grad，下一步梯度累�?
```

scaler 不管 zero_grad。下一�?`backward` 会把新梯度加到旧梯度上，训练崩�?

**解决**：循环开头永�?`opt.zero_grad()`，不是循环结尾�?

### 陷阱 12：序列化�?Python 版本�?

Python 3.10 存的 pickle �?3.9 读可�?`AttributeError` �?`ModuleNotFoundError`，因�?pickle 引用�?3.10 才有的类型�?

**解决**：跨版本场景�?`safetensors`（只存张量数据，无代码引用）�?ONNX。教学版 pickle 只适合同版本往返�?

---

## 12.7 与真�?PyTorch 对照

| minitorch                              | torch                                     | 关键差异                                                     |
| -------------------------------------- | ----------------------------------------- | -------------------------------------------------------- |
| `save` / `load` (pickle)                | `torch.save` / `torch.load` (zip 格式)      | 真实版用 zip 容器，每�?Tensor 独立文件，支持懒加载、分片、版本元数据            |
| `_to_serializable` (`__tensor__` 标记)   | `_rebuild_tensor_v2` �?pickle reduce 协议   | 真实版用 pickle �?`__reduce__` 机制，更紧凑但更难调�?               |
| `save_state_dict` / `load_state_dict`   | 同名便利函数                                  | 一�?                                                      |
| `state_dict` 扁平�?                      | �?                                        | 一�?                                                      |
| `load_state_dict` 静默跳过缺失�?             | `strict=True` 默认报错                       | 教学版太宽松；真实版严格，迁移学习时手动 `strict=False`                  |
| `Autocast` 全局开�?                       | `torch.amp.autocast(device_type=...)`     | 真实版按 device 分开状态，支持 CPU/GPU 不同策略；按 op 名查表自�?cast      |
| `autocast_tensor` 辅助函数                 | 算子内部 C++ 自动 cast                        | 教学版要算子自觉调；真实版在 C++ 层拦�?                               |
| `GradScaler` API                       | `torch.amp.GradScaler`                   | 一致；真实版还�?`unscale` (非原�?、`get_scale_sync` 等多 GPU 同步方法 |
| `init_scale=2^16`                       | �?                                        | 一�?                                                      |
| `growth_interval=2000`                  | �?                                        | 一�?                                                      |
| numpy fp16 模拟                          | 真实 GPU fp16/bf16                         | 教学版无 GPU 加速；真实�?bf16 是更友好的范围（指数 8 位像 fp32�?           |
| �?                                     | `torch.amp.bf16` / `torch.compile(amp=...)` | 真实�?bf16 不需�?GradScaler（范围够大），compile 自动 AMP         |

!!! tip "bf16 是什�?"
bf16（bfloat16）：1 + 8 + 7 位。指数位�?fp32 一样多，所�?*范围�?fp32 一�?*（�?.4e38），只是精度差（~1%）。这意味着 bf16 **几乎不会溢出**，也不需�?GradScaler。Google TPU 最早用，现�?NVIDIA A100/H100 也支持。是 fp16 的有力替代�?

---

## 12.8 历史背景

**序列化：**

- **PyTorch 0.x**：直接用 pickle，跨版本兼容差，大模型慢�?
- **1.0**：引�?zip 容器格式（`torch.save` �?zip），每个 Tensor 独立文件，支持懒加载�?
- **1.6**：`torch.save` 默认用新 zip 格式，旧 pickle 格式仍可读但不再写�?
- **1.13+**：引�?`safetensors` 支持作为安全替代；`torch.load` �?`weights_only=True` 选项避免执行任意代码�?
- **2.0+**：`torch.save` 进一步优化分片，支持 TB 级模型分卡存储�?

**混合精度�?*

- **2017 �?*：用户手�?`.half()` 模型和输入，反向下溢没人管，AMP 概念不存在�?
- **NVIDIA 2017**：提�?`apex` 库的 `FP16_Optimizer`，第一次系统化 GradScaler 思想�?
- **PyTorch 1.6�?020�?*：`torch.cuda.amp` 正式合入主干，API 稳定�?
- **1.10**：CPU autocast 支持（用�?bf16 推理）�?
- **2.0**：`torch.amp` 命名空间统一 CPU/CUDA；`torch.compile` 内置 AMP 优化，不再需要显�?`Autocast` 包裹�?
- **未来**：bf16 普及�?GradScaler 可能逐渐退出历史舞台�?

minitorch 这套实现对应 PyTorch 1.6 的经�?AMP API，用 numpy fp16 模拟 GPU fp16，把核心数学讲清楚�?

### 12.8.1 NVIDIA apex 的遗�?

PyTorch 1.6 之前，AMP 主要�?NVIDIA 的第三方�?`apex`。它�?`apex.ampp.initialize` API 长这样：

```python
from apex import amp
model, opt = amp.initialize(model, opt, opt_level="O1")
with amp.scale_loss(loss, opt) as scaled_loss:
    scaled_loss.backward()
```

`opt_level` �?`O0/O1/O2/O3` 四档，从�?fp32 到纯 fp16，用户要自己选。这�?API 强大但复杂，�?NVIDIA 维护、不�?PyTorch 主干�?

1.6 �?`O1`（混合精度）模式合入主干�?`torch.cuda.amp`，API 简化成 `Autocast` + `GradScaler`。apex 逐渐弃用。这个演化说明：**好的 API 是从实践中长出来的，不是一开始设计出来的**。apex 的四档暴露了太多实现细节，PyTorch 的两件套（Autocast + GradScaler）只暴露必要复杂度�?

---

## 12.9 练习�?

### 练习 1：实�?`strict` 参数

修改 `load_state_dict` 接受 `strict=True` 参数。`strict=True` 时若 state_dict 有模型不认识的键，或模型�?state_dict 没提供的键，报错。`strict=False` 时静默跳过�?

??? 解答
    ```python
    def load_state_dict(model, state_dict, strict=True, prefix=""):
        model_keys = set(model.state_dict().keys())
        sd_keys = set(state_dict.keys())
        if strict:
            missing = model_keys - sd_keys
            unexpected = sd_keys - model_keys
            if missing:
                raise KeyError(f"Missing keys: {missing}")
            if unexpected:
                raise KeyError(f"Unexpected keys: {unexpected}")
        # 然后按原逻辑加载，跳过缺失键
        for name, p in model.named_parameters():
            if name in state_dict:
                src = state_dict[name]
                p._storage._data[:] = src.numpy().ravel()
    ```
???

### 练习 2：解释为什�?`unscale_` 即使 `found_inf=True` 也要�?

??? 解答
    两个理由�?
    1. **用户检�?*：用户可能想 `print(p.grad)` �?unscale 后的梯度诊断问题。如果不除，梯度还是放大 65536 倍的值，看不出真实量级�?
    2. **统一语义**：`unscale_` 的契约是"调用后梯度是真实�?。如果有时除有时不除，用户要写两套逻辑处理。统一除（即使�?inf）让契约简单�?
    inf 除以有限数还�?inf，所以除法本身不会改�?是否 inf"的判断�?
???

### 练习 3：推导稳�?scale

假设每步溢出概率独立�?p。求稳态时 scale 的期望对数值（提示：列平衡方程）�?

??? 解答
    设对�?scale �?x = log2(S)。每步：
    - 概率 p：x �?x - 1（backoff 0.5 = 2^-1�?
    - 概率 (1-p)：x �?x + 1/growth_interval（每 2000 �?+1，平均每�?+1/2000�?
    稳态期望变化为 0�?
        -p + (1-p) / 2000 = 0
        p = 1 / 2001
    即稳态时溢出概率�?1/2001 �?0.05%。这正是 growth_interval=2000 的设计：把溢出概率控制在约万分之五�?
???

### 练习 4：为什�?matmul �?fp16 �?sum 保留 fp32

用具体数字说明�?

??? 解答
    matmul：C[i,j] = sum_k A[i,k] * B[k,j]。每个乘�?A*B 通常�?0~10 量级，fp16 精度 0.1% 够。中间累加次�?= K（隐藏维度），通常 64~1024，fp16 累加这么多次误差仍可控（GPU �?matmul �?fp32 累加器，更稳）�?
    sum：s = sum_i x[i]，i 可能上百万。若每个 x ~ 1e-3，fp16 表示 1e-3 的误差约 1e-6。累�?1e6 次，误差 ~1，可能比真实 sum 还大。fp32 误差 1e-10，累�?1e6 次误�?1e-4，安全�?
    所以大 reduction 必须�?fp32 累加器�?
???

### 练习 5：写一个不需�?GradScaler �?bf16 训练循环

??? 解答
    ```python
    # bf16 范围�?fp32 一样，不会溢出，不需�?scaler
    for x, y in loader:
        opt.zero_grad()
        with Autocast(dtype=np.bfloat16):    # 假设支持
            pred = model(x)
            loss = loss_fn(pred, y)
        loss.backward()                      # 直接反向，无 scale
        opt.step()                           # 直接 step，无 unscale
    ```
    bf16 的代价是精度差（~1%），但对大多数训练够用。这也是为什么新硬件（A100/H100/TPU）主�?bf16�?
???

---

## 12.10 关键测试解读

`tests/test_serialization.py`�?

```python
def test_save_load_roundtrip():
    model = Sequential(Linear(4, 3), Linear(3, 2))
    sd = model.state_dict()
    save(sd, path)
    loaded = load(path)
    for key in sd:
        assert np.allclose(sd[key].numpy(), loaded[key].numpy())
```

**解读**：往返测试——存再读，每个键的值应�?bit 级一致（allclose 容忍浮点误差）。这是序列化最基本的不变量：`load(save(x)) == x`�?

```python
def test_save_load_state_dict_into_model():
    model1 = Sequential(Linear(4, 3), Linear(3, 2))
    model2 = Sequential(Linear(4, 3), Linear(3, 2))
    save_state_dict(model1, path)
    load_state_dict(model2, path)
    # model2 现在应该�?model1 参数完全一�?
```

**解读**：两个独立初始化的模型，�?model1 灌进 model2，应该完全一致。验�?`load_state_dict` 真的把数据写进了参数存储，而不只是读了 state_dict 放着�?

`tests/test_amp.py`�?

```python
def test_grad_scaler_skips_inf():
    p.grad = Tensor.from_numpy(np.array([float("inf")]))
    opt = SGD([p], lr=0.1)
    scaler = GradScaler(init_scale=128.0)
    scaler.unscale_(opt)
    assert scaler._found_inf            # 检测到 inf
    original = p.numpy().copy()
    scaler.step(opt)
    assert np.allclose(p.numpy(), original)   # 参数没变，step 被跳�?
```

**解读**：核心安全测试——梯�?inf �?step 必须被跳过，参数原封不动。如�?`step` 没检�?`_found_inf`，参数会�?inf 更新�?NaN，训练彻底崩�?

```python
def test_grad_scaler_update_backoff():
    scaler = GradScaler(init_scale=128.0, backoff_factor=0.5)
    scaler._found_inf = True
    scaler.update()
    assert scaler.get_scale() == 64.0
```

**解读**：溢出后 scale 应该减半。验�?`update` �?backoff 分支�?

```python
def test_grad_scaler_update_growth():
    scaler = GradScaler(init_scale=128.0, growth_factor=2.0, growth_interval=3)
    scaler._found_inf = False
    scaler.update()   # tracker=1
    scaler.update()   # tracker=2
    assert scaler.get_scale() == 128.0   # 还没�?3，不增长
    scaler.update()   # tracker=3 �?增长
    assert scaler.get_scale() == 256.0
```

**解读**：连续无溢出达到 interval 才增长，不是每步都长。验�?`_growth_tracker` 计数逻辑�?

```python
def test_autocast_context():
    from minitorch.amp.autocast import is_autocast_enabled
    assert not is_autocast_enabled()
    with Autocast(enabled=True):
        assert is_autocast_enabled()
    assert not is_autocast_enabled()
```

**解读**：上下文管理器的基本契约——进入时开、离开时关。三个断言：进前关、进后开、出后关。如�?`__exit__` 忘了恢复，最后一个断言挂。这个测试看似简单，但能抓到"上下文泄�?这种最难调�?bug�?

```python
def test_autocast_tensor_dtype():
    from minitorch.amp.autocast import autocast_tensor
    t = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    with Autocast(enabled=True):
        t_cast = autocast_tensor(t)
        assert t_cast.dtype == np.float16
    assert t.dtype == np.float64
```

**解读**：两个关键点�?1) autocast 开启时 `autocast_tensor` �?fp64 转成 fp16�?2) **�?tensor 不被修改**——`t.dtype` 仍是 fp64。这验证 cast �?*非原�?*的（返回�?Tensor）。如果原地改了，用户�?autocast 外再�?`t` 会发现变 fp16 了，难以调试�?

```python
def test_grad_scaler_scale():
    scaler = GradScaler(init_scale=128.0)
    loss = Tensor.from_numpy(np.array([1.0, 2.0]))
    scaled = scaler.scale(loss)
    assert np.allclose(scaled.numpy(), [128.0, 256.0])
```

**解读**：`scale` �?loss 乘以 `_scale`。`[1,2] * 128 = [128, 256]`。这个测试验证乘法正确，且返回的是新 Tensor（不修改�?loss）。如�?`scale` 实现成原地改 loss，下游再�?loss 会拿到放大后的值�?

```python
def test_grad_scaler_normal_step():
    p = Tensor.from_numpy(np.array([1.0]))
    p.requires_grad = True
    p.grad = Tensor.from_numpy(np.array([0.5]))
    opt = SGD([p], lr=0.1)
    scaler = GradScaler(init_scale=128.0)
    scaler.unscale_(opt)
    assert not scaler._found_inf
    scaler.step(opt)
    assert np.allclose(p.numpy(), [1.0 - 0.1 * 0.5 / 128.0])
```

**解读**：完整正常路径测试。梯�?0.5，scale 128，unscale 后梯度变 0.5/128。SGD 更新 `p = p - lr * grad = 1.0 - 0.1 * (0.5/128)`。这个测试验证了 **scale �?unscale �?step 的完整数学链**：放大后再除回，最终更新用的是真实梯度。如�?unscale 漏了除法，参数会�?128 倍梯度更新，直接飞�?

```python
def test_save_load_tensor():
    t = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    path = "_test_tensor.pkl"
    save(t, path)
    loaded = load(path)
    assert np.allclose(loaded.numpy(), [1.0, 2.0, 3.0])
```

**解读**：单独存一�?Tensor（不�?state_dict）。验�?`_to_serializable` / `_from_serializable` 对裸 Tensor 的处理——`save` 把它包成 `{"__tensor__": True, ...}`，`load` 识别标记重建。如果标记键拼错或漏检查，这里会返�?dict 而不�?Tensor�?

---

## 12.11 优劣势总结

**优势�?*

- **序列化极简**：pickle + 标记 dict�?0 行代码讲清核心�?
- **state_dict 扁平�?*：好序列化、好 diff、好分片，与真实 PyTorch 一致�?
- **AMP 数学完整**：scale/unscale/check/step/update 五步齐全，动态调整逻辑�?PyTorch 1.6+ 一致�?
- **GradScaler �?optimizer 解�?*：任�?optimizer 都能用，不用�?optimizer 代码�?

**劣势�?*

- **pickle 不安�?*：load 任意文件会执行代码，生产环境必须�?safetensors�?
- **无懒加载**：load 整个文件进内存，大模�?OOM�?
- **无压�?*：fp32 参数存盘 4 字节/元素，没压缩，磁盘占用大�?
- **autocast 不强�?*：算子自觉调 `autocast_tensor`，容易忘�?
- **CPU fp16 �?*：教学版�?CPU 上跑 fp16 �?fp64 慢，无法演示真实加速�?
- **�?bf16**：现代训练主�?bf16，教学版没实现�?

**教学价�?*：把"为什么放�?loss"�?为什么动态调�?scale"这两件最容易被当成黑盒的事讲透了。理解了这套数学，去�?PyTorch `GradScaler` 几百�?C++ 代码就不会迷路——核心就是本章这 60 �?Python�?

---

## 12.12 下一章预�?

到目前为止，模型都是"命令�?执行——一行行 Python 跑。下一章我们进�?*图与编译导论**�?

- 什么是**符号追踪**？怎么�?Proxy 替换真实 Tensor 拦截运算�?
- `Node` �?op 分类（placeholder/call_function/call_method/output）各自代表什么？
- `Graph` 怎么 codegen �?Python 源码�?
- `GraphModule` 怎么解释执行一�?Graph�?
- 算子融合 pass 怎么做模式匹配和图重建？
- FX 为什么处理不�?data-dependent control flow�?
- `torch.compile` (Dynamo + Inductor) �?FX 强在哪？

这是�?用框�?走向"理解编译�?的第一步�?
