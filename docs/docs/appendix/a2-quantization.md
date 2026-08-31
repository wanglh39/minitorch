# A2 量化导论

> 本附录对应主�?Ch6/Ch7。量化是�?fp32 模型压缩�?int8，减少显存、加速推理�?
> 本附录以原理讲解为主，重点讲清量化映射、Observer、FakeQuantize 三个核心概念�?

---

## A2.1 为什么量�?

fp32 模型推理的瓶颈：

| 瓶颈 | fp32 | int8 | 提升 |
|------|------|------|------|
| 显存占用 | 4 bytes/参数 | 1 byte/参数 | 4× 压缩 |
| 内存带宽 | 32 bit/load | 8 bit/load | 4× 带宽 |
| 计算速度 | FP32 FLOPS | INT8 FLOPS | 2-4× 加�?|

int8 �?甜点"：精度损失可控，硬件支持广泛（CPU AVX-VNNI、GPU Tensor Core）�?

### A2.1.1 精度 vs 速度 vs 显存的三方权�?

量化不是"免费午餐"，而是在三个维度上权衡�?

```
        精度
         �?
         �?
    fp32 �?
         �?
    fp16 �?
         �?     �?
    int8 ●──────�? �?工程甜点
         �?     �?
    int4 �?      �?
         �?       �?
     int1 ●────────�?速度/显存 �?
```

- **fp32 �?fp16**：精度几乎无损，2× 加速，硬件原生支持（Tensor Core�?
- **fp32 �?int8**：精度损 0.5-3%�?× 压缩，需量化流程
- **int8 �?int4**：精度损 3-10%�?× 压缩，需 QAT + 特殊硬件
- **int4 �?int1 (BNN)**：精度大幅下降，32× 压缩，仅学术

**工程选择**：大多数推理场景�?int8；极致显存约束（手机 NPU）�?int4；训练用 fp16/bf16 混合精度（见 Ch12）�?

### A2.1.2 量化的适用场景

| 场景 | 量化收益 | 备注 |
|------|---------|------|
| 服务�?CPU 推理 | 大（内存带宽是瓶颈） | FBGEMM 后端 |
| 服务�?GPU 推理 | 中（计算已很快） | CUDA Tensor Core int8 |
| 边缘设备（手�?嵌入式） | 极大（显�?算力都紧�?| TFLite / CoreML |
| 训练 | **不适用** | 量化仅用于推理，训练仍需 fp32/fp16 |
| 检�?分割等敏感任�?| 需 QAT 保精�?| PTQ 可能�?3-5% |

---

## A2.2 量化映射：浮�?�?整数

### A2.2.1 仿射量化（Affine Quantization�?

最常用的量化方案：�?fp32 范围 `[r_min, r_max]` 线性映射到 int8 范围 `[q_min, q_max]`�?

```
量化:   q = round(r / scale + zero_point)
反量�? r = scale * (q - zero_point)

其中:
  scale = (r_max - r_min) / (q_max - q_min)
  zero_point = round(q_min - r_min / scale)

  q_min = -128, q_max = 127  (signed int8)
  r_min, r_max = 观测到的最�?最大浮点�?
```

**图示**�?

```
浮点�? r_min -------- 0 -------- r_max
           �?scale
整数�? q_min --- zero_point --- q_max

scale = (r_max - r_min) / (q_max - q_min)
```

### A2.2.2 对称量化（Symmetric Quantization�?

简化版：令 `zero_point = 0`，浮点范围对�?`[-r_max, r_max]`�?

```
scale = r_max / 127        # signed int8
q = round(r / scale)
r = scale * q
```

**优点**：`zero_point = 0`，计算时省一次减法，推理更快�?
**缺点**：如果浮点范围不对称（如 ReLU 输出全正），浪费一半整数范围�?

### A2.2.3 量化误差

量化�?*有损压缩**，误差来�?round 取整�?

```
r = 0.123, scale = 0.01
q = round(0.123 / 0.01) = round(12.3) = 12
r' = 12 * 0.01 = 0.120    # 误差 0.003
```

误差上界 = `scale / 2`（半步长）。scale 越小（范围越窄），精度越高�?

### A2.2.4 完整数值示�?

用一个具体例子走完整个量化流程，加深直觉�?

**场景**：量化一�?ReLU 的输出张�?`x = [0.1, 1.2, 3.4, 0.5, 2.7]`，用对称 int8 量化�?

```
步骤 1: 观测范围
  r_max = max(|x|) = 3.4
  r_min = -3.4  (对称量化)

步骤 2: 计算 scale
  scale = r_max / 127 = 3.4 / 127 = 0.02677

步骤 3: 量化（fp32 �?int8�?
  q[0] = round(0.1 / 0.02677) = round(3.74) = 4
  q[1] = round(1.2 / 0.02677) = round(44.8) = 45
  q[2] = round(3.4 / 0.02677) = round(127.0) = 127
  q[3] = round(0.5 / 0.02677) = round(18.7) = 19
  q[4] = round(2.7 / 0.02677) = round(100.9) = 101

  q = [4, 45, 127, 19, 101]   �?存储�?5 �?int8�? bytes�?

步骤 4: 反量化（int8 �?fp32，推理时�?
  r'[0] = 4   * 0.02677 = 0.107   (�?0.1, 误差 0.007)
  r'[1] = 45  * 0.02677 = 1.205   (�?1.2, 误差 0.005)
  r'[2] = 127 * 0.02677 = 3.400   (�?3.4, 误差 0.000)
  r'[3] = 19  * 0.02677 = 0.509   (�?0.5, 误差 0.009)
  r'[4] = 101 * 0.02677 = 2.704   (�?2.7, 误差 0.004)

  r' = [0.107, 1.205, 3.400, 0.509, 2.704]
```

**观察**�?
- 最大�?`3.4` 量化误差�?0（刚好对齐到 127�?
- 小�?`0.1` 误差 7%（相对误差大——这是量化的固有缺陷：小值精度差�?
- 误差上界 = `scale/2 = 0.0134`，所有误差都在上界内

### A2.2.5 非对称量化的优势

ReLU 输出全非负，用对称量化浪费一半范围。非对称量化更好�?

```
x = [0.1, 1.2, 3.4, 0.5, 2.7]  (ReLU 输出，全�?

非对称量�?
  r_min = 0.1, r_max = 3.4
  scale = (3.4 - 0.1) / 255 = 0.01294
  zero_point = round(-128 - 0.1 / 0.01294) = round(-135.7) = -136
  �?clamp �?[-128, 127]，zero_point = -128

  q[0] = round(0.1 / 0.01294 + (-128)) = round(-120.3) = -120
  q[2] = round(3.4 / 0.01294 + (-128)) = round(134.7) = 127

  �?用了 [-120, 127] 范围�?47 个级别），比对称�?[4, 127]�?23 级别）精�?2×
```

**结论**：激活值（ReLU 后全正）用非对称量化，权重（有正有负）用对称量化，是工程最佳实践�?

---

## A2.3 两条路线：PTQ vs QAT

### A2.3.1 PTQ（Post-Training Quantization，训练后量化�?

训练�?fp32 模型后，用少量校准数�?观测"各层激活的范围，直接量化�?

```
1. 训练 fp32 模型 �?model_fp32
2. 用校准数据跑前向 �?Observer 记录每层激活的 min/max
3. 用观测到�?min/max 计算 scale �?量化�?int8
4. 推理�?int8 模型
```

**优点**：快，不需要重新训练�?
**缺点**：精度损失可能较大（尤其是低 bit 量化�?int4）�?

### A2.3.2 QAT（Quantization-Aware Training，量化感知训练）

在训练中**模拟量化误差**，让模型适应量化后的精度损失�?

```
1. 在模型中插入 FakeQuantize 算子（前向量化、反向用直通估计器�?
2. 继续训练几个 epoch �?模型学会在量化噪声下工作
3. 导出真正�?int8 模型
```

**优点**：精度损失最小�?
**缺点**：需要训练数据和训练时间�?

### A2.3.3 选择建议

| 场景 | 推荐 | 理由 |
|------|------|------|
| 快速部署、可接受 1-2% 精度�?| PTQ | 省时�?|
| 精度敏感（如检测、分割） | QAT | 保精�?|
| int4 / 二进制等超低 bit | QAT | PTQ 精度崩塌 |
| 没有训练数据 | PTQ | 唯一选择 |

### A2.3.4 PTQ 的精度崩塌现�?

为什�?PTQ 在低 bit 下会崩塌？用一个例子说明：

```
fp32 模型的某一层权重分�?
  大部分权重在 [-0.1, 0.1]（重要）
  少数 outlier �?[-3.0, 3.0]（不重要但拉大范围）

PTQ int8 对称量化:
  scale = 3.0 / 127 = 0.0236
  �?0.1 量化�?round(0.1/0.0236) = 4，反量化 = 0.094（误�?6%�?
  �?大部分重要权重的相对误差 6%，累积后精度崩塌

QAT int8:
  训练�?FakeQuantize 引入同样误差
  �?模型学会�?outlier 权重缩小（正则化效应�?
  �?最终权重分布更集中，scale 更小，精度更�?
```

**QAT 的本�?*：让模型主动调整权重分布，使�?量化友好"——范围集中、outlier 少�?

---

## A2.4 Observer：观测数据范�?

Observer 的职责：在前向传播中收集激活值的统计量（min/max 或直方图），用于计算 scale�?

### A2.4.1 MinMaxObserver

最简单：记录所有观测值的全局 min �?max�?

```python
class MinMaxObserver:
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def __call__(self, x):
        self.min_val = min(self.min_val, x.min().item())
        self.max_val = max(self.max_val, x.max().item())

    def calculate_scale(self):
        scale = (self.max_val - self.min_val) / 255
        zero_point = round(-128 - self.min_val / scale)
        return scale, zero_point
```

**问题**：极�?outlier 会拉大范围，导致大部分正常值量化精度差�?

### A2.4.2 MovingAverageMinMaxObserver

�?EMA 平滑 min/max，避�?outlier 主导�?

```python
class MovingAverageMinMaxObserver:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.min_val = None
        self.max_val = None

    def __call__(self, x):
        cur_min = x.min().item()
        cur_max = x.max().item()
        if self.min_val is None:
            self.min_val, self.max_val = cur_min, cur_max
        else:
            self.min_val = self.alpha * cur_min + (1 - self.alpha) * self.min_val
            self.max_val = self.alpha * cur_max + (1 - self.alpha) * self.max_val
```

### A2.4.3 HistogramObserver

更精细：记录激活值的**直方�?*分布，然后搜索最优裁剪点（去�?outlier 的尾部，使量化误差最小）�?

```python
class HistogramObserver:
    def __init__(self, bins=1024):
        self.histogram = torch.zeros(bins)
        self.bins = bins

    def __call__(self, x):
        hist = torch.histc(x, bins=self.bins, min=x.min(), max=x.max())
        self.histogram += hist

    def calculate_scale(self):
        # 搜索最优裁剪阈值：去掉直方图尾�?x%，使 L2 量化误差最�?
        best_loss = float('inf')
        for threshold_percent in range(0, 20):  # 尝试裁掉 0%-20%
            threshold = percentile(self.histogram, 100 - threshold_percent)
            loss = compute_quant_error(self.histogram, threshold)
            if loss < best_loss:
                best_loss = loss
                best_threshold = threshold
        return best_threshold / 127  # scale
```

### A2.4.4 Observer 对比实验

三种 Observer 在不同数据分布下的表现：

```
数据分布 A: 均匀分布 [-1, 1]
  MinMax:        scale = 1/127 = 0.00787,  误差 = 0.0039
  MovingAverage: scale = 1/127 = 0.00787,  误差 = 0.0039  (�?outlier，三者相�?
  Histogram:     scale = 1/127 = 0.00787,  误差 = 0.0039

数据分布 B: 均匀 [-1, 1] + 1% outlier �?[5, 6]
  MinMax:        scale = 6/127 = 0.0472,   误差 = 0.0236  �?outlier 拉大范围�?
  MovingAverage: scale = 1.5/127 = 0.0118, 误差 = 0.0059  �?EMA 平滑�?outlier
  Histogram:     scale = 1.2/127 = 0.0094, 误差 = 0.0047  �?裁掉 outlier 尾部

数据分布 C: 长尾分布（如 softmax 输出，大部分接近 0，少数接�?1�?
  MinMax:        scale = 1/127,  大部分值量化为 0 �?灾难性精度损�?
  MovingAverage: 类似 MinMax，EMA 对长尾无�?
  Histogram:     裁掉长尾，scale = 0.3/127,  小值精度好 �?推荐
```

**选择建议**�?

| 数据分布 | 推荐 Observer |
|---------|--------------|
| 均匀、无 outlier | MinMax（最简单） |
| 偶尔�?outlier | MovingAverageMinMax |
| 长尾分布（常见于激活） | Histogram |
| 权重（静态，只观测一次） | MinMax（权重不变，�?outlier 问题�?|

### A2.4.5 校准数据的选择

PTQ 的精度高度依赖校准数据的质量�?

```python
# 好的校准数据
calibration_data = load_representative_subset(
    dataset=test_set,
    n_samples=100,          # 100-500 样本通常�?
    strategy="diverse",     # 覆盖各种输入分布
)

# 坏的校准数据
calibration_data = [test_set[0]] * 100  # 同一张图重复 100 �?�?激活范围不全面
```

**校准数据要求**�?
1. **代表�?*：覆盖推理时的真实输入分�?
2. **多样�?*：不同类别、不同难度、不同光�?角度
3. **数量**�?00-500 样本通常足够，再多收益递减
4. **与训练数据同分布**：不要用训练集外的数据校�?

**常见错误**：用全零输入校准 �?所有激活都�?0 �?scale = 0 �?量化崩塌�?

---

## A2.5 FakeQuantize：训练中模拟量化

QAT 的核心：在训练前向时**真的量化再反量化**，引入量化误差；反向时用**直通估计器（STE�?* 让梯度穿过量化算子�?

### A2.5.1 前向：量�?+ 反量�?

```python
class FakeQuantize:
    def __init__(self, observer):
        self.observer = observer
        self.scale = None
        self.zero_point = None

    def forward(self, x):
        # 1. 观测当前激活范�?
        self.observer(x)
        self.scale = self.observer.calculate_scale()
        self.zero_point = self.observer.calculate_zero_point()

        # 2. 量化：fp32 �?int8
        q = torch.clamp(
            torch.round(x / self.scale + self.zero_point),
            -128, 127
        )

        # 3. 反量化：int8 �?fp32（带量化误差�?
        x_fake = (q - self.zero_point) * self.scale
        return x_fake
```

前向输出 `x_fake` �?`x` 形状相同、类型相同（fp32），但值有量化误差。模型在这个误差下训练，学会适应�?

### A2.5.2 反向：直通估计器（Straight-Through Estimator�?

量化函数 `q = round(r)` 不可导（round 的梯度处处为 0）。STE �?trick�?*假装量化函数的梯度为 1**�?

```
∂L/∂x �?∂L/∂x_fake * 1    # 直接把下游梯度传给上�?
```

```python
    def backward(self, grad_output):
        # STE: 梯度直通，�?clip 范围外梯度为 0
        grad_input = grad_output.clone()
        # 超出量化范围的输入梯度为 0（因�?clamp 截断了）
        grad_input[x < self.min_clamp] = 0
        grad_input[x > self.max_clamp] = 0
        return grad_input
```

**为什�?STE 有效**：量化本质是"加一个小噪声"，STE 假设这个噪声对梯度的扰动可忽略。实验证明对 int8 量化有效，对 int4 需要更精细的梯度估计�?

### A2.5.3 STE 的数学推�?

更严谨地理解 STE 为什么有效�?

**量化函数**：`q(r) = round(r / s) * s`（对称量化简化版�?

**真实梯度**：`dq/dr = 0`（round 函数几乎处处导数�?0）→ 反向传播梯度永远�?0，无法训练�?

**STE 近似**：`dq/dr �?1`

**为什么合�?*？把量化看作"加噪�?�?

```
q(r) = r + ε(r)    其中 ε(r) = round(r/s)*s - r 是量化误�?

dq/dr = 1 + dε/dr

|ε(r)| �?s/2  （量化误差上界）
|dε/dr| �?1   （误差变化率有界�?

�?dq/dr �?[0, 2]，STE �?dq/dr = 1 是合理近�?
```

**更精细的替代**�?
- **LSQ（Learned Step Size Quantization�?*：把 scale 也当作可学习参数，用真实梯度更新 scale
- **DSQ（Differentiable Soft Quantization�?*：用 soft round 替代 hard round，梯度可�?

### A2.5.4 FakeQuantize 完整实现（带 autograd�?

```python
import torch
import torch.nn as nn

class FakeQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, q_min=-128, q_max=127):
        # 量化 + 反量�?
        q = torch.clamp(
            torch.round(x / scale + zero_point),
            q_min, q_max
        )
        x_fake = (q - zero_point) * scale

        # 保存反向传播需要的信息
        ctx.save_for_backward(x)
        ctx.scale = scale
        ctx.q_min = q_min
        ctx.q_max = q_max
        return x_fake

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        scale = ctx.scale

        # STE: 梯度直通，超出范围的梯度为 0
        # 范围: [q_min * scale, q_max * scale]
        within_range = (x >= ctx.q_min * scale) & (x <= ctx.q_max * scale)
        grad_input = grad_output * within_range.float()

        # scale 的梯度（LSQ 风格，可选）
        # �?scale 求导，让 scale 也可学习
        grad_scale = None  # 简化版不更�?scale

        return grad_input, grad_scale, None, None, None


class FakeQuantize(nn.Module):
    def __init__(self, observer_class=MovingAverageMinMaxObserver, bits=8):
        super().__init__()
        self.observer = observer_class()
        self.bits = bits
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.register_buffer('initialized', torch.tensor(False))

    def forward(self, x):
        # 观测
        self.observer(x)
        if not self.initialized:
            self.scale, self.zero_point = self.observer.calculate_scale()
            self.initialized = True

        # FakeQuant
        q_min = -2 ** (self.bits - 1)
        q_max = 2 ** (self.bits - 1) - 1
        return FakeQuantizeFunction.apply(
            x, self.scale, self.zero_point, q_min, q_max
        )
```

### A2.5.5 QAT 的学习率调整

QAT 初期模型需要适应量化噪声，学习率策略与普通训练不同：

```
QAT 典型训练曲线:
  Epoch 1-3:  �?fp32 模型开始，lr = 0.1 × 原始lr（小 lr 适应量化噪声�?
  Epoch 4-10: lr = 原始lr（正常训练）
  Epoch 11+:  lr 衰减（收敛）

  �?QAT 通常只需 5-10% 的原始训�?epoch �?
  �?�?fp32 checkpoint 恢复，不是从头训
```

---

## A2.6 量化的粒�?

### A2.6.1 Per-Tensor Quantization

整个张量共用一�?scale�?

```
scale = max(abs(tensor)) / 127
```

**优点**：一�?scale，开销最小�?
**缺点**：不同通道的值域差异大时，精度差�?

### A2.6.2 Per-Channel Quantization

每个通道（如 weight 的每个输出通道）各有一�?scale�?

```
scale[c] = max(abs(tensor[c, :, :])) / 127  # 每个通道独立
```

**优点**：精度好，权重量化标准做法�?
**缺点**：scale 数组占用额外空间（但通常可忽略）�?

### A2.6.3 选择

| 量化对象 | 推荐粒度 | 理由 |
|---------|---------|------|
| 权重 | Per-Channel | 不同通道值域差异�?|
| 激�?| Per-Tensor | 激活值域较均匀，且 Per-Channel 推理开销�?|

### A2.6.4 Per-Channel 的数值对�?

```
Conv 权重 shape = [3, 3, 3, 3]  (3 个输出通道)

Per-Tensor:
  全局 max = 2.0 (�?channel 0)
  scale = 2.0 / 127 = 0.0157
  channel 1 �?max = 0.3 �?量化�?[0, 19]，只用了 15% 范围 �?精度�?

Per-Channel:
  scale[0] = 2.0 / 127 = 0.0157  (channel 0 范围�?
  scale[1] = 0.3 / 127 = 0.00236 (channel 1 范围小，精度�?
  scale[2] = 1.1 / 127 = 0.00866

  �?每个通道都用�?[-127, 127] 范围，整体精度更�?
```

---

## A2.7 量化算子的数�?

### A2.7.1 量化矩阵乘法

```
fp32:  Y = X * W
int8:  Y_q = X_q * W_q

反量化关�?
  X �?scale_x * X_q
  W �?scale_w * W_q
  Y �?scale_x * scale_w * (X_q * W_q)

因此:
  1. int8 矩阵乘法: Y_q = X_q * W_q  (硬件加�?
  2. 乘以 scale: Y = Y_q * scale_x * scale_w  (标量乘法，便�?
```

### A2.7.2 量化 ReLU

```
fp32:  y = relu(x) = max(0, x)
int8:  y_q = max(zero_point, x_q)  # zero_point 对应 0
```

ReLU 不改�?scale，只需 clip �?`zero_point` 以上�?

### A2.7.3 量化 Conv2d 的完整公�?

Conv2d 比矩阵乘法复杂，涉及 bias 和多通道�?

```
fp32 Conv2d:
  Y[n, oc, i, j] = bias[oc] + Σ_{ic, kh, kw} X[n, ic, i+kh, j+kw] * W[oc, ic, kh, kw]

量化 (Per-Channel 权重, Per-Tensor 激�?:
  X �?scale_x * (X_q - zp_x)
  W[oc] �?scale_w[oc] * W_q[oc]      # 每通道独立 scale
  bias �?scale_x * scale_w[oc] * bias_q  # bias �?int32

  Y_q = scale_x * scale_w[oc] * (
    Σ (X_q - zp_x) * W_q[oc] + bias_q / (scale_x * scale_w[oc])
  )

硬件执行:
  1. int8 × int8 �?int32 累加 (VNNI 指令)
  2. �?int32 bias
  3. �?scale_x * scale_w[oc] (fp32 标量)
  4. requantize �?int8 (round + clamp)
```

**关键**：int8×int8 的累加用 int32 防止溢出，最后再 requantize �?int8�?

### A2.7.4 BatchNorm 融合（BN Fold�?

推理�?BatchNorm 可以融合进前面的 Conv，省一次计算：

```
fp32 Conv + BN:
  y = BN(conv(x)) = γ * (conv(x) - μ) / �?σ² + ε) + β
    = (γ / �?σ² + ε)) * conv(x) + (β - γ * μ / �?σ² + ε))
    = γ' * conv(x) + β'

  其中:
    γ' = γ / �?σ² + ε)
    β' = β - γ * μ / �?σ² + ε)

融合�?
  W_fused = W * γ'     # 缩放权重
  b_fused = b * γ' + β'  # 调整 bias

  �?只需一�?Conv，BN 消失
```

**为什么量化必须先�?BN Fold**�?
- 量化 Conv 后再量化 BN 会引�?*两次量化误差**
- 融合后只量化一次，精度更好
- PyTorch �?`quant.fuse_modules([['conv', 'bn', 'relu']])` 自动做这件事

```python
# 融合�? Conv �?BN �?ReLU (3 个算�? 3 次量化误�?
# 融合�? ConvReLU (1 个算�? 1 次量化误�?
model_fused = torch.ao.quantization.fuse_modules(
    model, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']]
)
```

---

## A2.8 PyTorch 量化 API

### A2.8.1 PTQ 示例

```python
import torch.ao.quantization as quant

# 1. 准备模型
model_fp32 = MyModel().eval()

# 2. 插入量化桩（QuantStub / DeQuantStub�?
model_fp32 = quant.fuse_modules(model_fp32, [['conv', 'bn', 'relu']])
model_fp32.qconfig = quant.get_default_qconfig('fbgemm')

# 3. 校准：用校准数据观测激活范�?
model_calibrated = quant.prepare(model_fp32)
with torch.no_grad():
    for x in calibration_data:
        model_calibrated(x)

# 4. 转换为真正的 int8 模型
model_int8 = quant.convert(model_calibrated)
```

### A2.8.2 QAT 示例

```python
# 1. 插入 FakeQuantize
model_qat = quant.prepare_qat(model_fp32)
model_qat.train()

# 2. 训练（FakeQuantize 在前向模拟量化）
for epoch in range(epochs):
    for x, y in dataloader:
        loss = model_qat(x, y)
        loss.backward()
        optimizer.step()

# 3. 转换�?int8
model_int8 = quant.convert(model_qat.eval())
```

### A2.8.3 模块融合详解

`fuse_modules` 是量化的关键预处理步骤：

```python
# 常见融合模式
fuse_patterns = [
    ['conv', 'bn', 'relu'],      # Conv-BN-ReLU �?1 个算�?
    ['conv', 'bn'],               # Conv-BN �?1 个算�?
    ['conv', 'relu'],             # Conv-ReLU �?1 个算�?
    ['linear', 'relu'],           # Linear-ReLU �?1 个算�?
]

# 融合的数�?(Conv-BN-ReLU):
#   y = relu(bn(conv(x)))
#   = relu(γ * (conv(x) - μ) / �?σ²+ε) + β)
#   = relu(γ' * conv(x) + β')           �?BN 融进 Conv
#   = relu(conv_fused(x))               �?一个融合算�?

# 代码
class MyModel(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(3, 16, 3)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))  # 顺序: conv �?bn �?relu

model = MyModel().eval()
# 融合: �?conv/bn/relu 三个 Module 合并�?conv 一�?Module
model_fused = quant.fuse_modules(model, [['conv', 'bn', 'relu']])
# 现在 model_fused.forward 只调一�?conv（内部已�?BN+ReLU�?
```

**融合的收�?*�?
1. **减少算子�?*�?�?，减�?kernel launch 开销
2. **减少量化误差**：只量化 1 次而非 3 �?
3. **减少中间激�?*：不�?BN/ReLU 的中间结果，省显�?

### A2.8.4 量化调试

量化后精度下降，如何定位是哪一层的问题�?

```python
# 方法 1: 逐层对比 fp32 vs int8 输出
class QuantDebugLogger:
    def __init__(self, model_fp32, model_int8):
        self.model_fp32 = model_fp32
        self.model_int8 = model_int8

    def compare(self, x):
        # 注册 hook 记录每层输出
        fp32_outputs = {}
        int8_outputs = {}

        for name, module in self.model_fp32.named_modules():
            module.register_forward_hook(
                lambda m, inp, out, n=name: fp32_outputs.update({n: out})
            )
        for name, module in self.model_int8.named_modules():
            module.register_forward_hook(
                lambda m, inp, out, n=name: int8_outputs.update({n: out})
            )

        self.model_fp32(x)
        self.model_int8(x)

        # 对比每层
        for name in fp32_outputs:
            diff = (fp32_outputs[name] - int8_outputs[name].dequantize()).abs().mean()
            print(f"{name}: MSE = {diff:.6f}")

# 方法 2: 逐层量化敏感度分�?
def sensitivity_analysis(model, calibration_data):
    """逐层量化，看每层对精度的影响"""
    for layer_name in model.named_modules():
        # 只量化这一层，其他保持 fp32
        model_partial = quantize_only_one_layer(model, layer_name)
        accuracy = evaluate(model_partial, calibration_data)
        print(f"{layer_name}: accuracy = {accuracy}")
    # 找出精度下降最大的�?�?对该层用更高精度�?QAT
```

**常见问题与解�?*�?

| 问题 | 原因 | 解决 |
|------|------|------|
| 某层激�?outlier �?| 该层 Observer 范围过大 | �?HistogramObserver |
| 第一层精度差 | 输入数据未归一�?| 预处理归一化输�?|
| 最后一层精度差 | 输出范围�?| 最后一层不量化（keep fp32�?|
| 整体精度�?5%+ | 模型对量化不友好 | �?QAT 重新训练 |

---

## A2.9 硬件指令支持

### A2.9.1 CPU: AVX-VNNI

x86 CPU �?int8 矩阵乘法指令�?

```
AVX-VNNI (AVX512-VNNI):
  VPDPBUSD z, x, y:  z += Σ x[i] * y[i]  (int8 × int8 �?int32 累加)

  一次指令处�?16 �?int8 × int8 乘加
  �?�?fp32 FMA �?4×（数据宽�?4× + 指令吞吐�?

PyTorch 后端: FBGEMM (Facebook GEneral Matrix Multiplication)
  专门�?x86 int8 优化，用 AVX-VNNI + cache 友好的分�?
```

### A2.9.2 GPU: Tensor Core int8

NVIDIA Tensor Core（Volta+）支�?int8 矩阵乘法�?

```
Tensor Core IMMA (Integer Matrix Multiply-Accumulate):
  D = A × B + C
  A: int8 [16×16], B: int8 [16×16], C/D: int32 [16×16]
  �?一个时钟周期完�?16×16×16 = 4096 �?int8 乘加

  RTX 3090:  142 TFLOPS int8  (vs 35 TFLOPS fp32)
  A100:      624 TOPS int8    (vs 19.5 TFLOPS fp32)
```

### A2.9.3 其他硬件

| 硬件 | int8 支持 | 框架 |
|------|----------|------|
| Apple M1/M2 NPU | 是（Neural Engine�?| CoreML |
| Qualcomm Hexagon DSP | �?| Qualcomm SNPE |
| Intel NPU (Lunar Lake) | �?| OpenVINO |
| ARM Cortex-A (手机) | 部分（SDOT 指令�?| TFLite |
| Google TPU | 是（原生 int8�?| JAX/TF |

---

## A2.10 与真�?PyTorch 对照

| 概念 | PyTorch | 文件 |
|------|---------|------|
| Observer | `torch.ao.quantization.observer` | `torch/ao/quantization/observer.py` |
| FakeQuantize | `torch.ao.quantization.fake_quantize` | `torch/ao/quantization/fake_quantize.py` |
| PTQ | `quant.prepare` �?`quant.convert` | `torch/ao/quantization/quantize.py` |
| QAT | `quant.prepare_qat` �?`quant.convert` | 同上 |
| int8 kernel | `aten/src/ATen/native/quantized/` | FBGEMM / oneDNN 后端 |
| 量化算子分发 | `QuantizedCPU` dispatch key | `aten/src/ATen/Dispatch.h` |
| BN 融合 | `fuse_modules` | `torch/ao/quantization/fuser.py` |
| 量化敏感�?| `QuantizationAccuracyTester` | `torch/ao/quantization/_quantize.py` |

---

## A2.11 优劣势总结

| 优势 | 劣势 |
|------|------|
| 4× 显存压缩 | 精度损失 0.5-3% |
| 2-4× 推理加�?| 仅推理用，训练仍需 fp32/fp16 |
| 硬件广泛支持 | 量化流程复杂（fuse/prepare/convert�?|

### A2.11.1 更激进的量化

| 方案 | bit | 精度 | 硬件支持 |
|------|-----|------|---------|
| int8 | 8 | �?| 广泛 |
| int4 | 4 | 需 QAT | 部分新硬�?|
| 二值网�?(BNN) | 1 | �?| 学术 |
| 混合精度量化 | 8/4 �?| �?| �?|

### A2.11.2 混合精度量化

不是所有层都量化到同一 bit，按敏感度分配：

```
敏感度分析结�?
  conv1: 量化后精度降 0.1% �?int8
  conv2: 量化后精度降 0.05% �?int8
  conv3: 量化后精度降 3.0% �?保持 fp16 (敏感层不量化)
  conv4: 量化后精度降 0.2% �?int8

�?混合精度: 大部�?int8 + 少数 fp16，精度接�?fp32，速度接近 int8
```

```python
# PyTorch 混合精度量化
model.qconfig = quant.get_default_qconfig('fbgemm')

# 对敏感层用更高精�?
for name, module in model.named_modules():
    if name in sensitive_layers:
        module.qconfig = None  # 不量化该�?
```

---

## A2.12 minitorch 量化草图

minitorch 未实现量化（教学范围外），但可以基于已有模块勾勒框架�?

```python
# minitorch/quantization.py (草图，未实现)
from minitorch import Tensor
from minitorch.nn import Module

class QuantStub(Module):
    """前向: fp32 �?int8 (实际只标记，不真量化)"""
    def __init__(self):
        self.observer = MovingAverageMinMaxObserver()
        self.scale = None
        self.zero_point = None

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            self.observer(x)
        return x  # minitorch �?int8 Tensor，保�?fp32

class DeQuantStub(Module):
    """前向: int8 �?fp32"""
    def forward(self, x: Tensor) -> Tensor:
        return x

class QuantizeLinear(Module):
    """量化线性层: Conv/Linear + FakeQuantize"""
    def __init__(self, in_features, out_features):
        self.linear = Linear(in_features, out_features)
        self.input_quant = FakeQuantize()
        self.weight_quant = FakeQuantize()

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_quant(x)
        w = self.weight_quant(self.linear.weight)
        return self.linear.forward_with_weight(x, w)
```

**为什�?minitorch 不实现完整量�?*�?
1. 需�?int8 Tensor 类型（minitorch 只有 fp32�?
2. 需�?int8 算子内核（需 C++ AVX/VNNI 实现�?
3. 量化流程的框架代码量大（prepare/convert/fuse�?
4. 教学目标已由本附录的原理讲解覆盖

---

## A2.13 小结

量化的三个核心概念：

1. **量化映射**：`q = round(r / scale + zero_point)`，scale 由数据范围决�?
2. **Observer**：观测激�?权重�?min/max，计算最�?scale
3. **FakeQuantize**：前向量化引入误差，反向 STE 传梯度，让模型适应量化

PTQ 快但精度差，QAT 精度好但需训练。int8 是工程甜点，更低 bit 需要更精细的方法�?

**关键工程实践**�?
- 权重�?Per-Channel 对称量化，激活用 Per-Tensor 非对称量�?
- 先做 BN Fold 再量化（减少量化误差�?
- 校准数据要代表性、多样化
- 精度下降大时做敏感度分析，对敏感层保持高精度
- QAT �?fp32 checkpoint 恢复，只需 5-10% 原始训练�?
