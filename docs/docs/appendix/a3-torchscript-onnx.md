# A3 TorchScript 与 ONNX export

> 本附录对应主线 Ch13（图与编译导论）。TorchScript 是 PyTorch 的第一代图编译方案，已被 `torch.compile` 取代；ONNX 是跨框架模型交换格式，至今广泛使用。

---

## A3.1 TorchScript：把动态模型变静态

### A3.1.1 为什么需要 TorchScript

PyTorch 模型是 Python 代码——部署到 C++ 环境（如手机、嵌入式、服务器 C++ 后端）时，没有 Python 解释器。TorchScript 把 Python 模型**编译成一个可序列化的图**，脱离 Python 运行。

```
Python 模型 → TorchScript 编译 → ScriptModule → 序列化 → C++ 加载执行
```

**部署场景**：
- C++ 服务器后端（libtorch）
- 移动端（PyTorch Mobile）
- 嵌入式设备（无 Python 解释器）
- 需要序列化 + 版本化的生产环境

### A3.1.2 两种编译方式

**Tracing（追踪）**：

```python
model = MyModel().eval()
example_input = torch.randn(1, 3, 224, 224)
scripted = torch.jit.trace(model, example_input)
```

用示例输入跑一次前向，**记录所有算子调用**，构建图。

- 优点：简单，不需要改代码
- 缺点：**控制流被固化**——if/for 按示例输入的路径走，另一条路径丢失

**Scripting（脚本化）**：

```python
scripted = torch.jit.script(model)
```

**分析 Python 源码**，把控制流（if/for/while）翻译成图节点。

- 优点：保留完整控制流
- 缺点：需要代码符合 TorchScript 子集（类型注解、无动态属性等）

### A3.1.3 Tracing 的控制流问题

```python
class MyModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:      # ← data-dependent control flow
            return x + 1
        else:
            return x - 1

# trace 时如果 example_input.sum() > 0
scripted = torch.jit.trace(model, example_input)
# → 图里只有 "add"，else 分支丢失！
# 推理时遇到 sum < 0 的输入，结果错误
```

**解决**：用 `torch.jit.script` 保留 if/else：

```python
scripted = torch.jit.script(model)
# → 图里有 "if" 节点，两条路径都在
```

### A3.1.4 Tracing vs Scripting 的选择

| 情况 | 推荐 | 原因 |
|------|------|------|
| 无控制流（纯算子序列） | trace | 简单，无需改代码 |
| 控制流依赖输入值 | script | trace 会丢分支 |
| 控制流依赖 shape（非值） | trace | shape 在 trace 时已知，不是 data-dependent |
| 复杂 Python 语法 | trace | script 可能不支持 |
| 需要精确控制图 | script | 可手动优化图 |

```python
# shape-dependent control flow → trace 安全
class Model(nn.Module):
    def forward(self, x):
        if x.dim() == 4:          # ← shape-dependent, trace OK
            x = x.flatten(1)
        return self.fc(x)

# data-dependent control flow → 必须 script
class Model(nn.Module):
    def forward(self, x):
        if x.mean() > 0.5:        # ← data-dependent, trace 危险
            return self.branch_a(x)
        else:
            return self.branch_b(x)
```

### A3.1.5 ScriptModule

TorchScript 编译后得到 `ScriptModule`，它是一个**图结构**而非 Python 代码：

```python
scripted = torch.jit.script(model)

# 查看图
print(scripted.graph)
# graph(%self : __torch__.MyModule, %x.1 : Tensor):
#   %1 : Tensor = aten::sum(%x.1)
#   %2 : bool = aten::gt(%1, 0)
#   %3 : Tensor = prim::If(%2)
#     block0():
#       %4 : Tensor = aten::add(%x.1, 1)
#       -> (%4)
#     block1():
#       %5 : Tensor = aten::sub(%x.1, 1)
#       -> (%5)
#   return (%3)

# 序列化（脱离 Python）
scripted.save("model.pt")

# C++ 端加载
# auto module = torch::jit::load("model.pt");
# auto output = module.forward({input});
```

### A3.1.6 TorchScript IR 细节

TorchScript IR 是一个**SSA（静态单赋值）图**：

```
graph(%self : __torch__.MyModule, %x.1 : Tensor):
  # %self, %x.1 是输入值（SSA 变量，% 前缀）
  # 类型注解: Tensor, __torch__.MyModule

  %1 : Tensor = aten::sum(%x.1)
  # %1 是新 SSA 变量，由 aten::sum 算子计算

  %2 : bool = aten::gt(%1, 0)
  # 常量 0 自动提升为 Tensor

  %3 : Tensor = prim::If(%2)
  # prim::If 是控制流节点，包含两个 block
    block0():           # true 分支
      %4 : Tensor = aten::add(%x.1, 1)
      -> (%4)           # block 返回值
    block1():           # false 分支
      %5 : Tensor = aten::sub(%x.1, 1)
      -> (%5)

  return (%3)
```

**IR 元素**：
- **Value**：SSA 变量，`%name : Type`
- **Node**：算子调用，`%output = op(%inputs)`
- **Block**：控制流体（if 的分支、loop 的循环体）
- **Graph**：顶层图，包含所有 Node

**算子命名空间**：
- `aten::`：ATen 算子（`add`, `mul`, `conv2d`...）
- `prim::`：原语（`If`, `Loop`, `Constant`, `GetAttr`...）
- `__torch__.`：用户定义的 Module

### A3.1.7 TorchScript 的子集限制

TorchScript 不支持所有 Python 语法：

| 支持 | 不支持 |
|------|--------|
| if/for/while（静态可分析） | 动态类定义、exec/eval |
| 标准类型（int, float, str, List, Dict） | 任意 Python 对象 |
| Tensor 算子 | 第三方库调用（numpy 等） |
| nn.Module 子类 | 多继承、元类 |
| 类型注解的函数 | 可变参数 *args/**kwargs（部分支持） |
| List[Tensor] 等容器 | 容器内混合类型 |

```python
# TorchScript 要求类型注解
@torch.jit.script
def my_fn(x: Tensor, n: int) -> Tensor:
    result = torch.zeros_like(x)
    for i in range(n):       # ← n 必须是 int，不能是 Tensor
        result = result + x * i
    return result
```

**常见报错与修复**：

```python
# 报错: "Unsupported statement type"
# 原因: 用了 TorchScript 不支持的语法（如 try/except）
# 修复: 把逻辑移到 @torch.jit.script 外面

# 报错: "Expected a value of type 'Tensor' but got 'Optional[Tensor]'"
# 原因: 类型推导失败，可能为 None
# 修复: 显式类型断言
def forward(self, x):
    y = self.maybe_none(x)  # 返回 Optional[Tensor]
    if y is not None:
        return y
    return x  # TorchScript 能推导，但有时需 torch.jit.annotate

# 报错: "unknown attribute"
# 原因: 动态添加属性
# 修复: 在 __init__ 中声明所有属性

# 报错: "torch.jit.script does not support ..."
# 原因: 用了不支持的 Python 特性
# 修复: 重写为 TorchScript 兼容代码，或改用 torch.jit.trace
```

### A3.1.8 TorchScript 的图优化

TorchScript 支持图级优化（在 ScriptModule 上）：

```python
scripted = torch.jit.script(model)

# 常量折叠
scripted = torch.jit.freeze(scripted)
# → 把运行时不变的属性折叠成常量，消除 GetAttr 节点

# 内联
scripted = torch.jit._pass_manager.inline(scripted)
# → 把子 Module 调用展开到主图

# 死代码消除
scripted = torch.jit._pass_manager.dce(scripted)
# → 删除不影响输出的节点

# 查看优化后的图
print(scripted.graph)
# 优化后: 更少节点，更直的数据流
```

**freeze 的效果**：

```
优化前:
  %weight = prim::GetAttr(%self, "weight")  # 运行时查找属性
  %bias = prim::GetAttr(%self, "bias")
  %1 = aten::linear(%x, %weight, %bias)

优化后 (freeze):
  %weight = prim::Constant(value=...)  # 编译时内联权重
  %bias = prim::Constant(value=...)
  %1 = aten::linear(%x, %weight, %bias)
  → 消除了 GetAttr 开销
```

---

## A3.2 TorchScript 的衰落

### A3.2.1 为什么被 torch.compile 取代

| 问题 | TorchScript | torch.compile |
|------|-------------|---------------|
| 控制流 | 需要手写符合子集的代码 | 自动处理（Dynamo 字节码追踪） |
| 性能 | 仅图执行，无 kernel 融合 | Inductor 自动融合 + Triton codegen |
| 易用性 | 需改代码、处理报错 | `model = torch.compile(model)` 一行 |
| 动态 shape | 需重新 trace | Guard 自动切分 |

**结论**：PyTorch 2.0+ 后 TorchScript 进入维护模式，新项目用 `torch.compile`。

### A3.2.2 TorchScript 的遗产

TorchScript 虽然被取代，但留下了重要遗产：

- **IR 设计**：TorchScript IR（`torch::jit::Graph`）影响了 FX Graph 和 Dynamo 的设计
- **C++ 序列化**：`.pt` 格式的 C++ 加载能力仍被使用
- **类型系统**：TorchScript 的类型推导思想延续到 `torch.jit.annotate`
- **Mobile 部署**：TorchScript Mobile 至今仍是 PyTorch 移动端方案

---

## A3.3 ONNX：跨框架模型交换

### A3.3.1 什么是 ONNX

ONNX（Open Neural Network Exchange）是一个**开放的模型表示格式**。不同框架（PyTorch、TensorFlow、MXNet）都能导出 ONNX，不同推理引擎（ONNX Runtime、TensorRT、OpenVINO）都能加载 ONNX。

```
PyTorch ─┐
TensorFlow ─┼──→ ONNX ──→ ONNX Runtime / TensorRT / OpenVINO
MXNet ──┘
```

**ONNX 的价值**：
- 框架无关：训练用 PyTorch，推理用 TensorRT（不用重写模型）
- 引擎竞争：多个推理引擎支持同一格式，用户选最优的
- 标准化：算子语义统一定义，避免各框架差异

### A3.3.2 ONNX 的结构

ONNX 模型是一个**有向无环图（DAG）**，用 protobuf 序列化：

```protobuf
// ONNX 模型结构（简化）
ModelProto {
  GraphProto {
    NodeProto[] nodes     // 算子节点
    ValueInfoProto[] inputs   // 输入
    ValueInfoProto[] outputs  // 输出
    TensorProto[] initializers  // 权重
  }
  OpSetImport[] opset_import  // 算子集版本
}

// 节点示例
NodeProto {
  op_type: "Conv"
  inputs: ["x", "W", "b"]
  outputs: ["y"]
  attributes: {
    kernel_shape: [3, 3]
    strides: [1, 1]
    pads: [1, 1, 1, 1]
  }
}
```

**protobuf 的优势**：
- 跨语言（Python/C++/Java/Go 都能读写）
- 向后兼容（新增字段不破坏旧解析器）
- 紧凑的二进制格式（比 JSON 小）

### A3.3.3 PyTorch 导出 ONNX

```python
import torch

model = MyModel().eval()
example_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    opset_version=17,        # ONNX 算子集版本
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={           # 支持动态 batch
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)
```

### A3.3.4 导出原理：trace + 翻译

ONNX export 的流程与 TorchScript trace 类似：

```
1. 用 example_input 跑前向 → 记录算子调用序列
2. 把 PyTorch 算子翻译成 ONNX 算子:
   aten::conv2d  → onnx::Conv
   aten::relu    → onnx::Relu
   aten::matmul  → onnx::MatMul
   aten::add     → onnx::Add
3. 序列化为 protobuf 格式 → model.onnx
```

**详细流程**：

```python
# torch.onnx.export 内部:
# 1. 创建 "tracer" 模式，拦截所有 aten:: 算子调用
# 2. 执行 model(example_input) → 记录算子序列
# 3. 对每个算子，查 symbolic_opset{N}.py 找到对应的 ONNX 符号函数
# 4. 符号函数把 aten 算子翻译成 ONNX NodeProto
# 5. 把所有 NodeProto 组装成 GraphProto → ModelProto
# 6. protobuf 序列化 → model.onnx

# 符号函数示例 (torch/onnx/symbolic_opset17.py):
def add(g, self, other, alpha=None):
    if alpha is not None and alpha != 1:
        other = g.op("Mul", other, g.op("Constant", value_t=torch.tensor(alpha)))
    return g.op("Add", self, other)
# → 把 aten::add(alpha) 翻译成 ONNX Mul + Add
```

### A3.3.5 算子映射表

| PyTorch 算子 | ONNX 算子 | 备注 |
|-------------|-----------|------|
| `torch.conv2d` | `Conv` | 直接映射 |
| `torch.relu` | `Relu` | 直接映射 |
| `torch.matmul` | `MatMul` | 直接映射 |
| `torch.add` | `Add` | 直接映射 |
| `torch.softmax` | `Softmax` | opset 1+ |
| `torch.layer_norm` | `LayerNormalization` | opset 17+ |
| `torch.einsum` | 无直接对应 | 需拆分成 MatMul + Transpose |
| `torch.flip` | `Slice` | 用 Slice 实现 |
| `torch.where` | `Where` | opset 9+ |
| `torch.gather` | `Gather` | opset 1+ |

**问题**：不是所有 PyTorch 算子都有 ONNX 对应。遇到不支持的算子，需要：
1. 注册自定义 ONNX 算子（`torch.onnx.register_custom_op`）
2. 或用 `torch.jit.script` 替代 trace（保留控制流，但 ONNX 对控制流支持有限）

### A3.3.6 自定义算子导出

```python
# 自定义算子: my_hardswish(x) = x * relu6(x+3) / 6
class HardSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * torch.clamp(x + 3, 0, 6) / 6

# 注册 ONNX 符号函数
def hardswish_symbolic(g, x):
    # 用已有 ONNX 算子组合实现
    three = g.op("Constant", value_t=torch.tensor(3.0))
    six = g.op("Constant", value_t=torch.tensor(6.0))
    x_plus_3 = g.op("Add", x, three)
    clamped = g.op("Clip", x_plus_3, g.op("Constant", value_t=torch.tensor(0.0)),
                   six)
    mul = g.op("Mul", x, clamped)
    return g.op("Div", mul, six)

# 注册
from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic("my_ops::hardswish", hardswish_symbolic, 1)

# 现在导出时会用 hardswish_symbolic 翻译
torch.onnx.export(model_with_hardswish, x, "model.onnx", opset_version=17)
```

### A3.3.7 动态轴

默认 ONNX export 固化 batch size（trace 时的 batch）。`dynamic_axes` 允许指定维度为动态：

```python
torch.onnx.export(model, x, "model.onnx",
    dynamic_axes={"input": {0: "batch"}}  # 第 0 维动态
)
# 导出的模型支持任意 batch size
```

**动态轴的限制**：
- 只能指定**整维**为动态（不能"维度 1 是 3 或 4"）
- 动态维度影响推理引擎优化（固定 shape 更好优化）
- 某些算子（如 reshape 到固定 shape）与动态轴冲突

```python
# 常见动态轴设置
dynamic_axes = {
    "input":  {0: "batch", 2: "height", 3: "width"},  # 动态 batch + 分辨率
    "output": {0: "batch"},
}

# 生产建议: 如果 batch 固定，不要设动态 → 推理引擎优化更好
```

### A3.3.8 Opset 版本

ONNX 算子集（OpSet）有版本演进，新版本增加算子或修改语义：

| OpSet | 关键新增 | PyTorch 支持 |
|-------|---------|-------------|
| 9 | `Where`, `NonZero` | PyTorch 1.x |
| 11 | `DynamicQuantizeLinear`, `Pad` 修改 | PyTorch 1.6+ |
| 13 | `Squeeze`/`Unsqueeze` 改用 axes 输入 | PyTorch 1.10+ |
| 17 | `LayerNormalization`, `Resize` 修改 | PyTorch 1.13+ |
| 18 | `DynamicQuantizeLinear` 增强 | PyTorch 2.0+ |
| 20 | `Attention`, `GroupNormalization` | PyTorch 2.1+ |

```python
# 选择 opset 版本
torch.onnx.export(model, x, "model.onnx", opset_version=17)
# → 用 opset 17 的算子定义
# → 推理引擎必须支持 opset 17

# 查看模型用的 opset
import onnx
model = onnx.load("model.onnx")
print(model.opset_import)  # [opset: domain="ai.onnx.ml", version=1, ...]
```

**选择建议**：用推理引擎支持的最高版本（新 opset 算子更丰富，导出成功率高）。

### A3.3.9 ONNX 控制流

ONNX 从 opset 11 起支持控制流算子：

```protobuf
// If 算子
NodeProto {
  op_type: "If"
  inputs: ["cond"]           // bool 条件
  attributes: {
    then_branch: GraphProto { ... }   // true 分支（子图）
    else_branch: GraphProto { ... }   // false 分支（子图）
  }
}

// Loop 算子
NodeProto {
  op_type: "Loop"
  inputs: ["max_iter", "cond", "initial_state"]
  attributes: {
    body: GraphProto { ... }  // 循环体（子图）
  }
}
```

**限制**：
- 控制流用子图表示，比 TorchScript 的 `prim::If` 更重
- 不是所有推理引擎都支持 ONNX 控制流（ORT 支持，TensorRT 部分支持）
- 导出 data-dependent 控制流仍需 `torch.jit.script`（trace 无法捕获）

```python
# 带控制流的模型导出
class Model(nn.Module):
    def forward(self, x):
        if x.sum() > 0:           # data-dependent
            return torch.relu(x)
        else:
            return torch.sigmoid(x)

# 方法 1: script + export（推荐）
scripted = torch.jit.script(Model())
torch.onnx.export(scripted, x, "model.onnx", opset_version=17)
# → 导出包含 If 算子的 ONNX

# 方法 2: 直接 export（trace，会丢控制流——危险）
torch.onnx.export(Model(), x, "model.onnx", opset_version=17)
# → 只导出 trace 时走的分支，另一分支丢失
```

### A3.3.10 ONNX 模型检查

```python
import onnx
import onnx.shape_inference

# 加载
model = onnx.load("model.onnx")

# 1. 合法性检查
onnx.checker.check_model(model)

# 2. shape 推导（填充中间 shape 信息）
model_with_shapes = onnx.shape_inference.infer_shapes(model)

# 3. 打印模型信息
print(f"IR version: {model.ir_version}")
print(f"Opset: {model.opset_import[0].version}")
print(f"Inputs: {[i.name for i in model.graph.input]}")
print(f"Outputs: {[o.name for o in model.graph.output]}")
print(f"Nodes: {len(model.graph.node)}")
print(f"Initializers (weights): {len(model.graph.initializer)}")

# 4. 打印所有算子
for node in model.graph.node:
    print(f"  {node.op_type}: {list(node.input)} → {list(node.output)}")
```

---

## A3.4 ONNX Runtime 推理

```python
import onnxruntime as ort

# 加载 ONNX 模型
session = ort.InferenceSession("model.onnx")

# 推理
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: x.numpy()})

# 性能优化：指定 provider
session = ort.InferenceSession("model.onnx", providers=[
    "CUDAExecutionProvider",   # GPU
    "CPUExecutionProvider",    # CPU fallback
])
```

ONNX Runtime 的图优化（融合、常量折叠、内存规划）与 `torch.compile` 的 Inductor 类似，但作用在 ONNX IR 上。

### A3.4.1 ORT 的图优化

```python
# 启用所有优化
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_file = "model_optimized.onnx"  # 保存优化后的图

session = ort.InferenceSession("model.onnx", sess_options, providers=["CPUExecutionProvider"])
```

**优化级别**：
- `ORT_DISABLE_ALL`：无优化
- `ORT_ENABLE_BASIC`：常量折叠、冗余算子消除
- `ORT_ENABLE_EXTENDED`：算子融合（Conv+BN+ReLU → ConvFusion）
- `ORT_ENABLE_ALL`：+ 布局转换、内存规划

### A3.4.2 ORT 性能调优

```python
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4      # 算子内并行（多线程算子）
sess_options.inter_op_num_threads = 1      # 算子间并行（通常 1）
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # 顺序执行
# sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # 并行执行独立算子

# 内存 arena（减少分配开销）
sess_options.enable_mem_pattern = True     # 启用内存模式
sess_options.enable_cpu_mem_arena = True   # CPU 内存 arena
```

---

## A3.5 ONNX vs TorchScript vs torch.compile

| 维度 | TorchScript | ONNX | torch.compile |
|------|-------------|------|---------------|
| 目标 | 脱离 Python 部署 | 跨框架交换 | 训练 + 推理加速 |
| IR | TorchScript IR | ONNX IR | FX Graph / Dynamo IR |
| 控制流 | script 支持 | 有限支持（If/Loop 算子） | 自动（Dynamo） |
| 序列化 | `.pt` (pickle) | `.onnx` (protobuf) | 不序列化（JIT 编译） |
| 推理引擎 | libtorch C++ | ORT / TensorRT / OpenVINO | PyTorch 自身 |
| 状态 | 维护模式 | 活跃 | 活跃（未来方向） |

### A3.5.1 选择建议

| 场景 | 推荐 |
|------|------|
| 训练加速 | `torch.compile` |
| C++ 部署（libtorch） | TorchScript（或 torch.export） |
| 跨框架/第三方推理引擎 | ONNX |
| 边缘设备（TensorRT/OpenVINO） | ONNX |
| 生产推理服务 | ONNX Runtime |
| 移动端 | TorchScript Mobile / torch.export |

---

## A3.6 torch.export：新一代导出

PyTorch 2.1+ 引入 `torch.export`，替代 TorchScript 的导出功能：

```python
# 新方式
exported = torch.export.export(model, (example_input,))
exported.save("model.pt2")

# C++ 加载
# auto module = torch::inductor::load("model.pt2");
```

优势：
- 不需要 TorchScript 子集限制
- 支持 dynamic shapes（`torch.export.Dim`）
- 与 `torch.compile` 共用 IR

### A3.6.1 动态 shape

```python
# 声明动态维度
batch = torch.export.Dim("batch", min=1, max=256)
exported = torch.export.export(
    model,
    (torch.randn(32, 3, 224, 224),),
    dynamic_shapes={"x": {0: batch}},
)
# → 导出的模型支持 batch=1 到 256
```

### A3.6.2 torch.export 的 IR

`torch.export` 用 **ExportedProgram**，基于 FX Graph + ATen 算子：

```python
print(exported.graph)
# graph():
#   %x : Tensor [batch, 3, 224, 224]
#   %weight_1 : Tensor [16, 3, 3, 3]  # 常量权重
#   %conv_out = aten.conv2d(%x, %weight_1, ...)
#   %relu_out = aten.relu(%conv_out)
#   return (%relu_out)

# 特点:
# - 纯 ATen 算子（无 prim::If，控制流已展开或用 higher-order op）
# - 显式 shape 符号（batch 是符号，不是固定值）
# - 权重作为 graph 输入（不是 attribute），方便共享
```

---

## A3.7 导出 troubleshooting

### A3.7.1 常见导出错误

```python
# 错误 1: "Unsupported operand"
# 原因: 用了 ONNX 不支持的算子
# 诊断: torch.onnx.export(..., verbose=True) 看哪个算子失败
# 修复: 用等价的支持算子替换，或注册自定义算子

# 错误 2: "Tracer encountered unsupported operation"
# 原因: 控制流或动态操作
# 修复: 用 torch.jit.script 先转 ScriptModule，再导出

# 错误 3: "dynamic_axes specified but dimension is fixed"
# 原因: 某操作固化了维度（如 reshape 到固定 shape）
# 修复: 检查 reshape/view 操作，确保与动态轴兼容

# 错误 4: "opset version too low"
# 原因: 算子需要更高 opset
# 修复: 提高 opset_version
```

### A3.7.2 导出验证

```python
import onnx
import onnxruntime as ort
import numpy as np

# 1. 检查 ONNX 模型合法性
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)  # 不报错 = 合法

# 2. 对比 PyTorch 和 ONNX Runtime 输出
model.eval()
with torch.no_grad():
    pt_output = model(example_input)

session = ort.InferenceSession("model.onnx")
onnx_output = session.run(None, {input_name: example_input.numpy()})[0]

# 3. 数值对比
np.testing.assert_allclose(
    pt_output.numpy(), onnx_output,
    rtol=1e-3, atol=1e-5  # ONNX 用 fp32，误差应很小
)
print("导出验证通过！")
```

### A3.7.3 简化模型

导出的 ONNX 模型可能有冗余算子，用 `onnxsim` 简化：

```python
import onnxsim

# 简化（常量折叠 + 死代码消除）
model_simplified, check = onnxsim.simplify("model.onnx")
onnx.save(model_simplified, "model_sim.onnx")

# 通常能减少 10-30% 算子数，推理更快
```

---

## A3.8 与真实 PyTorch 对照

| 概念 | PyTorch | 文件 |
|------|---------|------|
| TorchScript trace | `torch.jit.trace` | `torch/jit/_trace.py` |
| TorchScript script | `torch.jit.script` | `torch/jit/_script.py` |
| ScriptModule | `torch.jit.ScriptModule` | `torch/jit/_script.py` |
| TorchScript IR | `torch::jit::Graph` | `torch/csrc/jit/ir/graph.h` |
| ONNX export | `torch.onnx.export` | `torch/onnx/utils.py` |
| ONNX 算子映射 | `torch.onnx.symbolic_opset*.py` | `torch/onnx/symbolic_opset17.py` |
| torch.export | `torch.export.export` | `torch/export/` |
| ExportedProgram | `torch.export.ExportedProgram` | `torch/export/exported_program.py` |

---

## A3.9 minitorch 的图与导出

minitorch 在 Ch13 实现了 FX 风格的图捕获（`fx` 模块），可对照理解导出原理：

```python
# minitorch 的图捕获 (Ch13)
from minitorch.fx import symbolic_trace

# 把 minitorch 模型变成图
graph = symbolic_trace(model)
print(graph)
# Graph:
#   %x = placeholder
#   %1 = conv2d(%x, weight=...)
#   %2 = relu(%1)
#   %3 = linear(%2, weight=...)
#   return %3

# 与 ONNX export 的关系:
# - minitorch fx: 捕获图用于优化/编译（类似 torch.compile）
# - ONNX export: 捕获图 + 翻译成 ONNX 算子
# - 共同点: 都是 "trace 前向 → 记录算子 → 构建图"
```

**minitorch 未实现 ONNX 导出**（教学范围外），但 fx 模块展示了"图捕获"的核心思想，这是所有导出方案的基础。

---

## A3.10 TensorRT 与边缘部署

### A3.10.1 TensorRT 工作流

TensorRT 是 NVIDIA 的高性能推理引擎，从 ONNX 构建：

```
ONNX 模型 → TensorRT Builder → 优化引擎 → 序列化 .engine → 部署
```

```python
import tensorrt as trt

# 1. 创建 builder
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

# 2. 解析 ONNX
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

# 3. 配置优化
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace

# 4. 构建（耗时，一次性）
serialized = builder.build_serialized_network(network, config)

# 5. 保存引擎
with open("model.engine", "wb") as f:
    f.write(serialized)
```

### A3.10.2 TensorRT 的优化

| 优化 | 效果 |
|------|------|
| 算子融合 | Conv+BN+ReLU → 1 个 kernel |
| 精度校准 | fp32 → int8（用 ONNX 范围校准） |
| Kernel auto-tuning | 选最优 kernel（针对当前 GPU） |
| 内存规划 | 复用 buffer，减少分配 |
| Layer fusion | 横向融合（同层独立算子并行） |

**性能**：TensorRT 通常比 ONNX Runtime（CUDA）快 2-5×，因为做了 GPU 特定优化。

### A3.10.3 其他推理引擎

| 引擎 | 平台 | 特点 |
|------|------|------|
| ONNX Runtime | CPU/GPU | 通用，易用 |
| TensorRT | NVIDIA GPU | 最快，NVIDIA 专属 |
| OpenVINO | Intel CPU/NPU | Intel 硬件优化 |
| CoreML | Apple | Apple Neural Engine |
| TFLite | 移动端 | ARM 优化 |
| TVM | 跨平台 | 自动 kernel 生成 |

**选择**：目标硬件有什么，就用对应的引擎。ONNX 是它们共同的输入格式。

---

## A3.11 TorchScript Mobile

TorchScript 的一个仍活跃的用例是移动端部署：

```python
# 导出移动端模型
scripted = torch.jit.script(model)
scripted = torch.jit.freeze(scripted)  # 消除可变属性
scripted._save_for_lite_interpreter("model_lite.pt")

# 移动端加载 (C++)
# auto module = torch::jit::load("model_lite.pt");
// Lite Interpreter 比 full TorchScript 轻量 10×
```

**移动端优化**：
- `torch.jit.freeze`：消除运行时属性查找
- Lite Interpreter：精简的字节码解释器
- 量化：int8 进一步减小模型体积
- 算子白名单：只编译用到的算子，减小二进制体积

---

## A3.12 小结

**TorchScript** 是 PyTorch 第一代"脱离 Python"方案，已被 `torch.compile` / `torch.export` 取代，但它的 IR 设计影响了后续方案。

**ONNX** 是跨框架模型交换标准，至今活跃。导出流程是 trace + 算子翻译，推理用 ONNX Runtime / TensorRT 等专用引擎。

**torch.export** 是新一代导出，支持动态 shape，与 torch.compile 共用 IR，是未来方向。

三者的共同思想：**把动态 Python 模型变成静态图**——这正是 Ch13 讲的"图与编译"的核心动机。

**关键工程实践**：
- 导出前先 `model.eval()`，固定 BN/Dropout 行为
- 用 `onnx.checker.check_model` 验证导出合法性
- 用 `np.testing.assert_allclose` 对比 PyTorch 和 ONNX 输出
- 用 `onnxsim` 简化模型，减少冗余算子
- 选推理引擎支持的最高 opset 版本
- 动态轴只在需要时设（固定 shape 推理更快）
