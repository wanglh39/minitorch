# �?13 章：图与编译导论

> 前面所有章节里，模型都�?命令�?执行的：一行行 Python 跑，前向算完丢给反向�?
> 这简单直接，但有个问题—�?*Python 解释器很�?*，每算一个算子都要走一次派发、检�?shape、查 dtype…�?
> 如果我们能在执行�?*先把模型记录成一张图**，然后对图做优化（融合算子、重排、并行），再一次性执行，就能快很多�?
> 这就�?`torch.fx`：把 PyTorch 模型变成可操纵的图。本章我们从零实�?fx 的核心：Proxy 追踪、Graph 数据结构、GraphModule 解释执行、算子融�?pass�?
> 这是通往 `torch.compile`（Dynamo + Inductor）的第一步�?

---

## 13.1 本章目标

读完这一章，你应当能够：

1. 解释**符号追踪**的原理：�?Proxy 替换真实 Tensor，拦截运算记录成 Node�?
2. 说出 `Proxy.__getattr__` 如何把方法调用（�?`x.sum()`）也记录进图�?
3. 区分 Node 的四�?op：`placeholder` / `call_function` / `call_method` / `output`，各代表什么�?
4. 写出 `Graph.codegen` 生成 Python 源码的过程，并解释为什�?codegen 有用�?
5. 描述 `GraphModule.forward` 的解释执行：遍历 Node，用 `env` 字典存中间值�?
6. 实现一个简单的算子融合 pass：模式匹�?+ 图重建�?
7. 说清 FX 的根本局限：**不能处理 data-dependent control flow**（如 `if x > 0`）�?
8. 对比 FX �?`torch.compile`（Dynamo + Inductor）的差异，知道为什么后者更强�?

---

## 13.2 原理铺垫

### 13.2.1 命令�?vs 图式：两种执行模�?

**命令式（eager�?*�?

```python
def f(x):
    a = x + 1
    b = a * 2
    return b
```

每行立刻执行，Python 解释器一步步走。好处：好调试、能 print、能 if。坏处：慢，且无法跨算子优化（解释器不知�?`+1` 后面紧跟 `*2`，没法融合成 `(x+1)*2` 一个算子）�?

**图式（graph/symbolic�?*�?

```python
# 不真的算，只记录"要算什�?
graph:
    x = placeholder          # 输入
    a = add(x, 1)            # 记录一个加法节�?
    b = mul(a, 2)            # 记录一个乘法节�?
    return b
```

先建图，再对图做变换（融合、并行、内存规划），最后一次性执行。好处：能全局优化。坏处：建图有开销、不好调试、不是所�?Python 都能记录进图�?

FX 就是把命令式模型**转成�?*的工具。转完之后你可以�?
- 打印图（看模型到底在算什么）�?
- 对图做变换（融合算子、剪枝、量化）�?
- 重新生成代码（codegen 出一个等价但更快�?forward）�?

### 13.2.2 符号追踪：用 Proxy 拦截运算

怎么把一个普�?Python 函数 `f` 转成图？核心思路�?*不给它真数据，给它一�?�?数据（Proxy），看它�?Proxy 做什么操作，每个操作记录成一�?Node**�?

```python
def f(x):
    return (x + 1) * 2

# 追踪过程�?
x = Proxy("x")        # 不传�?Tensor，传 Proxy
# f 内部执行 x + 1�?
#   Proxy.__add__(1) 被调�?
#   它不真的加，而是创建一�?Node(op=add, args=[x, 1])，返回新 Proxy 包这�?Node
a = x + 1             # a �?Proxy(Node(add, [x, 1]))
# f 内部执行 a * 2�?
b = a * 2             # b �?Proxy(Node(mul, [a, 2]))
# f 返回 b，追踪器看到返回的是 Proxy，把它标记为 output
```

追踪结束后，我们手里有一�?Node：`[placeholder x, add, mul, output]`，这就是图�?

**关键�?*：Proxy 重载了所有运算符（`__add__`、`__mul__`、`__matmul__`...），每次运算不真算，只记录。这�?*运算符重载拦�?*，是符号追踪的经典技巧�?

### 13.2.3 `__getattr__` 拦截方法调用

`x + 1` 能被 `__add__` 拦截，但 `x.sum()` 呢？这是方法调用，不是运算符�?

解法：Proxy 重载 `__getattr__`。当访问 `x.sum` 时，`__getattr__("sum")` 被调用，它返回一�?*闭包函数**，这个函数被调用时记录一�?`call_method` Node�?

```python
def __getattr__(self, name):
    def method(*args, **kwargs):
        return self._record("call_method", name, (self, *args), kwargs)
    return method

# x.sum(dim=0) 触发�?
#   __getattr__("sum") 返回 method 闭包
#   method(dim=0) 被调�?�?记录 Node(call_method, "sum", args=[x], kwargs={"dim":0})
```

这招很巧妙：�?Python 的属性访�?+ 闭包，把任意方法调用都拦截下来。代价是 Proxy 不能有真实属性（任何 `proxy.foo` 都会被当成方法），所以内部状态用下划线开头并�?`__getattr__` 里拒绝：

```python
def __getattr__(self, name):
    if name.startswith("_"):
        raise AttributeError(name)   # 内部属性走 __getattribute__ 默认机制
    ...
```

### 13.2.4 Node 的四�?op

追踪下来的每�?Node 有一�?`op` 字段，标记它是什么类型：

| op               | 含义                          | 例子                          |
| ---------------- | ----------------------------- | ----------------------------- |
| `placeholder`    | 输入占位�?                     | 函数的参�?`x`                |
| `call_function`  | 调用一个自由函�?               | `add(x, 1)`、`mul(a, 2)`      |
| `call_method`    | 调用某对象的方法               | `x.sum(dim=0)`                |
| `output`         | 输出节点，标记函数返回什�?     | `return b`                    |

`placeholder` �?`output` 是图的边界（输入和输出），中间的 `call_function` / `call_method` 是计算节点�?

每个 Node 还持有：
- `name`：唯一名字（如 `add_0`），用于 codegen �?env 查找�?
- `target`：要调用的函�?方法名。`call_function` 时是函数对象本身，`call_method` 时是方法名字符串�?
- `args` / `kwargs`：调用的参数。参数可以是别的 Node（表示依赖）、也可以是常量（�?`1`、`2`）�?
- `users`：依赖这�?Node 的下�?Node 集合（反向边，做图变换时用）�?

### 13.2.5 Graph �?codegen：把图变回代�?

有了图，可以把它**生成�?Python 源码**�?

```python
def forward(self, x):
    add_1 = _add(x, 1)
    mul_2 = _mul(add_1, 2)
    return mul_2
```

为什么有用？
1. **可读�?*：用户能直接看到追踪出来的模型长啥样，调试方便�?
2. **可序列化**：源码可以存盘，跨版本加载�?
3. **可执�?*：把源码 `exec` 进一个新模块，就得到一个等价的 forward�?
4. **教学**：清楚展�?�?= 一串赋值语�?�?

codegen 的规则就是遍�?Node，按 op 类型生成对应语法�?
- `placeholder` �?`x = x`（参数直接用�?
- `call_function` �?`name = target(args)`
- `call_method` �?`name = obj.method(args)`
- `output` �?`return result`

### 13.2.6 GraphModule：解释执行图

�?codegen 也能执行图—�?*直接遍历 Node 算一�?*�?

```python
def forward(self, *inputs):
    env = {}                          # Node.name �?计算出的�?
    for node in self.graph.nodes:
        if node.op == "placeholder":
            env[node.name] = inputs[对应下标]
        elif node.op == "call_function":
            args = [env[a.name] if isinstance(a, Node) else a for a in node.args]
            env[node.name] = node.target(*args)
        elif node.op == "call_method":
            obj = env[node.args[0].name]
            env[node.name] = getattr(obj, node.target)(*args)
        elif node.op == "output":
            return env[node.args[0].name]
```

这就�?*图解释器**：按拓扑序一个个算，中间结果�?`env` 字典。比 codegen �?exec 慢一点（多了字典查找），但实现简单、好调试�?

### 13.2.7 算子融合 pass：模式匹�?+ 图重�?

有了图就能做变换�?*算子融合**：把相邻�?`add` + `mul` 合成一�?`fused_add_mul`�?

```
融合�?                      融合�?
x �?add(x,1) �?mul(_,2) �?out   x �?fused_add_mul(x,1,2) �?out
3 个计算节�?                 2 个计算节�?
```

好处：少一�?kernel launch（GPU 上每个算子启动有几十微秒开销）、少一次中间结果访存�?

pass 的算法：
1. **模式匹配**：扫一遍图，找所�?mul 的输入是 add"的模式，记下要融合的 (add, mul) 对�?
2. **图重�?*：建一个新 Graph，遍历旧 Node�?
   - 跳过被融合掉�?add（它已经进到 fused 里）�?
   - 碰到 mul 且它是融合对的下游，生成 `fused_add_mul` 节点替代�?
   - 其他 Node 原样复制，但 args 里的 Node 引用要重映射到新图�?

这是最经典�?**graph transformation pass** 范式：匹�?+ 重建。LLVM、GCC、TVM、XLA 都是这个套路�?

### 13.2.8 FX 的根本局限：data-dependent control flow

```python
def f(x):
    if x > 0:          # �?这个 if 依赖 x 的�?
        return x + 1
    else:
        return x - 1
```

追踪�?`x` �?Proxy，`x > 0` 产生另一�?Proxy，但 Python �?`if` 要一�?bool，Proxy 不是 bool——追踪直接挂�?

这叫 **data-dependent control flow**：分支取决于运行时数据。FX 没法处理，因为它�?*符号**执行，没有真值�?

解法（部分）�?
- �?`if` 改成 `torch.where(cond, x+1, x-1)`，用算子表达分支�?
- 或者用 `torch.fx.wrap` 标记某些函数不追踪，运行时再算�?
- 或者干脆用 `torch.compile`（Dynamo），它能处理一部分 control flow（通过 guard 切分图）�?

### 13.2.9 torch.compile (Dynamo + Inductor) 对比

`torch.compile` �?PyTorch 2.0 引入的，�?FX 强得多：

| 维度            | FX                          | torch.compile                          |
| --------------- | --------------------------- | -------------------------------------- |
| 追踪方式        | Proxy 重载运算�?           | 改写 Python 字节码（Dynamo�?          |
| control flow    | 不支�?data-dependent       | 支持（通过 guard 切分多个图）          |
| 副作�?         | 不支持（print、in-place�?  | 支持（标记为 side-effect，不优化�?    |
| 动�?shape      | 假设静�?                   | 支持动�?shape（重新追踪）             |
| 后端            | 解释执行 / codegen Python   | Inductor 生成 Triton/C++ kernel        |
| 性能            | 中等（少 launch�?          | 强（kernel 融合 + 自动向量化）         |
| 用�?           | 模型变换、量化、剪�?       | 训练加�?                               |

Dynamo 的核心招数：**字节码改�?*。它不靠 Proxy，而是直接修改 Python 字节码，在执行时拦截每个算子调用。遇�?`if x > 0` 时，它记�?�?x>0 时走这个�?，并设一�?guard：下�?x �?>0 就复用图，否则重新追踪。这样既保留了图优化的好处，又支持了 control flow�?

FX �?静态图"路线，Dynamo �?动态图 + guard"路线。后者更通用但更复杂�?

---

## 13.3 设计决策与权�?

| 决策                          | 我们的选择                              | 理由                                            | 代价                                       |
| --------------------------- | ---------------------------------- | --------------------------------------------- | ---------------------------------------- |
| 追踪机制                       | Proxy 重载运算�?+ `__getattr__`    | 实现简单，�?Python，无字节码魔�?                   | 不支�?data-dependent control flow         |
| Node op 分类                  | 4 类（placeholder/call_function/call_method/output�?| 与真�?fx 一致，覆盖常见场景                  | 没有 `call_module` / `get_attr`，不能追踪嵌�?Module |
| Node 名字生成                  | `{op}_{序号}`                       | 简单且唯一                                      | 不语义化（真�?fx �?`add`、`mul` 等可读名�?  |
| users 维护                    | 创建 Node 时反向更�?args �?users   | 图变换时要查下游                                  | args 改了要手动维护（教学版没做）            |
| codegen 范围                  | 单函�?forward，单输入 x             | 教学简�?                                       | 多输入、多输出、嵌套模块都不支�?             |
| GraphModule 执行              | 解释执行（env 字典�?               | 实现简单、好调试                                  | 比编译执行慢（字典查找开销�?               |
| 融合 pass 策略                | 单趟线性扫描，匹配 add→mul          | 实现极简，讲清范�?                               | 只融合一种模式，不级联、不重排              |
| 融合后图重建                   | 新建 Graph，按�?Node 顺序复制      | 保持拓扑序，安全                                  | 不做死代码消除、公共子表达式提�?           |
| 不支�?in-place               | 没特殊处�?                         | 教学简�?                                       | `x += 1` 追踪结果可能不对                  |
| 不支�?control flow           | 直接报错或行为未定义                | FX 本来就不支持                                  | 用户�?`if` 会得到奇怪结�?               |

---

## 13.4 代码逐行实现

### 13.4.1 `graph.py`：Node �?Graph

```python
"""Graph：计算图数据结构（Ch13）�?

Node 持有 op/target/args/kwargs/name/users�?
Graph 持有 Node 序列，可 codegen forward�?
对应真实 PyTorch �?fx/graph.py（简化为 call_function/call_method）�?
"""

from __future__ import annotations

from collections.abc import Callable


class Node:
    def __init__(
        self,
        name: str,
        op: str,
        target: Callable | str,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        self.name = name
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.users: set[Node] = set()      # 谁依赖我（反向边�?
        self._update_users()

    def _update_users(self):
        # 遍历 args/kwargs，遇�?Node 就把自己加到它的 users �?
        # 这维护了"反向�?，图变换时查下游 O(1)
        for arg in self.args:
            if isinstance(arg, Node):
                arg.users.add(self)
        for arg in self.kwargs.values():
            if isinstance(arg, Node):
                arg.users.add(self)

    def __repr__(self):
        return f"Node({self.name}, op={self.op})"


class Graph:
    def __init__(self):
        self.nodes: list[Node] = []

    def create_node(self, op: str, target, args=(), kwargs=None) -> Node:
        # 自动生成唯一名字：{op}_{当前节点数}
        # 因为节点数单调增，序号唯一
        name = f"{op}_{len(self.nodes)}"
        node = Node(name, op, target, args, kwargs)
        self.nodes.append(node)
        return node

    def placeholder(self, name: str = "x") -> Node:
        # placeholder 是输入占位符，名字用用户给的（如 "x0", "x1"�?
        node = self.create_node("placeholder", name)
        node.name = name       # 覆盖自动名，�?codegen �?"x" 而不�?"placeholder_0"
        return node

    def call_function(self, target: Callable, args=(), kwargs=None) -> Node:
        return self.create_node("call_function", target, args, kwargs)

    def call_method(self, method: str, args=(), kwargs=None) -> Node:
        return self.create_node("call_method", method, args, kwargs)

    def output(self, result: Node) -> Node:
        # output 节点�?args 是要返回�?Node（或 Node 元组�?
        return self.create_node("output", "output", (result,))

    def codegen(self) -> str:
        # 生成 Python 源码：def forward(self, x): ...
        lines = ["def forward(self, x):"]
        for node in self.nodes:
            if node.op == "placeholder":
                # placeholder 直接绑定到输入参�?
                lines.append(f"    {node.name} = x")
            elif node.op == "call_function":
                # name = target(args, kwargs)
                args_str = self._format_args(node.args)
                kwargs_str = self._format_kwargs(node.kwargs)
                # getattr(target, "__name__", str) 拿函数名，codegen 出可读的 _add 而不�?<function>
                target_name = getattr(node.target, "__name__", str(node.target))
                lines.append(f"    {node.name} = {target_name}({args_str}{kwargs_str})")
            elif node.op == "call_method":
                # name = obj.method(args)
                method = node.target
                # args[0] 是对象本身，args[1:] 是方法参�?
                args_str = self._format_args(node.args[1:])
                lines.append(f"    {node.name} = {node.args[0].name}.{method}({args_str})")
            elif node.op == "output":
                lines.append(f"    return {node.args[0].name}")
        return "\n".join(lines)

    def _format_args(self, args) -> str:
        parts = []
        for arg in args:
            parts.append(self._format_value(arg))
        return ", ".join(parts)

    def _format_kwargs(self, kwargs: dict) -> str:
        if not kwargs:
            return ""
        parts = []
        for k, v in kwargs.items():
            parts.append(f"{k}={self._format_value(v)}")
        return ", " + ", ".join(parts) if parts else ""

    def _format_value(self, val) -> str:
        # Node �?它的名字；常�?�?repr（如 1 �?"1"�?
        if isinstance(val, Node):
            return val.name
        return repr(val)

    def __repr__(self):
        return f"Graph({len(self.nodes)} nodes)"
```

**逐行要点�?*

- `Node._update_users` �?`__init__` 末尾调，保证 users 始终最新。代价是 args 后续不能改（改了 users 不会自动更新），教学版接受这个限制�?
- `Graph.create_node` �?`len(self.nodes)` 当序号，因为节点只增不删，序号唯一�?
- `placeholder` 覆盖自动名，�?codegen 输出 `x = x` 而不�?`placeholder_0 = x`，可读性更好�?
- `codegen` �?`call_method` �?`args[0]` 是对象本身（�?`x`），`args[1:]` 是方法参数。所以生�?`add_1 = x.sum(dim=0)` 而不�?`add_1 = sum(x, dim=0)`�?
- `_format_value` �?Node 用名字、对常量�?`repr`，这�?`1` 生成 `1`，`"hello"` 生成 `'hello'`�?

### 13.4.2 `proxy.py`：追踪代�?

```python
"""Proxy：符号追踪代理对象（Ch13）�?

�?Proxy 替换输入，拦截所有运算记录为 Graph Node�?
对应真实 PyTorch �?fx/proxy.py�?
"""

from __future__ import annotations

from .graph import Graph, Node


class Proxy:
    def __init__(self, node: Node, graph: Graph):
        self.node = node
        self.graph = graph

    def _record(self, op: str, target, args, kwargs=None) -> Proxy:
        # �?args 里的 Proxy 替换成它包的 Node（图里存 Node 引用，不�?Proxy�?
        proxy_args = tuple(a.node if isinstance(a, Proxy) else a for a in args)
        proxy_kwargs = {
            k: v.node if isinstance(v, Proxy) else v for k, v in (kwargs or {}).items()
        }
        # �?graph 里建一个新 Node
        node = self.graph.create_node(op, target, proxy_args, proxy_kwargs)
        # 返回包这�?Node 的新 Proxy，让链式调用能继�?
        return Proxy(node, self.graph)

    # ── 运算符重载：每个都记录一�?call_function Node ──
    def __add__(self, other):
        return self._record("call_function", _add, (self, other))

    def __mul__(self, other):
        return self._record("call_function", _mul, (self, other))

    def __sub__(self, other):
        return self._record("call_function", _sub, (self, other))

    def __truediv__(self, other):
        return self._record("call_function", _div, (self, other))

    def __matmul__(self, other):
        return self._record("call_function", _matmul, (self, other))

    def __neg__(self):
        return self._record("call_function", _neg, (self,))

    def __getattr__(self, name: str):
        # 内部属性（_node、_graph 等）走默认机制，不拦�?
        # 注意：__getattr__ 只在正常查找失败时才调用，所�?__init__ 里设�?self.node 不会触发
        if name.startswith("_"):
            raise AttributeError(name)

        # 返回一个闭包，调用时记�?call_method
        # 这让 proxy.sum(dim=0) �?记录 Node(call_method, "sum", args=[proxy], kwargs={"dim":0})
        def method(*args, **kwargs):
            return self._record("call_method", name, (self, *args), kwargs)

        return method

    def __repr__(self):
        return f"Proxy({self.node})"


# 模块级自由函数，作为 Node.target �?
# 之所以单独定义而不是用 operator.add，是为了 codegen �?__name__ 是可读的 "_add"
def _add(a, b):
    return a + b


def _mul(a, b):
    return a * b


def _sub(a, b):
    return a - b


def _div(a, b):
    return a / b


def _matmul(a, b):
    return a @ b


def _neg(a):
    return -a
```

**逐行要点�?*

- `_record` 是核心：�?Proxy 参数解包�?Node、建 Node、返回新 Proxy。所有运算符都委托给它�?
- `proxy_args` �?`a.node if isinstance(a, Proxy) else a`：如果参数是 Proxy 就取它的 Node（图里存 Node），如果是常量（�?`1`）就原样存。这让图能区�?依赖另一个节�?�?常量参数"�?
- `__getattr__` �?`if name.startswith("_"): raise AttributeError` 是关键防御：内部属性访问（�?`self.node`）不能被拦截。注�?`__getattr__` 只在正常查找失败时调用，所�?`__init__` �?`self.node = node` 走的�?`__setattr__`，之后访�?`self.node` �?`__getattribute__`（命中实例字典），不会触�?`__getattr__`。但访问不存在的 `_foo` 会触发，这时�?raise �?Python 知道这不是方法�?
- 闭包 `method` 捕获 `self` �?`name`，调用时记录。这�?`x.sum(dim=0)`、`x.reshape(2,3)` 等都能被追踪�?
- `_add` 等模块级函数�?`__name__`，codegen 时能输出 `_add(x, 1)` 而不�?`<lambda>(x, 1)`�?

### 13.4.3 `tracer.py`：符号追踪入�?

```python
"""tracer：符号追踪器（Ch13）�?

symbolic_trace(func) �?Proxy 替换输入运行 func，记录运算为 Graph Node�?
对应真实 PyTorch �?fx/_symbolic_trace.py�?
"""

from __future__ import annotations

from collections.abc import Callable

from .graph import Graph
from .proxy import Proxy


def symbolic_trace(func: Callable, n_inputs: int = 1) -> Graph:
    graph = Graph()
    # 为每个输入创建一�?placeholder Node + 包它�?Proxy
    proxies = [Proxy(graph.placeholder(f"x{i}"), graph) for i in range(n_inputs)]
    # �?Proxy 当真 Tensor 喂给 func —�?func 内部运算会被 Proxy 拦截记录
    result = func(*proxies)
    # func 返回的可能是单个 Proxy �?tuple/list of Proxy
    if isinstance(result, Proxy):
        graph.output(result.node)
    elif isinstance(result, tuple | list):
        # 多输出：output 节点�?args 是所有输�?Node
        output_node = graph.create_node("output", "output", tuple(r.node for r in result))
        output_node.name = "output"
    return graph
```

**逐行要点�?*

- `symbolic_trace` 不执行真计算，它"假装"执行：用 Proxy 当输入，func 内部所有运算都被记录�?
- `n_inputs` 支持多输入函数，每个输入一�?placeholder�?
- 返回值处理分两种：单 Proxy �?`graph.output(node)`；tuple/list �?多输出节点。教学版没处�?返回常量"�?返回 None"的情况�?

### 13.4.4 `graph_module.py`：可执行图模�?

```python
"""GraphModule：可执行图模块（Ch13）�?

持有 Graph，forward 时解释执行每�?Node�?
对应真实 PyTorch �?fx/graph_module.py�?
"""

from __future__ import annotations

from .graph import Graph, Node


def _resolve(arg, env: dict[str, object]):
    # 如果 arg �?Node，从 env 取它已算出的值；否则是常量，原样返回
    if isinstance(arg, Node):
        return env[arg.name]
    return arg


class GraphModule:
    def __init__(self, graph: Graph):
        self.graph = graph

    def forward(self, *inputs):
        # env: Node.name �?计算出的�?
        # 按拓扑序（即 nodes 列表顺序）逐个�?
        env: dict[str, object] = {}
        for node in self.graph.nodes:
            if node.op == "placeholder":
                # placeholder：从 inputs 取对应下�?
                # 名字约定 "x0", "x1" ...，下标从名字第二个字符起解析
                idx = int(node.name[1:]) if len(node.name) > 1 else 0
                env[node.name] = inputs[idx]
            elif node.op == "call_function":
                # 解析所�?args/kwargs（把 Node 替换�?env 里的值）
                args = [_resolve(a, env) for a in node.args]
                kwargs = {k: _resolve(v, env) for k, v in node.kwargs.items()}
                # 调用真实函数
                env[node.name] = node.target(*args, **kwargs)
            elif node.op == "call_method":
                method = node.target
                # args[0] 是对象本�?
                obj = _resolve(node.args[0], env)
                args = [_resolve(a, env) for a in node.args[1:]]
                kwargs = {k: _resolve(v, env) for k, v in node.kwargs.items()}
                env[node.name] = getattr(obj, method)(*args, **kwargs)
            elif node.op == "output":
                # output：返回结�?
                return _resolve(node.args[0], env)
        return None

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def code(self) -> str:
        # 便利方法：返�?codegen 出的源码
        return self.graph.codegen()
```

**逐行要点�?*

- `env` 字典是图解释器的核心数据结构：存每个 Node 算出的值，按名字查找�?
- `placeholder` 的下标从名字解析（`"x0" �?0`）。这依赖 tracer �?placeholder 命名约定，比较脆弱——真�?fx �?`node.args` 存下标信息�?
- `call_function` �?`call_method` 都先 `_resolve` 所有参数（�?Node 引用换成 env 里的值），再调用真实函数/方法�?
- `output` 直接返回，不�?env�?
- `__call__` 委托�?`forward`，让 GraphModule 像普通函数一样可调用�?

### 13.4.5 `passes/fusion.py`：算子融�?

```python
"""fusion：算子融�?pass（Ch13）�?

相邻算子合并（如 add+mul），减少 kernel launch 与访存�?
融合前后数值不变，节点数减少�?
对应真实 PyTorch �?fx/passes/fusion.py�?
"""

from __future__ import annotations

from ..graph import Graph, Node
from ..proxy import _add, _mul


def _fused_add_mul(a, b, c):
    # 融合后的算子�?a + b) * c
    return (a + b) * c


def fuse_add_mul(graph: Graph) -> Graph:
    """�?x = add(a, b); y = mul(x, c) 融合�?y = fused_add_mul(a, b, c)�?""
    # 第一步：模式匹配，找出所有要融合�?(add, mul) �?
    fuse_pairs: dict[str, str] = {}   # add_name �?mul_name
    skip_nodes: set[str] = set()      # 要跳过的 add（已融进 mul�?

    for node in graph.nodes:
        # 找一�?mul 节点
        if node.op == "call_function" and node.target is _mul:
            prev_node = node.args[0]
            # 它的第一个参数是一�?add 节点
            if (
                isinstance(prev_node, Node)
                and prev_node.op == "call_function"
                and prev_node.target is _add
            ):
                fuse_pairs[prev_node.name] = node.name
                skip_nodes.add(prev_node.name)

    # 第二步：图重�?
    new_graph = Graph()
    node_map: dict[str, Node] = {}    # �?Node.name �?�?Node

    for node in graph.nodes:
        # 被融合的 add 跳过（不复制到新图）
        if node.name in skip_nodes:
            continue
        if node.op == "placeholder":
            new_node = new_graph.placeholder(node.name)
            node_map[node.name] = new_node
        elif node.op == "call_function" and node.target is _mul and node.name in fuse_pairs.values():
            # 这个 mul 是融合对的下�?�?生成 fused_add_mul
            # 找到它对应的 add
            prev_name = next(k for k, v in fuse_pairs.items() if v == node.name)
            prev_node = next(n for n in graph.nodes if n.name == prev_name)
            # add 的两个参�?+ mul 的第二个参数
            add_a = _remap_arg(prev_node.args[0], node_map)
            add_b = _remap_arg(prev_node.args[1], node_map)
            mul_c = _remap_arg(node.args[1], node_map)
            new_node = new_graph.call_function(_fused_add_mul, (add_a, add_b, mul_c))
            new_node.name = f"fused_{prev_name}_{node.name}"
            # mul 的输出映射到�?fused 节点（下游引�?mul 的会找到 fused�?
            node_map[node.name] = new_node
        elif node.op == "output":
            mapped = _remap_arg(node.args[0], node_map)
            new_graph.output(mapped)
        else:
            # 其他节点原样复制，args 里的 Node 引用重映�?
            new_node = _copy_node(new_graph, node, node_map)
            node_map[node.name] = new_node

    return new_graph


def _remap_arg(arg, node_map: dict[str, Node]):
    # 把旧 Node 引用换成�?Node
    if isinstance(arg, Node):
        return node_map.get(arg.name, arg)
    return arg


def _copy_node(new_graph: Graph, node: Node, node_map: dict[str, Node]) -> Node:
    # 复制节点到新图，args/kwargs 里的 Node 重映�?
    args = tuple(_remap_arg(a, node_map) for a in node.args)
    kwargs = {k: _remap_arg(v, node_map) for k, v in node.kwargs.items()}
    new_node = new_graph.create_node(node.op, node.target, args, kwargs)
    new_node.name = node.name
    return new_node
```

**逐行要点�?*

- 融合分两步：**匹配**（扫一遍找模式）和**重建**（按拓扑序建新图）。这是所有图变换 pass 的标准范式�?
- `fuse_pairs` �?dict 而不�?list，因为要�?add 名字快速查 mul 名字，按 mul 名字快速查 add 名字�?
- `skip_nodes` 记下要跳过的 add——它已经被融�?fused，不该单独出现在新图�?
- 重建�?`node_map` 维护"�?Node �?�?Node"映射。任何下游引用旧 Node 的，都要换成�?Node（`_remap_arg`）�?
- 融合节点名字 `fused_{add_name}_{mul_name}` 方便调试�?
- `_fused_add_mul` 是融合后的算子，数值上等于 `(a+b)*c`，但�?launch 一�?kernel�?

---

## 13.5 完整示例

```python
import numpy as np
from minitorch import Tensor
from minitorch.fx import GraphModule, symbolic_trace
from minitorch.fx.passes.fusion import fuse_add_mul

# ── 1. 追踪一个简单函�?─────────────────────────────────
def f(x):
    return (x + 1) * 2

graph = symbolic_trace(f, n_inputs=1)
print("节点�?", len(graph.nodes))
for n in graph.nodes:
    print(" ", n)

# ── 2. codegen 看生成的源码 ────────────────────────────
print("\n--- codegen ---")
print(graph.codegen())

# ── 3. GraphModule 执行，验证等�?───────────────────────
gm = GraphModule(graph)
x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
expected = f(x)
actual = gm(x)
print("\nexpected:", expected.numpy())
print("actual:  ", actual.numpy())
assert np.allclose(actual.numpy(), expected.numpy())

# ── 4. matmul 也能追踪 ─────────────────────────────────
def g(x):
    return x @ x + 1

graph_g = symbolic_trace(g, n_inputs=1)
gm_g = GraphModule(graph_g)
x2 = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
assert np.allclose(gm_g(x2).numpy(), g(x2).numpy())
print("\nmatmul trace OK")

# ── 5. 算子融合 ────────────────────────────────────────
print("\n--- fusion ---")
fused_graph = fuse_add_mul(graph)
print("原节点数:", len(graph.nodes))
print("融合后节点数:", len(fused_graph.nodes))
print(fused_graph.codegen())

gm_fused = GraphModule(fused_graph)
assert np.allclose(gm_fused(x).numpy(), gm(x).numpy())
print("fusion preserves values: OK")

# ── 6. 多输入追�?──────────────────────────────────────
def h(x0, x1):
    return x0 + x1

graph_h = symbolic_trace(h, n_inputs=2)
gm_h = GraphModule(graph_h)
a = Tensor.from_numpy(np.array([1.0, 2.0]))
b = Tensor.from_numpy(np.array([3.0, 4.0]))
print("\nmulti-input result:", gm_h(a, b).numpy())
```

预期输出（节选）�?

```
节点�? 4
  Node(x, op=placeholder)
  Node(call_function_1, op=call_function)
  Node(call_function_2, op=call_function)
  Node(call_function_3, op=output)

--- codegen ---
def forward(self, x):
    x = x
    call_function_1 = _add(x, 1)
    call_function_2 = _mul(call_function_1, 2)
    return call_function_2

expected: [4. 6. 8.]
actual:   [4. 6. 8.]

--- fusion ---
原节点数: 4
融合后节点数: 3
def forward(self, x):
    x = x
    fused_call_function_1_call_function_2 = _fused_add_mul(x, 1, 2)
    return fused_call_function_1_call_function_2
fusion preserves values: OK

multi-input result: [4. 6.]
```

---

## 13.6 常见陷阱

### 陷阱 1：在追踪的函数里�?`if x > 0`

```python
def f(x):
    if x > 0:        # �?x �?Proxy，x > 0 �?Proxy，if �?bool �?�?
        return x
    return x + 1
```

Proxy 没重�?`__bool__`，Python 会调 `__bool__` 失败后报错或行为未定义�?

**解决**：改写成 `torch.where`，或者把分支移到追踪外（按固定模式追踪）�?

### 陷阱 2：在追踪的函数里�?`print`

```python
def f(x):
    print(x)         # �?�?print 一�?Proxy 对象
    return x + 1
```

不会挂，�?print 出来�?`Proxy(Node(x, op=placeholder))`，不是真值。副作用也没被记录进图�?

**解决**：追踪时别放 print；要调试�?`gm.code()` 看生成的源码�?

### 陷阱 3：in-place 操作追踪结果�?

```python
def f(x):
    x += 1           # �?__iadd__ 可能没重载，或重载后行为不对
    return x * 2
```

教学�?Proxy 没重�?`__iadd__`，Python fallback �?`__add__` 再赋值，但赋值改的是局部变量，原图�?placeholder 没变�?

**解决**：永远用 `x = x + 1` 而不�?`x += 1`�?

### 陷阱 4：融�?pass 后忘验证数值等�?

融合是图变换，可能引�?bug。不验证就上线，模型悄悄算错�?

**解决**：每�?pass 后跑一组随机输入，`assert allclose(原结�? 新结�?`。测试里 `test_fusion_preserves_values` 就是干这个的�?

### 陷阱 5：以�?FX 能加�?

FX 本身**不加�?*——GraphModule 解释执行比原函数还慢一点（多了 env 字典查找）。FX 的价值是**让图可变�?*，加速来自后续的 codegen + 编译（如 Inductor 生成 Triton kernel）�?

**解决**：要加速用 `torch.compile`；FX 用于变换（量化、剪枝、融合准备）�?

### 陷阱 6：追踪嵌�?Module 没处�?

```python
model = Sequential(Linear(4, 8), Linear(8, 1))
symbolic_trace(model)   # �?教学版不支持，因�?Linear.forward 没被 Proxy �?
```

教学�?`symbolic_trace` 只接受纯函数，不接受 nn.Module。真�?fx �?`fx.symbolic_trace(model)` 能处�?Module（通过 `call_module` op）�?

**解决**：把 model.forward 包成纯函数再追踪，或用真�?PyTorch �?fx�?

### 陷阱 7：节点名字冲�?

融合 pass 里新节点�?`fused_{add}_{mul}`，如果原图已有同名节点会冲突。教学版没检查�?

**解决**：用全局计数器生成唯一名，或检查冲突时加后缀�?

---

## 13.7 与真�?PyTorch 对照

| minitorch                              | torch.fx                                  | 关键差异                                                     |
| -------------------------------------- | ----------------------------------------- | -------------------------------------------------------- |
| `Node` 4 �?op                          | 6 种（�?`call_module`、`get_attr`�?      | 真实版能追踪嵌套 Module 和模块属性；教学版只追踪纯函�?           |
| `Node.name` 自动 `{op}_{序号}`            | 语义化（`add`、`mul`、`linear1_weight`�?  | 真实版可读性好；教学版够用但丑                                    |
| `Graph.codegen` 单输入单输出              | 多输入多输出、支�?`self` 属性访�?           | 教学版简�?                                                 |
| `GraphModule.forward` 解释执行           | 同（也有 codegen + exec 模式�?            | 一致；真实版默�?codegen �?exec，更�?                         |
| `Proxy` 重载 6 个运算符 + `__getattr__`   | 重载全部 + 处理 Tensor 方法、numpy 互操�?   | 教学版覆盖核心；真实版处理几十种算子                                |
| `symbolic_trace(func, n_inputs)`       | `symbolic_trace(module)` / `Tracer().trace()` | 教学版只追踪函数；真实版追踪 Module                          |
| `fuse_add_mul` 单模式单�?               | 多模式、多趟、级联融�?                     | 教学版讲范式；真实版融合几十种模�?                               |
| �?                                     | `Graph.erase_node` / `lint` / `to_dot`   | 真实版有图编辑、可视化、合法性检�?                               |
| �?                                     | `torch.compile` (Dynamo + Inductor)      | 字节码追踪、guard 切分、Triton codegen、动�?shape           |

!!! tip "torch.compile 为什么更�?"
Dynamo 不靠 Proxy，而是**改写 Python 字节�?*：在执行时拦截每�?`CALL` 指令，把算子记录进图。遇�?`if x > 0` 时，它记录当�?x 的值作�?**guard**，生成一�?�?x>0 时用这个�?的特化版本。下�?x �?>0 就复用，否则重新追踪。这样既保留了图优化，又支持�?control flow 和动�?shape。代价是实现极其复杂（要理解 CPython 字节码、帧栈、guard 逻辑）�?

---

## 13.8 历史背景

**图与编译在深度学习框架里的演化：**

- **2014~2016（TF1.x / Theano�?*：静态图为主。模型先编译成图再执行，优化强但难调试�?define then run"�?
- **2016~2018（PyTorch 0.x�?*：命令式（eager）逆袭�?define by run"，每行立刻执行，好调试。代价是没有图优化�?
- **2019（PyTorch 1.3，torch.fx 实验�?*：在 eager 之上�?事后追踪"——模型写完再转成图。FX 诞生，用于变换（量化、剪枝）。不追求训练加速，追求可操纵性�?
- **2021（torch.fx 稳定�?*：API 定型，`symbolic_trace` / `GraphModule` / `Node` 成为标准。广泛用于量化（�?PyTorch 1.8 的量�?API 基于 fx）�?
- **2022（torch.compile 预览�?*：PyTorch 2.0 预告 Dynamo + Inductor，目标是"零代码改动加速训�?。字节码追踪 + Triton codegen�?
- **2023（PyTorch 2.0 正式�?*：`torch.compile` GA，多数模型加一�?`model = torch.compile(model)` 就快 30%+。FX 仍用于变换场景�?
- **未来**：Inductor 越来越强（自�?kernel 融合、shape 特化、分布式规划）；FX 作为"图中间表�?仍是底层基础之一�?

minitorch 这套 fx 实现对应 PyTorch 1.9 前后�?fx 核心，去掉了 `call_module` / `get_attr` 和图编辑 API，保留追�?+ 解释执行 + �?pass 融合的最小完整闭环�?

---

## 13.9 练习�?

### 练习 1：让 Proxy 支持 `__pow__`

修改 Proxy，让 `x ** 2` 能被追踪�?`call_function` 节点�?

??? 解答
    ```python
    def _pow(a, b):
        return a ** b

    class Proxy:
        ...
        def __pow__(self, other):
            return self._record("call_function", _pow, (self, other))
    ```
    并在 `proxy.py` 模块级定�?`_pow`。这�?`x ** 2` 追踪�?`Node(call_function, _pow, args=[x, 2])`�?
???

### 练习 2：实�?`fuse_mul_add`（mul→add 融合�?

仿照 `fuse_add_mul`，把 `y = mul(a, b); z = add(y, c)` 融合�?`z = fused_mul_add(a, b, c) = a*b + c`�?

??? 解答
    ```python
    def _fused_mul_add(a, b, c):
        return a * b + c

    def fuse_mul_add(graph):
        fuse_pairs = {}
        skip_nodes = set()
        for node in graph.nodes:
            if node.op == "call_function" and node.target is _add:
                prev = node.args[0]
                if (isinstance(prev, Node)
                        and prev.op == "call_function"
                        and prev.target is _mul):
                    fuse_pairs[prev.name] = node.name
                    skip_nodes.add(prev.name)
        # 重建逻辑�?fuse_add_mul 对称，把 mul 跳过，add 换成 fused_mul_add
        # ...（结构与 fuse_add_mul 相同，只�?_mul/_add 角色互换�?
    ```
???

### 练习 3：解释为什�?`__getattr__` 里要拒绝下划线开头的名字

??? 解答
    Proxy 内部状态（`self.node`、`self.graph`）存在实�?`__dict__` 里。访�?`self.node` �?Python 先走 `__getattribute__`（命�?`__dict__`），不会触发 `__getattr__`。但访问**不存�?*�?`_foo` 时，`__getattribute__` 失败，回退�?`__getattr__`。如果不拒绝，`__getattr__` 会返回一个闭包，�?`_foo` 当方法记录进图——这�?bug。拒绝（raise AttributeError）让 Python 知道这不是属性，正常报错�?
???

### 练习 4：追�?`x.sum()` 看图长什么样

写出 `symbolic_trace(lambda x: x.sum(), 1)` 得到�?Graph 节点序列�?

??? 解答
    ```python
    graph = symbolic_trace(lambda x: x.sum(), 1)
    # nodes:
    #   Node(x, op=placeholder)
    #   Node(call_method_1, op=call_method, target="sum", args=(Node(x),))
    #   Node(call_method_2, op=output, args=(Node(call_method_1),))
    # codegen:
    #   def forward(self, x):
    #       x = x
    #       call_method_1 = x.sum()
    #       return call_method_1
    ```
    关键：`x.sum` 触发 `__getattr__("sum")` 返回闭包，闭包被调用记录 `call_method`�?
???

### 练习 5：为什�?FX 不能处理 `if x.sum() > 0` �?torch.compile �?

??? 解答
    FX �?Proxy 追踪。`x.sum()` 返回 Proxy，`Proxy > 0` 返回另一�?Proxy，`if Proxy` �?bool——Proxy 没有真值，无法决定走哪个分支，追踪挂�?
    
    torch.compile (Dynamo) 不用 Proxy，而是改写字节码在真执行时拦截。遇�?`if x.sum() > 0` 时，�?*真的�?* `x.sum() > 0` 得到 bool，按真值走分支，同时记�?�?x.sum()>0 时走这个分支"作为 guard。下次输入若 guard 满足就复用图，否则重新追踪。这样既支持�?control flow，又保留了图优化（在每个 guard 分支内）�?
    
    本质区别：FX �?*符号**执行（没真值），Dynamo �?*�?guard 的真**执行（有真值但记录条件）�?
???

---

## 13.10 关键测试解读

`tests/test_fx.py`�?

```python
def test_trace_node_count_matches_calls():
    def f(x):
        return (x + 1) * 2
    graph = symbolic_trace(f, n_inputs=1)
    assert len(graph.nodes) == 4
    assert graph.nodes[0].op == "placeholder"
    assert graph.nodes[1].op == "call_function"
    assert graph.nodes[2].op == "call_function"
    assert graph.nodes[3].op == "output"
```

**解读**：`(x+1)*2` �?2 个算子调用（add、mul），�?placeholder �?output �?4 个节点。验证追�?*正确计数**——如果多记或漏记，节点数不对�?

```python
def test_graph_module_forward_equivalent():
    def f(x):
        return (x + 1) * 2
    graph = symbolic_trace(f, n_inputs=1)
    gm = GraphModule(graph)
    x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    expected = f(x)
    actual = gm(x)
    assert np.allclose(actual.numpy(), expected.numpy())
```

**解读**�?*等价性测�?*——追踪后再执行，结果应该和直接执行原函数一致。这�?fx 最核心的不变量：`GraphModule(symbolic_trace(f))(x) == f(x)`。如果不等，说明追踪或解释执行有 bug�?

```python
def test_fusion_preserves_values():
    def f(x):
        return (x + 1) * 2
    graph = symbolic_trace(f, n_inputs=1)
    fused_graph = fuse_add_mul(graph)
    assert len(fused_graph.nodes) < len(graph.nodes)
    gm_orig = GraphModule(graph)
    gm_fused = GraphModule(fused_graph)
    x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(gm_orig(x).numpy(), gm_fused(x).numpy())
```

**解读**：融�?pass 的两个不变量�?*节点数减�?*（融合真的合并了算子�? **数值等�?*（融合没改变语义）。前者验�?pass 有效，后者验�?pass 正确�?

```python
def test_fusion_reduces_node_count():
    ...
    assert fused_count == original_count - 1
```

**解读**：精确断言"�?1 个节�?。`add + mul` 融合�?1 �?`fused_add_mul`，少�?1 个计算节点（placeholder �?output 不变）。如果融�?pass 误删了别的节点，这个精确断言会挂�?

```python
def test_multi_input_trace():
    def f(x0, x1):
        return x0 + x1
    graph = symbolic_trace(f, n_inputs=2)
    gm = GraphModule(graph)
    a = Tensor.from_numpy(np.array([1.0, 2.0]))
    b = Tensor.from_numpy(np.array([3.0, 4.0]))
    result = gm(a, b)
    assert np.allclose(result.numpy(), [4.0, 6.0])
```

**解读**：多输入追踪。验�?placeholder 命名（`x0`、`x1`）和 GraphModule 下标解析（`int("x0"[1:]) == 0`）正确对应。如果下标解析错（如都取 0），结果会变�?`a + a` 而不�?`a + b`�?

---

## 13.11 优劣势总结

**优势�?*

- **极简核心**：Proxy + Graph + GraphModule 三件套，�?250 行代码讲清符号追踪全流程�?
- **运算�?+ 方法都支�?*：`__getattr__` 闭包技巧让 `x.sum()`、`x.reshape()` 都能追踪�?
- **codegen 可读**：能生成 Python 源码，调试和教学友好�?
- **融合 pass 讲清范式**：匹�?+ 重建是所有图变换的通用模式，掌握后能看�?LLVM/TVM �?pass�?
- **与真�?fx API 一�?*：`symbolic_trace` / `GraphModule` / `Node.op` 命名和语义都对齐，迁移成本低�?

**劣势�?*

- **不支持嵌�?Module**：没�?`call_module` op，不能追�?`nn.Sequential`�?
- **不支持模块属�?*：没�?`get_attr` op，不能追�?`self.weight`�?
- **不支�?control flow**：`if` / `for` 依赖数据时挂�?
- **不支�?in-place**：`+=`、`*=` 行为可能错�?
- **融合 pass 单一**：只融合 add→mul 一种模式，不级联、不重排�?
- **解释执行�?*：env 字典查找开销，比 codegen + exec 慢�?
- **节点名不语义�?*：`call_function_1` 不如真实 fx �?`add` 可读�?

**教学价�?*：这套实现把"符号追踪"这件听起来很玄的事讲透了——本质就�?�?Proxy 拦截运算、记�?Node"。理解了�?250 行，去看 PyTorch fx 几千行代码就不会迷路，再去看 Dynamo 字节码追踪就有了"图是怎么来的"这个基础�?

---

## 13.12 下一章预�?

本章我们实现了静态图追踪。但 FX 追踪出来后是**解释执行**，没真正编译。要真正加速，需要：

- 把图 lowering 成更低的中间表示（如 Triton IR）�?
- 自动 kernel 融合 + 向量�?+ 内存规划�?
- 处理动�?shape �?guard 切分�?

这就�?`torch.compile` (Dynamo + Inductor) 干的事，超出�?minitorch 的范围。如果继续这个系列，下一章会讲：

- Dynamo 的字节码追踪原理�?
- Guard 切分如何支持 control flow�?
- Inductor �?Triton codegen�?
- 自动 kernel 融合的调度算法�?

那是真正�?编译�?领域，比 fx 深一个量级。本章是通往那里的第一级台阶�?
