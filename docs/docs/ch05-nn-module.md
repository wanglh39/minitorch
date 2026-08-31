# 第五章 nn.Module 体系：参数注册、递归遍历与容器设计

> 前四章我们解决了"张量怎么存"、"算子怎么分发"、"自动微分怎么算"、"计算图怎么管"。
> 但一个真正的神经网络有几十上百个参数，靠用户手写 `w1, b1, w2, b2, ...` 管理是不现实的。
> 本章讲 minitorch 怎么用 `nn.Module` 这一个基类，通过 `__setattr__` 拦截、三大注册表、
> 递归遍历，把"一堆参数"组织成"一个可训练的整体"。这是 PyTorch 最具工程美感的设计之一。

---

## 5.1 本章目标

读完本章后，你应当能够：

1. 解释 `Module.__setattr__` 为什么能在你写 `self.weight = Parameter(...)` 时自动把参数注册到 `_parameters` 字典，而普通赋值走 `object.__setattr__`。
2. 画出 `__getattr__` 的查找顺序：`__dict__` → `_parameters` → `_modules` → `_buffers` → 抛 `AttributeError`，并说明为什么 `__getattr__` 只在正常查找失败时才被调用。
3. 区分 `_parameters`、`_modules`、`_buffers` 三大注册表的用途，说出 `BatchNorm` 的 `running_mean` 应该放进哪一个。
4. 手写一个递归 `parameters()` 生成器，正确处理嵌套子模块，并解释为什么用 `yield from` 而不是 `return list`。
5. 描述 `state_dict()` 的键命名规则（`fc1.weight`、`fc1.fc2.bias`），并实现一个 `load_state_dict` 把外部权重灌回模型。
6. 解释 `train()` / `eval()` 为什么必须递归调用子模块，以及 `Dropout` 和 `BatchNorm` 为什么依赖 `self.training` 标志。
7. 用 `Sequential` 和 `ModuleList` 拼出一个多层感知机，并说出两者在 `forward` 上的差别。
8. 写一个 `register_forward_hook`，在 `Linear` 前向前后插入打印逻辑，理解 hook 是调试和可视化（如 activation hook）的基础。

---

## 5.2 原理铺垫：从一个朴素的需求说起

### 5.2.1 朴素写法的问题

假设没有 `nn.Module`，你要写一个两层 MLP：

```python
class MLP:
    def __init__(self):
        self.w1 = np.random.randn(4, 8)
        self.b1 = np.zeros(8)
        self.w2 = np.random.randn(8, 2)
        self.b2 = np.zeros(2)

    def forward(self, x):
        h = np.maximum(0, x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2
```

看起来没问题。但现在你要：

- **收集所有参数交给优化器**：得手写 `params = [self.w1, self.b1, self.w2, self.b2]`，每加一层就改一次。
- **保存模型**：得手写 `{"w1": self.w1, "b1": self.b1, ...}`，键名容易写错。
- **加载模型**：得反向一个个赋值。
- **切换 train/eval**：如果有 `Dropout`，得给每个子层单独调 `dropout.train()`。
- **嵌套**：如果 `MLP` 里套一个 `Block`，`Block` 里又套 `Linear`，参数收集要递归。

这些都是**机械重复**的工作。`nn.Module` 的目标就是把这些自动化。

### 5.2.2 核心思路：拦截赋值，自动注册

Python 对象的属性都存在 `self.__dict__` 里。`nn.Module` 的核心 trick 是：

> **重写 `__setattr__`，在赋值时判断值的类型，如果是 `Parameter` 就塞进 `_parameters` 字典，如果是 `Module` 就塞进 `_modules` 字典，否则走默认行为。**

这样用户写 `self.weight = Parameter(...)` 时，`weight` 不会进 `self.__dict__`，而是进 `self._parameters["weight"]`。读取时再靠 `__getattr__` 从 `_parameters` 取出来。

为什么不让参数直接进 `__dict__`？因为要**区分类型**：

- `Parameter`：要被优化器收集、要进 `state_dict`、要算梯度。
- `Module`：要递归遍历其参数、要递归切 train/eval。
- 普通属性（如 `self.training`）：什么都不做。

如果全塞 `__dict__`，就没法区分，得在每次遍历时 `isinstance` 判断全量扫描——慢且易错。

### 5.2.3 Python 属性访问机制回顾

要理解这套 trick，必须先搞清楚 Python 的属性访问协议：

```
读取 self.name：
  1. 先查 type(self).__dict__ 里的数据描述符（有 __get__ 和 __set__）
  2. 再查 self.__dict__（实例字典）
  3. 再查 type(self).__dict__ 里的非数据描述符/普通类属性
  4. 都没找到 → 调用 __getattr__(name)（如果定义了）

写入 self.name = value：
  1. 先查 type(self).__dict__ 里的数据描述符（有 __set__）
  2. 否则直接写 self.__dict__[name] = value
  3. 但如果重写了 __setattr__，则全部走 __setattr__
```

关键点：

!!! tip "`__getattr__` vs `__getattribute__`"
    - `__getattribute__`：**每次**属性访问都调用，性能开销大，容易递归爆栈。
    - `__getattr__`：**只在正常查找失败时**才调用，安全且高效。

    minitorch 用 `__getattr__`。所以 `self.training` 这种在 `__dict__` 里的属性，**不会**触发 `__getattr__`，直接走快速路径。只有 `self.weight`（不在 `__dict__`，在 `_parameters`）才会触发 `__getattr__`。

### 5.2.4 MRO（Method Resolution Order）与 `super()`

`Module.__setattr__` 里有一行 `super().__setattr__(name, value)`。这里的 `super()` 在 Python 3 里是动态的，等价于：

```python
super(type(self), self).__setattr__(name, value)
```

但实际更精确：它按 **MRO**（方法解析顺序）找到当前类的下一个类。对于 `Module` 单继承：

```
Module → object
```

所以 `super().__setattr__` 就是 `object.__setattr__`，即默认行为：写 `self.__dict__[name] = value`。

MRO 在多继承时才复杂（C3 线性化）。minitorch 的 `Module` 是单继承，MRO 就是 `[Module, object]`。但理解 MRO 对后续读 PyTorch 源码（`Linear(Module)`、`BatchNorm(_NormBase)`、`_NormBase(Module)`）有帮助。

```python
class Module:
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            ...  # 进 _parameters
        elif isinstance(value, Module):
            ...  # 进 _modules
        else:
            super().__setattr__(name, value)  # ← 走 object.__setattr__，进 __dict__
```

### 5.2.5 三大注册表的职责

```
self._parameters: dict[str, Parameter]   # 可学习参数（weight、bias）
self._modules:    dict[str, Module]      # 子模块（fc1、fc2、layer1）
self._buffers:    dict[str, Tensor]      # 不可学习但需保存的状态（running_mean、running_var）
```

| 注册表       | 谁进去                  | 进 state_dict | 进 parameters() | requires_grad | 典型例子                |
| ------------ | ----------------------- | ------------- | --------------- | ------------- | ----------------------- |
| `_parameters`| `Parameter` 实例        | 是            | 是              | 通常 True     | `Linear.weight`         |
| `_modules`   | `Module` 实例           | 递归子模块    | 递归子模块      | -             | `MLP.fc1`               |
| `_buffers`   | `Tensor`（手动 register）| 是            | 否              | 通常 False    | `BatchNorm.running_mean`|

`_buffers` 在 minitorch 里只声明了字典和 `register_buffer` 接口，本章末尾会讨论它的设计。`BatchNorm` 的完整实现留到后续章节。

### 5.2.6 递归遍历：为什么用生成器

`parameters()` 要收集所有嵌套子模块的参数。两种写法：

```python
# 写法 A：列表
def parameters(self):
    result = list(self._parameters.values())
    for m in self._modules.values():
        result.extend(m.parameters())
    return iter(result)

# 写法 B：生成器（minitorch 采用）
def parameters(self):
    yield from self._parameters.values()
    for m in self._modules.values():
        yield from m.parameters()
```

生成器的优势：

1. **惰性**：不一次性物化整个列表。大模型参数几亿个，列表本身也是内存。
2. **可组合**：`yield from` 自然处理嵌套递归，代码简洁。
3. **语义清晰**：`parameters()` 返回的是"迭代器"，暗示"遍历一次就完"，符合 PyTorch 习惯。

代价：生成器只能迭代一次。如果用户要多次遍历，得 `list(model.parameters())`。PyTorch 也是这样。

### 5.2.7 train/eval 的递归传播

很多层的行为依赖模式：

- `Dropout`：train 时随机置零，eval 时恒等。
- `BatchNorm`：train 时用 batch 统计并更新 running_mean，eval 时用 running_mean。
- `Linear`：不依赖模式，但仍要递归（因为可能套在 `Sequential` 里）。

所以 `train()` 必须递归：

```python
def train(self, mode=True):
    self.training = mode
    for m in self._modules.values():
        m.train(mode)   # ← 递归
    return self
```

返回 `self` 是为了链式调用：`model.train().to(device)`（虽然 minitorch 没有 `.to`）。

### 5.2.8 容器：Sequential vs ModuleList

`Sequential`：按顺序执行，前一个的输出是后一个的输入。

```python
net = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
out = net(x)   # 等价于 Linear(8,2)(ReLU()(Linear(4,8)(x)))
```

`ModuleList`：只是把一组模块存起来，**不**自动 forward。用户自己决定怎么调。

```python
layers = ModuleList([Linear(4, 8), Linear(8, 2)])
out = layers[1](layers[0](x))   # 用户自己写执行顺序
```

两者的 `__init__` 都用 `setattr(self, str(i), m)` 把模块注册成 `"0"`、`"1"`、`"2"`... 这样的属性名。这样 `__setattr__` 会自动把它们塞进 `_modules`，参数收集就能递归到。

### 5.2.9 hooks：在 forward 前后插入逻辑

```python
m.register_forward_pre_hook(lambda mod, inp: print("before", inp))
m.register_forward_hook(lambda mod, inp, out: print("after", out))
m(x)
# 输出：
# before (x,)
# after <forward 结果>
```

hooks 存在 `_forward_pre_hooks` 和 `_forward_hooks` 字典里，键是 `id(hook)`（避免重复注册同一个函数）。`__call__` 在调 `forward` 前后遍历这些字典调一遍。

hooks 的用途：

- **调试**：打印每层输入输出形状。
- **可视化**：取中间激活做特征图可视化。
- **梯度剪裁**：在 backward hook 里剪裁梯度（minitorch 暂未实现 backward hook）。
- **forward hook 写 feature**：CNN 可视化里取某一层输出。

---

## 5.3 设计决策与权衡

| 决策                                | 选择                              | 理由                                                | 代价                                              |
| ----------------------------------- | --------------------------------- | --------------------------------------------------- | ------------------------------------------------- |
| 拦截赋值的方式                      | 重写 `__setattr__`                | 用户写法最自然：`self.w = Parameter(...)`           | 所有赋值都过 `__setattr__`，有微小开销           |
| 区分参数/模块/普通属性              | `isinstance` 判断                 | 类型即语义，无需额外标记                            | `Parameter` 必须显式包装，不能直接传 Tensor       |
| 注册表数据结构                      | 三个 `dict`                       | O(1) 查找，键名即属性名                            | 三个 dict 占内存（但相对参数本身可忽略）         |
| `__getattr__` 而非 `__getattribute__`| 只在查找失败时触发               | 不影响 `self.training` 等快速路径                   | 需保证 `_parameters` 等本身在 `__dict__` 里       |
| `parameters()` 返回类型             | 生成器                            | 惰性、省内存、可递归                                | 只能迭代一次，多次用要 `list()`                  |
| `state_dict` 键命名                 | `prefix + name`，递归加 `.`       | 唯一且可读，匹配 PyTorch                            | 嵌套深时键名长（`block1.layer3.fc2.weight`）     |
| `train()` 返回值                    | 返回 `self`                       | 支持链式 `model.train().to(device)`                 | 容易误以为返回新对象，实际是 in-place             |
| `Sequential` 模块命名               | 用 `str(i)`（"0"、"1"）           | 简单、有序                                          | 不能用名字索引（PyTorch 也这样，要 `OrderedDict`）|
| hooks 的 key                        | `id(hook)`                        | 同一函数不重复注册                                  | lambda 每次新建 id 不同，会重复注册              |
| `__call__` 调 `forward`             | 不直接调 `forward`，走 `__call__` | hooks 只在 `__call__` 里                            | 直接 `model.forward(x)` 会绕过 hooks             |
| `_buffers` 是否自动注册             | 手动 `register_buffer`            | 区分普通 Tensor 和 buffer                           | 用户要记得调 `register_buffer`                   |
| `__repr__` 实现                     | 列出参数和子模块                  | 调试友好                                            | 大模型 repr 很长                                  |

---

## 5.4 代码逐行实现

### 5.4.1 `parameter.py`：Parameter 类

```python
class Parameter(Tensor):
    def __init__(self, data, requires_grad: bool = True):
        if isinstance(data, Tensor):
            # 已经是 Tensor：复用其 storage/shape/strides，强制 requires_grad=True
            super().__init__(
                data.storage, data.shape, data.strides, data.storage_offset, requires_grad=True
            )
        else:
            # 是 list/ndarray：先转 ndarray，再建 Storage
            arr = np.asarray(data)
            storage = Storage.from_numpy(arr)
            super().__init__(
                storage, arr.shape, _compute_contiguous_strides(arr.shape), 0, requires_grad=True
            )
        self.requires_grad = requires_grad   # ← 允许用户传 False 冻结参数
```

逐行解读：

- **`class Parameter(Tensor)`**：继承 Tensor，所以 `Parameter` 既能参与所有张量运算，又能被 `isinstance` 识别。
- **`if isinstance(data, Tensor)`**：支持两种构造方式：`Parameter(some_tensor)` 或 `Parameter([[1,2],[3,4]])`。前者复用底层 Storage（零拷贝），后者新建。
- **`requires_grad=True` 默认**：参数默认要算梯度。这与普通 Tensor 默认 `False` 形成对比——这是 `Parameter` 存在的核心意义。
- **`_compute_contiguous_strides`**：从 shape 算出连续布局的 strides，详见第一章。
- **最后一行 `self.requires_grad = requires_grad`**：允许 `Parameter(data, requires_grad=False)` 冻结参数（迁移学习常用）。

!!! warning "为什么不直接用 Tensor？"
    如果直接 `self.weight = Tensor(...)`，`__setattr__` 不会把它放进 `_parameters`（因为不是 `Parameter` 实例），优化器就收集不到。`Parameter` 的**类型本身**就是信号。这是"类型即语义"的典型应用。

### 5.4.2 `module.py`：`__init__`

```python
class Module:
    def __init__(self):
        self._parameters: dict[str, Parameter] = {}   # ① 参数注册表
        self._modules: dict[str, Module] = {}         # ② 子模块注册表
        self._buffers: dict[str, Tensor] = {}         # ③ 状态注册表
        self.training: bool = True                    # ④ train/eval 标志
        self._forward_pre_hooks: dict[int, Callable] = {}  # ⑤ 前 hook
        self._forward_hooks: dict[int, Callable] = {}      # ⑥ 后 hook
```

注意一个微妙点：这些赋值**会触发 `__setattr__`**。但此时 `_parameters` 等还不存在（`__dict__` 里没有），所以 `__setattr__` 里的 `isinstance(value, Parameter)` 判断：`{}` 不是 `Parameter`，不是 `Module`，走 `super().__setattr__`，正常写进 `__dict__`。**初始化顺序很重要**：必须先建 `_parameters` 等空字典，后续赋值才能往里塞。

如果反过来先 `self.weight = Parameter(...)`，此时 `__setattr__` 里 `self.__dict__.setdefault("_parameters", {})` 会自动建一个空字典再塞——所以 `setdefault` 的设计就是为了应对这种顺序问题。这是 PyTorch 源码里也有的防御性写法。

### 5.4.3 `module.py`：`__setattr__` 拦截

```python
def __setattr__(self, name: str, value: Any) -> None:
    if isinstance(value, Parameter):
        # 是参数 → 塞 _parameters，不进 __dict__
        self.__dict__.setdefault("_parameters", {})[name] = value
    elif isinstance(value, Module):
        # 是子模块 → 塞 _modules
        self.__dict__.setdefault("_modules", {})[name] = value
    else:
        # 普通值 → 走默认，进 __dict__
        super().__setattr__(name, value)
```

逐行解读：

- **`isinstance(value, Parameter)`**：因为 `Parameter` 继承 `Tensor`，所以 `isinstance(param, Tensor)` 也是 True。但这里先判 `Parameter`，所以参数不会走 `Tensor` 分支。**判断顺序很重要**：先具体后一般。
- **`self.__dict__.setdefault("_parameters", {})`**：直接操作 `__dict__` 而非 `self._parameters`，是为了**避免递归**。如果写 `self._parameters[name] = value`，会再次触发 `__setattr__`（给 `_parameters` 赋值？不，是读 `self._parameters`，读会触发 `__getattr__`...）。直接操作 `__dict__` 绕过属性协议。
- **`setdefault`**：如果 `_parameters` 不存在就建空字典。这处理了"用户在 `super().__init__()` 之前就赋值参数"的边界情况。
- **`super().__setattr__`**：走 `object.__setattr__`，即 `self.__dict__[name] = value`。

!!! tip "为什么不直接 `self.__dict__[name] = value`？"
    等价。但用 `super().__setattr__` 更"正统"，未来如果 MRO 中间插入了别的 `__setattr__`（如某个 mixin），能正确转发。这是防御性编程。

### 5.4.4 `module.py`：`__getattr__` 反查

```python
def __getattr__(self, name: str) -> Any:
    # ⚠️ 只在 self.__dict__ 没找到 name 时才被调用
    params = self.__dict__.get("_parameters", {})
    if name in params:
        return params[name]
    modules = self.__dict__.get("_modules", {})
    if name in modules:
        return modules[name]
    buffers = self.__dict__.get("_buffers", {})
    if name in buffers:
        return buffers[name]
    raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
```

逐行解读：

- **`__getattr__` 只在正常查找失败时调用**：所以 `self.training`（在 `__dict__`）不会进这里，直接走快速路径。这是性能关键。
- **`self.__dict__.get("_parameters", {})`**：用 `.get` 而非 `self._parameters`，避免递归。如果 `_parameters` 还没初始化（`__init__` 没调），返回 `{}` 而非抛错。
- **查找顺序**：`_parameters` → `_modules` → `_buffers`。实际中名字不会冲突（用户不会同时 `self.x = Parameter()` 和 `self.x = Module()`），但顺序定义了优先级。
- **抛 `AttributeError`**：必须抛，否则 Python 内部很多机制（如 `hasattr`、pickle）会失效。

!!! warning "一个经典坑"
    在 `__getattr__` 里写 `self._parameters` 会**无限递归**：`self._parameters` 触发 `__getattr__("_parameters")`，里面又 `self._parameters`... 所以必须用 `self.__dict__.get("_parameters")`。这是写 `__getattr__` 的铁律。

### 5.4.5 `module.py`：`__call__` 与 hooks

```python
def __call__(self, *args, **kwargs) -> Any:
    for hook in self._forward_pre_hooks.values():
        hook(self, args)                          # 前 hook：可改输入（这里不改）
    result = self.forward(*args, **kwargs)        # 真正的前向
    for hook in self._forward_hooks.values():
        hook(self, args, result)                  # 后 hook：可观察输出
    return result
```

逐行解读：

- **`__call__` 而非直接 `forward`**：用户写 `model(x)` 触发 `__call__`，`__call__` 调 `forward`。这样 hooks 有统一入口。如果用户直接 `model.forward(x)`，会绕过 hooks——所以文档都说"别直接调 forward"。
- **`hook(self, args)`**：签名是 `(module, input)`。input 是 tuple（因为 `*args`）。
- **`hook(self, args, result)`**：签名是 `(module, input, output)`。
- **遍历 `.values()`**：hooks 存在 dict 里，键是 `id(hook)`。遍历 values 不关心键。

### 5.4.6 `module.py`：递归 `parameters()` 与 `named_parameters()`

```python
def parameters(self) -> Iterator[Parameter]:
    yield from self._parameters.values()          # 自己的直接参数
    for m in self._modules.values():              # 遍历子模块
        yield from m.parameters()                 # 递归子模块的参数

def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
    for name, p in self._parameters.items():
        yield prefix + name, p                    # 自己的参数，加前缀
    for name, m in self._modules.items():
        yield from m.named_parameters(prefix + name + ".")  # 递归，前缀加 "子模块名."
```

逐行解读：

- **`yield from`**：把一个可迭代对象的每个元素逐个 yield 出来，等价于 `for x in iter: yield x`，但更高效（CPython 优化）。
- **递归终止**：叶子模块（如 `Linear`）的 `_modules` 为空，循环不进，自然终止。
- **`prefix` 累积**：`named_parameters` 在递归时把当前模块名加进前缀。`MLP.fc1.weight` 就是这么来的：`MLP` 调 `fc1.named_parameters("fc1.")`，`fc1` yield `("fc1.weight", w)`。

### 5.4.7 `module.py`：`state_dict` 与 `load_state_dict`

```python
def state_dict(self, prefix: str = "") -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, p in self._parameters.items():
        state[prefix + name] = p                  # 参数进 state
    for name, b in self._buffers.items():
        state[prefix + name] = b                  # buffer 也进 state
    for name, m in self._modules.items():
        state.update(m.state_dict(prefix + name + "."))  # 递归子模块
    return state

def load_state_dict(self, state_dict: dict[str, Tensor], prefix: str = "") -> None:
    for name, p in self._parameters.items():
        key = prefix + name
        if key in state_dict:
            src = state_dict[key]
            p._storage._data[:] = src._numpy_view().ravel()  # in-place 拷贝数据
    for name, m in self._modules.items():
        m.load_state_dict(state_dict, prefix + name + ".")    # 递归
```

逐行解读：

- **`state_dict` 包含参数和 buffer**：但**不**包含子模块结构。所以加载时模型结构必须匹配。
- **`load_state_dict` 是 in-place**：直接改 `p._storage._data`，不替换 `Parameter` 对象。这样优化器里对 `p` 的引用仍然有效。
- **`src._numpy_view().ravel()`**：把源张量展平成一维 numpy 视图，再 `[:]` 拷贝到目标 storage。`ravel` 处理形状不一致（实际应一致，但防御性）。
- **不报错如果 key 不在 state_dict**：`if key in state_dict` 静默跳过。PyTorch 会报 `missing keys`，minitorch 简化了。

### 5.4.8 `module.py`：`train` / `eval` / `zero_grad`

```python
def train(self, mode: bool = True) -> Module:
    self.training = mode
    for m in self._modules.values():
        m.train(mode)                             # 递归子模块
    return self                                   # 返回 self 支持链式

def eval(self) -> Module:
    return self.train(False)                      # eval 就是 train(False)

def zero_grad(self) -> None:
    for p in self.parameters():
        p.grad = None                             # 设 None 而非 0，省内存且符合 PyTorch 语义
```

逐行解读：

- **`train` 递归**：必须递归，否则子模块的 `Dropout` 不知道要切模式。
- **`return self`**：链式调用 `model.train().to(device)`。
- **`zero_grad` 设 `None`**：PyTorch 0.4+ 改成 `None` 而非零张量，原因是：① 省内存（不存零张量）；② 优化器能区分"这一步没算梯度"和"梯度是零"。minitorch 跟随这一设计。

### 5.4.9 `module.py`：hooks 注册

```python
def register_forward_pre_hook(self, hook: Callable) -> None:
    self._forward_pre_hooks[id(hook)] = hook      # 用 id(hook) 做 key

def register_forward_hook(self, hook: Callable) -> None:
    self._forward_hooks[id(hook)] = hook
```

逐行解读：

- **`id(hook)` 做 key**：避免重复注册同一个函数对象。`id` 返回内存地址，同一对象 `id` 相同，后注册会覆盖前注册。
- **lambda 每次新建 id 不同**：所以 `m.register_forward_hook(lambda ...)` 多次调用会注册多个不同 hook。这是已知行为，用户应注意。

### 5.4.10 `containers.py`：Sequential 与 ModuleList

```python
class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, str(i), m)              # 把模块注册成属性 "0"、"1"...

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)                              # 顺序执行，前一个输出是后一个输入
        return x


class ModuleList(Module):
    def __init__(self, modules: list[Module] | None = None):
        super().__init__()
        if modules is not None:
            for i, m in enumerate(modules):
                setattr(self, str(i), m)          # 同样注册成 "0"、"1"...

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)                              # 也提供默认 forward（顺序执行）
        return x
```

逐行解读：

- **`setattr(self, str(i), m)`**：等价于 `self."0" = m`（Python 不允许直接写 `self.0 = m`，但 `setattr` 可以）。这触发 `Module.__setattr__`，把 `m` 塞进 `_modules["0"]`。
- **`Sequential.forward` 顺序执行**：这是 Sequential 的语义。如果用户想要分支/跳连，得自己写 `Module` 子类。
- **`ModuleList` 有 `__len__` 和 `__iter__`**：可以 `len(ml)`、`for layer in ml`、`ml[0]`（后者 minitorch 未实现，PyTorch 通过 `__getitem__` 实现）。
- **`ModuleList.forward` 也顺序执行**：这是 minitorch 的简化。PyTorch 的 `ModuleList` **没有** `forward`，调用会报错——因为 ModuleList 的语义是"存储"而非"执行"。minitorch 加了默认 forward 是为了简化教学。

!!! warning "与 PyTorch 的差异"
    PyTorch 的 `ModuleList` 没有 `forward`，调 `ml(x)` 会 `NotImplementedError`。minitorch 给了默认顺序执行，这是教学简化。生产代码里 `ModuleList` 通常用在 `Module` 子类的 `forward` 里手动控制执行顺序。

### 5.4.11 `linear.py`：Linear 层

```python
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        bound = 1.0 / (in_features**0.5)          # Kaiming uniform 的简化版
        w = np.random.uniform(-bound, bound, (out_features, in_features))
        self.weight = Parameter(Tensor.from_numpy(w))   # ← 触发 __setattr__，进 _parameters
        if bias:
            b = np.random.uniform(-bound, bound, (out_features,))
            self.bias = Parameter(Tensor.from_numpy(b))
        else:
            self.bias = None                      # None 不触发注册

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose()         # x @ W^T，形状 (N, in) @ (in, out) = (N, out)
        if self.bias is not None:
            out = out + self.bias                 # 广播加 bias
        return out
```

逐行解读：

- **`bound = 1 / sqrt(in_features)`**：这是 PyTorch `Linear` 的默认初始化（Kaiming uniform 的一个特例）。让权重方差 ≈ `1/in`，保证前向方差稳定。
- **`weight` 形状 `(out, in)`**：注意是 `(out, in)` 不是 `(in, out)`。所以前向要 `x @ weight.T`。这是 PyTorch 的约定，历史原因。
- **`self.weight = Parameter(...)`**：这一行触发 `Module.__setattr__`，把 `weight` 塞进 `_parameters["weight"]`。用户后续 `self.weight` 读取时，`__getattr__` 从 `_parameters` 取出。
- **`self.bias = None`**：`None` 不是 `Parameter` 也不是 `Module`，走 `super().__setattr__`，进 `__dict__`。所以 `self.bias` 读取时直接从 `__dict__` 取，不触发 `__getattr__`。
- **`forward` 里 `self.weight`**：触发 `__getattr__`（因为 weight 不在 `__dict__`，在 `_parameters`），取出 `Parameter`。`Parameter` 是 `Tensor` 子类，能参与 `@` 和 `transpose`。

### 5.4.12 `module.py`：`__repr__` 调试输出

```python
def __repr__(self) -> str:
    lines = [f"{type(self).__name__("]                # ① 类名开头
    for name, p in self._parameters.items():
        lines.append(f"  {name}: {p}")                # ② 缩进列参数
    for name, m in self._modules.items():
        lines.append(f"  {name}: {m}")                # ③ 缩进列子模块（递归 repr）
    lines.append(")")
    return "\n".join(lines)
```

逐行解读：

- **`type(self).__name__`**：取类名字符串，避免硬编码。子类自动用子类名。
- **两空格缩进**：`  {name}:` 让嵌套结构可读。PyTorch 的 repr 更精细（带 `extra_repr`、参数形状而非值），minitorch 简化。
- **`{m}` 触发子模块的 `__repr__`**：递归。所以 `print(MLP())` 会展开所有嵌套层。
- **只列 `_parameters` 和 `_modules`**：不列 `_buffers` 和普通属性。PyTorch 列 buffer，minitorch 简化。

### 5.4.13 `module.py`：`register_parameter` 与 `register_buffer`

```python
def register_parameter(self, name: str, param: Parameter) -> None:
    self._parameters[name] = param                   # 直接塞，不走 __setattr__

def register_buffer(self, name: str, tensor: Tensor) -> None:
    self._buffers[name] = tensor                     # buffer 不进 _parameters
```

逐行解读：

- **`register_parameter`**：等价于 `setattr(self, name, param)`，但更显式。PyTorch 内部用这个 API 注册参数，避免 `__setattr__` 的隐式魔法。
- **`register_buffer`**：注册 buffer。buffer 不是参数（不进 `parameters()`、不算梯度），但要进 `state_dict`（要保存/加载）。典型例子：`BatchNorm.running_mean`、`running_var`。
- **直接操作 `_buffers` 字典**：不走 `__setattr__`（因为 `__setattr__` 只识别 `Parameter`/`Module`，普通 `Tensor` 会进 `__dict__` 而非 `_buffers`）。所以**必须**用 `register_buffer` 才能注册 buffer。

!!! tip "为什么 buffer 不直接 `self.running_mean = tensor`？"
    如果 `self.running_mean = some_tensor`，`__setattr__` 判断 `some_tensor` 不是 `Parameter` 也不是 `Module`，走默认分支进 `__dict__`。这样 `state_dict` 收集不到它（`state_dict` 只遍历 `_parameters`/`_buffers`/`_modules`），保存模型时会丢。所以 buffer 必须用 `register_buffer` 显式注册到 `_buffers` 字典。

!!! warning "minitorch 的 `state_dict` 漏了 buffer 的加载"
    看 `load_state_dict` 的实现：它只遍历 `_parameters`，**没**遍历 `_buffers`。所以即使 `state_dict` 里有 buffer 的键，`load_state_dict` 也不会灌回 buffer。这是 minitorch 的一个简化缺陷（PyTorch 的 `load_state_dict` 会处理 buffer）。如果要支持 BatchNorm 的完整保存加载，得在 `load_state_dict` 里加一段遍历 `_buffers` 的逻辑。

### 5.4.14 三大注册表的对比总结

把 `_parameters`、`_modules`、`_buffers` 放在一起对比，记住一个口诀：

> **参数要梯度、模块要递归、buffer 要保存。**

| 操作                | `_parameters` | `_modules` | `_buffers` | `__dict__`（普通属性）|
| ------------------- | ------------- | ---------- | ---------- | --------------------- |
| `parameters()` 收集 | 是            | 递归子模块 | 否         | 否                    |
| `state_dict()` 保存 | 是            | 递归子模块 | 是         | 否                    |
| `train()` 递归      | 否            | 是         | 否         | 否                    |
| `zero_grad()` 清    | 是            | 递归子模块 | 否         | 否                    |
| 算梯度              | 是（默认）    | -          | 否（默认） | 否                    |

这张表是本章最值得记住的：它解释了为什么要把不同东西放进不同字典——因为每种操作要"挑选"的对象不同，分开存就不用每次全量扫描判断。

---

## 5.5 完整示例

### 5.5.1 定义并使用一个 MLP

```python
import numpy as np
from minitorch import Tensor
from minitorch.nn import Linear, Module, Sequential

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)      # ← 触发 __setattr__，fc1 进 _modules
        self.fc2 = Linear(8, 2)      # ← fc2 进 _modules

    def forward(self, x):
        return self.fc2(self.fc1(x))

np.random.seed(0)
model = MLP()
x = Tensor.from_numpy(np.random.randn(3, 4))
out = model(x)
print(out.shape)                     # (3, 2)
```

### 5.5.2 查看注册表

```python
print(model._modules)                # {'fc1': Linear(...), 'fc2': Linear(...)}
print(model._parameters)             # {} ← MLP 自己没有直接参数
print(model.fc1._parameters.keys())  # dict_keys(['weight', 'bias'])
```

### 5.5.3 递归收集参数

```python
params = list(model.parameters())
print(len(params))                   # 4：fc1.weight, fc1.bias, fc2.weight, fc2.bias

names = dict(model.named_parameters())
print(names.keys())
# dict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])
```

### 5.5.4 state_dict 保存与加载

```python
sd = model.state_dict()
print(sd.keys())
# dict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])

# 新建一个同结构模型，加载权重
model2 = MLP()
model2.load_state_dict(sd)

# 验证输出一致
x = Tensor.from_numpy(np.random.randn(3, 4))
print(model(x).allclose(model2(x), atol=1e-6))   # True
```

### 5.5.5 train/eval 切换

```python
print(model.training, model.fc1.training, model.fc2.training)   # True True True
model.eval()
print(model.training, model.fc1.training, model.fc2.training)   # False False False
model.train()
print(model.training, model.fc1.training)                        # True True
```

### 5.5.6 Sequential 拼装

```python
net = Sequential(Linear(4, 8), Linear(8, 2))
x = Tensor.from_numpy(np.random.randn(2, 4))
out = net(x)
print(out.shape)                    # (2, 2)
print(len(list(net.parameters())))  # 4
print(net._modules)                 # {'0': Linear(4,8), '1': Linear(8,2)}
```

### 5.5.7 hooks 调试

```python
lin = Linear(3, 2)
calls = []
lin.register_forward_pre_hook(lambda mod, inp: calls.append(("pre", mod, inp)))
lin.register_forward_hook(lambda mod, inp, out: calls.append(("post", mod, out.shape)))

x = Tensor.from_numpy(np.random.randn(1, 3))
lin(x)
for tag, mod, info in calls:
    print(tag, type(mod).__name__, info)
# pre Linear (Tensor 形状 (1, 3),)
# post Linear (1, 2)
```

### 5.5.8 端到端：训练一个 MLP 回归（与第七章呼应）

```python
from minitorch.optim import SGD
from minitorch.nn import MSELoss

np.random.seed(42)
X = np.random.randn(32, 4)
W_true = np.random.randn(4, 1)
Y = X @ W_true + 0.1 * np.random.randn(32, 1)

model = MLP_regression()   # 假设有 4→8→1 的 MLP
opt = SGD(model.parameters(), lr=0.01)
crit = MSELoss()

for epoch in range(100):
    pred = model(Tensor.from_numpy(X))
    loss = crit(pred, Tensor.from_numpy(Y))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 20 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}")
# epoch 0, loss 1.2345
# epoch 20, loss 0.3456
# ...
# epoch 80, loss 0.0123
```

---

## 5.6 常见陷阱

### 陷阱 1：忘记调 `super().__init__()`

```python
class BadMLP(Module):
    def __init__(self):
        # 忘了 super().__init__()
        self.fc1 = Linear(4, 8)
```

**症状**：`AttributeError: 'BadMLP' object has no attribute '_parameters'`。

**原因**：`_parameters` 等字典在 `Module.__init__` 里创建。没调就不存在。虽然 `__setattr__` 用了 `setdefault` 能建 `_parameters`，但 `__getattr__` 里读 `_modules` 会失败。

**解决**：永远在 `__init__` 第一行调 `super().__init__()`。

### 陷阱 2：把 Tensor 当 Parameter

```python
class Wrong(Module):
    def __init__(self):
        super().__init__()
        self.weight = Tensor.from_numpy(np.random.randn(3, 3))   # ← 不是 Parameter!
```

**症状**：`list(model.parameters())` 是空的，优化器没参数可更新。

**原因**：`__setattr__` 判断 `isinstance(value, Parameter)`，普通 `Tensor` 不通过，走默认分支进 `__dict__`，不被注册。

**解决**：`self.weight = Parameter(Tensor.from_numpy(...))`。

### 陷阱 3：在 `__getattr__` 里访问 `self.xxx`

（这是实现者的陷阱，不是用户的。）如果你修改 `__getattr__`，里面写 `self._parameters`，会无限递归。必须用 `self.__dict__.get("_parameters", {})`。

### 陷阱 4：直接调 `forward` 绕过 hooks

```python
model.forward(x)   # ← 不触发 hooks
model(x)           # ← 触发 hooks
```

**解决**：永远用 `model(x)`。

### 陷阱 5：`parameters()` 是生成器，只能迭代一次

```python
params = model.parameters()
print(len(list(params)))   # 4
print(len(list(params)))   # 0 ← 已经耗尽!
```

**解决**：`params = list(model.parameters())` 物化成列表，或每次重新调 `model.parameters()`。

### 陷阱 6：`Sequential` 不能用名字索引

```python
net = Sequential(Linear(4, 8), Linear(8, 2))
net["0"]          # ← minitorch 不支持，会 TypeError
net[0]            # ← minitorch 也不支持（PyTorch 支持 __getitem__）
```

**解决**：用 `net._modules["0"]`（不推荐，访问私有），或改用 `ModuleList` + 自己写 forward。

### 陷阱 7：`bias=None` 后 `self.bias` 是 None 但在 `__dict__` 里

```python
lin = Linear(3, 2, bias=False)
print(lin.bias)                  # None
print("_parameters" in lin.__dict__ and "bias" in lin._parameters)  # False
print("bias" in lin.__dict__)    # True ← None 进了 __dict__
```

这不影响功能，但要知道 `None` 走的是默认 `__setattr__` 路径。

### 陷阱 8：lambda hook 重复注册

```python
for _ in range(3):
    lin.register_forward_hook(lambda m, i, o: None)
print(len(lin._forward_hooks))   # 3 ← 每次 lambda 都是新对象，id 不同
```

**解决**：把 lambda 提出来赋给变量，或用具名函数。

---

## 5.7 与真实 PyTorch 对照

| minitorch                                       | PyTorch                                                | 差异说明                                  |
| ----------------------------------------------- | ------------------------------------------------------ | ----------------------------------------- |
| `Module.__setattr__` 拦截 Parameter/Module      | 同                                                     | 一致                                      |
| `_parameters`/`_modules`/`_buffers` 三表        | 同                                                     | 一致                                      |
| `__getattr__` 反查三表                          | 同                                                     | 一致                                      |
| `parameters()` 返回生成器                       | 同                                                     | 一致                                      |
| `named_parameters(prefix="")`                   | 同                                                     | 一致                                      |
| `state_dict()` 含参数和 buffer                  | 同                                                     | 一致                                      |
| `load_state_dict` 静默跳过缺失 key              | 报 `RuntimeError: missing/unexpected keys`             | minitorch 简化，不严格检查                |
| `train()` 返回 self                             | 同                                                     | 一致                                      |
| `zero_grad()` 设 `grad = None`                  | 同（0.4+）                                             | 一致                                      |
| `register_forward_pre_hook` 用 `id(hook)` 做 key| 用 `RemovableHandle` 对象，返回 handle 可 `remove()`   | minitorch 不能移除单个 hook               |
| `Sequential` 用 `str(i)` 命名                   | 同（PyTorch 1.x 也用 "0"、"1"）                         | 一致                                      |
| `ModuleList` 有默认 `forward`（顺序执行）       | `ModuleList` 无 `forward`，调用报错                    | **差异**：minitorch 教学简化              |
| `ModuleList` 不支持 `__getitem__`               | 支持，返回 `ModuleList` 子集                           | minitorch 未实现                          |
| `Linear` 初始化 `uniform(-1/sqrt(in), 1/sqrt(in))`| 同                                                     | 一致（Kaiming uniform 特例）              |
| `Linear.weight` 形状 `(out, in)`                | 同                                                     | 一致                                      |
| `__repr__` 列出参数和子模块                     | 更精细，递归 repr，带缩进                              | minitorch 简化                            |
| 无 `__getstate__`/`__setstate__`（pickle）      | 实现，支持 `pickle.dump(model)`                        | minitorch 未实现                          |
| 无 `to(device)`/`cuda()`/`cpu()`                | 有                                                     | minitorch 单设备                          |
| 无 `extra_repr`                                 | 有，子类可扩展 repr                                    | minitorch 简化                            |
| 无 `apply(fn)` 递归对子模块调函数               | 有                                                     | minitorch 未实现，但 `train()` 内部用了类似逻辑 |
| 无 backward hooks                               | 有 `register_full_backward_hook` 等                     | minitorch 未实现                          |

### 5.7.1 关键差异详解：`ModuleList.forward`

PyTorch 的 `ModuleList` 故意不实现 `forward`，因为 `ModuleList` 的语义是"参数容器"而非"执行单元"。用户应该在 `Module` 子类的 `forward` 里手动写执行顺序：

```python
class MyNet(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleList([Linear(4, 8), Linear(8, 2)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

minitorch 给 `ModuleList` 加了默认 `forward`（顺序执行），是为了让 `Sequential` 和 `ModuleList` 在简单场景下行为一致，降低初学者认知负担。但**生产代码**应遵循 PyTorch 语义：`ModuleList` 不直接 call。

### 5.7.2 关键差异详解：hooks 的移除

PyTorch 的 `register_forward_hook` 返回一个 `RemovableHandle`，调 `handle.remove()` 能移除该 hook。minitorch 用 `id(hook)` 做 key，没有 handle，只能 `m._forward_hooks.clear()` 全清。这是工程上的简化。

---

## 5.8 历史背景

### 5.8.1 PyTorch 之前：Lua Torch 的 `nn.Module`

PyTorch 的前身是 Lua Torch。Lua Torch 也有 `nn.Module`，但 Lua 没有重写 `__setattr__` 的优雅机制（Lua 的 metatable `__newindex` 类似但语义不同）。参数注册更多靠显式 `self.weight = nn.Parameter(...)` 加上框架内部的注册逻辑。

### 5.8.2 PyTorch 0.1：`__setattr__` trick 的引入

PyTorch 0.1（2017 年初）确立了当前的 `nn.Module` 设计：用 `__setattr__` 拦截 `Parameter` 和 `Module`，自动注册到三个字典。这一设计来自 Soumith Chintala 等人，核心目标是"让用户代码尽可能自然"。

对比 TF1 的 `tf.Variable`：用户必须显式 `tf.get_variable("weight", ...)`，且要在 `tf.variable_scope` 下。PyTorch 的 `self.weight = Parameter(...)` 更接近"写普通 Python 类"的直觉。

### 5.8.3 `zero_grad` 从零张量到 None 的演化

PyTorch 0.3 之前，`zero_grad` 把 `grad` 设成零张量。0.4 改成 `None`。原因：

1. **省内存**：不存零张量。
2. **语义清晰**：`None` 表示"没算梯度"，零张量表示"算了，恰好是零"。优化器能区分。
3. **避免累加脏数据**：`backward` 是累加的，如果 `zero_grad` 设零张量，下次 `backward` 在零基础上累加；如果 `None`，`backward` 第一次直接赋值。后者更安全。

minitorch 跟随 0.4+ 的 `None` 语义。

### 5.8.4 `ModuleList` vs `Sequential` 的分化

早期 PyTorch 只有 `Sequential`。但用户越来越需要"存一组模块但执行顺序自定义"（如 ResNet 的残差块、Transformer 的多头注意力）。于是引入 `ModuleList`（只存不执行）、`ModuleDict`（按名存）、`ParameterList`、`ParameterDict` 等容器。minitorch 只实现了前两个。

### 5.8.5 hooks 的演化

早期 PyTorch 的 hooks 是 `dict` 存，key 是 `id`。后来引入 `RemovableHandle` 让 hook 可移除（避免内存泄漏——hook 闭包捕获大对象时不释放）。再后来（1.10+）引入 `register_full_backward_hook` 统一各种 backward hook 的混乱 API。minitorch 只实现了 forward hooks 的最简版本。

---

## 5.9 练习题

### 练习 1：实现 `ModuleDict`

实现一个 `ModuleDict(Module)`，接受 `dict[str, Module]`，支持按名索引、`__len__`、`__iter__`，且参数能被递归收集。

??? 解答
    ```python
    class ModuleDict(Module):
        def __init__(self, modules: dict[str, Module] | None = None):
            super().__init__()
            if modules is not None:
                for name, m in modules.items():
                    setattr(self, name, m)   # 触发 __setattr__，进 _modules

        def __len__(self):
            return len(self._modules)

        def __iter__(self):
            return iter(self._modules.values())

        def __getitem__(self, name):
            return self._modules[name]
    ```
    关键点：`setattr(self, name, m)` 用名字而非下标，所以 `state_dict` 键会是 `block_a.weight` 而非 `0.weight`。
???

### 练习 2：为什么 `__setattr__` 里用 `self.__dict__.setdefault` 而非 `self._parameters`

用代码演示如果写成 `self._parameters[name] = value` 会出什么问题。

??? 解答
    `self._parameters` 会触发 `__getattr__("_parameters")`。如果 `_parameters` 已在 `__dict__`，`__getattr__` 不被调用，直接从 `__dict__` 取，没问题。但如果 `__init__` 还没执行（`_parameters` 不存在），`__getattr__` 里 `self.__dict__.get("_parameters", {})` 返回 `{}`（一个临时空字典），往里塞 name→value，**但这个临时字典不是 `self.__dict__["_parameters"]`**，赋值丢失。所以必须用 `self.__dict__.setdefault("_parameters", {})` 直接操作 `__dict__`。
???

### 练习 3：实现 `apply(fn)` 递归对子模块调函数

PyTorch 的 `model.apply(init_fn)` 常用于初始化。实现一个。

??? 解答
    ```python
    def apply(self, fn):
        fn(self)
        for m in self._modules.values():
            m.apply(fn)
        return self
    ```
    用法：`model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)`。
???

### 练习 4：`state_dict` 为什么不存模块结构

如果 `state_dict` 只存参数不存结构，加载时怎么保证结构匹配？

??? 解答
    `state_dict` 只存 `{键: 张量}`，不存"模型有几个层、每层多大"。加载时模型结构由用户代码决定：`model = MLP(); model.load_state_dict(sd)`。如果 `MLP` 结构和保存时不一致（如 `fc1` 从 `Linear(4,8)` 变成 `Linear(4,16)`），`load_state_dict` 里 `p._storage._data[:] = src...` 会形状不匹配报错（或静默错误）。PyTorch 会检查 `missing_keys` 和 `unexpected_keys` 报警告。这种设计让"模型结构"和"参数"解耦：结构在代码里，参数在文件里。
???

### 练习 5：hooks 的执行顺序

如果注册了多个 pre_hook，执行顺序是什么？为什么用 `dict` 而非 `list`？

??? 解答
    顺序是 `dict.values()` 的迭代顺序，Python 3.7+ 是**插入顺序**。所以先注册先执行。用 `dict` 而非 `list` 是为了用 `id(hook)` 做 key 去重——同一个函数对象重复注册不会重复执行。代价是 lambda 每次新建对象，`id` 不同，会重复注册（已知行为）。
???

---

## 5.10 关键测试解读

`tests/test_module.py` 的每个测试都在防御一类 bug：

### `test_parameter_requires_grad`

```python
p = Parameter(Tensor.from_numpy(np.array([1.0, 2.0])))
assert p.requires_grad
```

**防御**：`Parameter` 默认 `requires_grad=True`。如果某次重构把 `super().__init__` 的 `requires_grad` 写成 `False`，这个测试会挂。

### `test_module_setattr_register`

```python
m = Linear(3, 2)
assert "weight" in m._parameters
assert "bias" in m._parameters
```

**防御**：`self.weight = Parameter(...)` 确实进了 `_parameters` 而非 `__dict__`。如果 `__setattr__` 的 `isinstance` 判断写反或漏了，参数会进错地方。

### `test_nested_parameters_recursive`

```python
m = _MLP()   # fc1: Linear(4,8), fc2: Linear(8,2)
params = list(m.parameters())
assert len(params) == 4
```

**防御**：递归收集。`MLP` 自己 `_parameters` 是空的，参数全在子模块里。如果 `parameters()` 没递归，会返回 0 个。

### `test_named_parameters_prefix`

```python
names = dict(m.named_parameters())
assert "fc1.weight" in names
assert "fc2.bias" in names
```

**防御**：前缀拼接正确。如果 `prefix + name + "."` 写错（如漏了 `.`），键会是 `fc1weight` 而非 `fc1.weight`。

### `test_state_dict_roundtrip`

```python
m = _MLP()
out1 = m(x)
sd = m.state_dict()
m2 = _MLP()
m2.load_state_dict(sd)
out2 = m2(x)
assert out1.allclose(out2, atol=1e-6)
```

**防御**：保存再加载后，模型行为不变。这是端到端测试，覆盖 `state_dict` + `load_state_dict` + 数据拷贝的正确性。

### `test_train_eval_propagates`

```python
m = _MLP()
assert m.training and m.fc1.training and m.fc2.training
m.eval()
assert not m.training and not m.fc1.training and not m.fc2.training
```

**防御**：`eval()` 递归切到所有子模块。如果 `train` 没递归，`fc1.training` 会还是 True。

### `test_forward_hook_order`

```python
calls = []
m.register_forward_pre_hook(lambda mod, inp: calls.append("pre"))
m.register_forward_hook(lambda mod, inp, out: calls.append("post"))
m(x)
assert calls == ["pre", "post"]
```

**防御**：pre_hook 在 forward 前，post_hook 在 forward 后，顺序正确。如果 `__call__` 里两个 hook 顺序写反，会挂。

### `test_zero_grad`

```python
(m(x).sum()).backward()
m.zero_grad()
for p in m.parameters():
    assert p.grad is None
```

**防御**：`zero_grad` 把 `grad` 设成 `None` 而非零张量。如果设成零张量，`p.grad is None` 会是 False。

### `test_linear_backward`

```python
(m(x).sum()).backward()
assert m.weight.grad is not None
assert m.bias.grad is not None
```

**防御**：`Linear` 的参数确实收到了梯度。这依赖整个 autograd 链路（`@`、`transpose`、`+`、`sum`）正确，是集成测试。

---

## 5.11 优劣势总结

### 优势

1. **用户代码自然**：`self.weight = Parameter(...)` 就是普通 Python 类写法，无需框架 DSL。
2. **类型即语义**：`Parameter` / `Module` / `Tensor` 三种类型自动分流，无需额外标记。
3. **递归免费**：`parameters()`、`state_dict()`、`train()` 自动递归，嵌套模块无需特殊处理。
4. **扩展性强**：新层只需继承 `Module` 实现 `forward`，注册/收集/保存全自动。
5. **与 autograd 解耦**：`Module` 只管组织参数，`Tensor` 管自动微分，职责清晰。

### 劣势

1. **`__setattr__` 全局拦截**：所有赋值都过一遍 `isinstance`，有微小开销。大模型 `__init__` 时几千个赋值，累积可感知。
2. **`__getattr__` 的递归陷阱**：实现者稍不留神就无限递归。这是 Python 属性协议的固有复杂度。
3. **`state_dict` 不存结构**：模型结构必须在代码里定义，加载时结构必须匹配。改结构要写迁移逻辑。
4. **hooks 不可移除**：lambda 重复注册、无法精确删一个 hook，工程上不便。
5. **`ModuleList` 语义偏离**：minitorch 给了默认 forward，与 PyTorch 语义不一致，可能误导初学者。
6. **无 pickle 支持**：不能直接 `pickle.dump(model)`，要自己存 `state_dict`。

---

## 5.12 下一章预告

本章我们解决了"参数怎么组织"。下一章 **第六章 优化器系统** 将回答：

- 有了参数和梯度，怎么更新？`SGD` 的 `p -= lr * grad` 背后有什么数学？
- 为什么需要动量？Nesterov 动量和普通动量差在哪？
- `Adam` 为什么对每个参数有自适应步长？一阶矩、二阶矩、bias correction 各解决什么问题？
- `param_groups` 怎么让不同层用不同学习率？
- 优化器的 `state` 为什么用 `id(p)` 做 key？参数对象和优化器状态怎么关联？
- `LR Scheduler` 怎么在训练过程中调学习率？`CosineAnnealingLR` 的曲线长什么样？

我们将从梯度下降的数学推导开始，一步步加出动量、Nesterov、Adam，并对照 minitorch 的 `optim/sgd.py` 和 `optim/adam.py` 逐行实现。
