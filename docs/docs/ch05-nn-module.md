# 第五�?nn.Module 体系：参数注册、递归遍历与容器设�?

> 前四章我们解决了"张量怎么�?�?算子怎么分发"�?自动微分怎么�?�?计算图怎么�?�?
> 但一个真正的神经网络有几十上百个参数，靠用户手写 `w1, b1, w2, b2, ...` 管理是不现实的�?
> 本章�?minitorch 怎么�?`nn.Module` 这一个基类，通过 `__setattr__` 拦截、三大注册表�?
> 递归遍历，把"一堆参�?组织�?一个可训练的整�?。这�?PyTorch 最具工程美感的设计之一�?

---

## 5.1 本章目标

读完本章后，你应当能够：

1. 解释 `Module.__setattr__` 为什么能在你�?`self.weight = Parameter(...)` 时自动把参数注册�?`_parameters` 字典，而普通赋值走 `object.__setattr__`�?
2. 画出 `__getattr__` 的查找顺序：`__dict__` �?`_parameters` �?`_modules` �?`_buffers` �?�?`AttributeError`，并说明为什�?`__getattr__` 只在正常查找失败时才被调用�?
3. 区分 `_parameters`、`_modules`、`_buffers` 三大注册表的用途，说出 `BatchNorm` �?`running_mean` 应该放进哪一个�?
4. 手写一个递归 `parameters()` 生成器，正确处理嵌套子模块，并解释为什么用 `yield from` 而不�?`return list`�?
5. 描述 `state_dict()` 的键命名规则（`fc1.weight`、`fc1.fc2.bias`），并实现一�?`load_state_dict` 把外部权重灌回模型�?
6. 解释 `train()` / `eval()` 为什么必须递归调用子模块，以及 `Dropout` �?`BatchNorm` 为什么依�?`self.training` 标志�?
7. �?`Sequential` �?`ModuleList` 拼出一个多层感知机，并说出两者在 `forward` 上的差别�?
8. 写一�?`register_forward_hook`，在 `Linear` 前向前后插入打印逻辑，理�?hook 是调试和可视化（�?activation hook）的基础�?

---

## 5.2 原理铺垫：从一个朴素的需求说�?

### 5.2.1 朴素写法的问�?

假设没有 `nn.Module`，你要写一个两�?MLP�?

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

看起来没问题。但现在你要�?

- **收集所有参数交给优化器**：得手写 `params = [self.w1, self.b1, self.w2, self.b2]`，每加一层就改一次�?
- **保存模型**：得手写 `{"w1": self.w1, "b1": self.b1, ...}`，键名容易写错�?
- **加载模型**：得反向一个个赋值�?
- **切换 train/eval**：如果有 `Dropout`，得给每个子层单独调 `dropout.train()`�?
- **嵌套**：如�?`MLP` 里套一�?`Block`，`Block` 里又�?`Linear`，参数收集要递归�?

这些都是**机械重复**的工作。`nn.Module` 的目标就是把这些自动化�?

### 5.2.2 核心思路：拦截赋值，自动注册

Python 对象的属性都存在 `self.__dict__` 里。`nn.Module` 的核�?trick 是：

> **重写 `__setattr__`，在赋值时判断值的类型，如果是 `Parameter` 就塞�?`_parameters` 字典，如果是 `Module` 就塞�?`_modules` 字典，否则走默认行为�?*

这样用户�?`self.weight = Parameter(...)` 时，`weight` 不会�?`self.__dict__`，而是�?`self._parameters["weight"]`。读取时再靠 `__getattr__` �?`_parameters` 取出来�?

为什么不让参数直接进 `__dict__`？因为要**区分类型**�?

- `Parameter`：要被优化器收集、要�?`state_dict`、要算梯度�?
- `Module`：要递归遍历其参数、要递归�?train/eval�?
- 普通属性（�?`self.training`）：什么都不做�?

如果全塞 `__dict__`，就没法区分，得在每次遍历时 `isinstance` 判断全量扫描——慢且易错�?

### 5.2.3 Python 属性访问机制回�?

要理解这�?trick，必须先搞清�?Python 的属性访问协议：

```
读取 self.name�?
  1. 先查 type(self).__dict__ 里的数据描述符（�?__get__ �?__set__�?
  2. 再查 self.__dict__（实例字典）
  3. 再查 type(self).__dict__ 里的非数据描述符/普通类属�?
  4. 都没找到 �?调用 __getattr__(name)（如果定义了�?

写入 self.name = value�?
  1. 先查 type(self).__dict__ 里的数据描述符（�?__set__�?
  2. 否则直接�?self.__dict__[name] = value
  3. 但如果重写了 __setattr__，则全部�?__setattr__
```

关键点：

!!! tip "`__getattr__` vs `__getattribute__`"
- `__getattribute__`�?*每次**属性访问都调用，性能开销大，容易递归爆栈�?
- `__getattr__`�?*只在正常查找失败�?*才调用，安全且高效�?

minitorch �?`__getattr__`。所�?`self.training` 这种�?`__dict__` 里的属性，**不会**触发 `__getattr__`，直接走快速路径。只�?`self.weight`（不�?`__dict__`，在 `_parameters`）才会触�?`__getattr__`�?

### 5.2.4 MRO（Method Resolution Order）与 `super()`

`Module.__setattr__` 里有一�?`super().__setattr__(name, value)`。这里的 `super()` �?Python 3 里是动态的，等价于�?

```python
super(type(self), self).__setattr__(name, value)
```

但实际更精确：它�?**MRO**（方法解析顺序）找到当前类的下一个类。对�?`Module` 单继承：

```
Module �?object
```

所�?`super().__setattr__` 就是 `object.__setattr__`，即默认行为：写 `self.__dict__[name] = value`�?

MRO 在多继承时才复杂（C3 线性化）。minitorch �?`Module` 是单继承，MRO 就是 `[Module, object]`。但理解 MRO 对后续读 PyTorch 源码（`Linear(Module)`、`BatchNorm(_NormBase)`、`_NormBase(Module)`）有帮助�?

```python
class Module:
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            ...  # �?_parameters
        elif isinstance(value, Module):
            ...  # �?_modules
        else:
            super().__setattr__(name, value)  # �?�?object.__setattr__，进 __dict__
```

### 5.2.5 三大注册表的职责

```
self._parameters: dict[str, Parameter]   # 可学习参数（weight、bias�?
self._modules:    dict[str, Module]      # 子模块（fc1、fc2、layer1�?
self._buffers:    dict[str, Tensor]      # 不可学习但需保存的状态（running_mean、running_var�?
```

| 注册�?      | 谁进�?                 | �?state_dict | �?parameters() | requires_grad | 典型例子                |
| ------------ | ----------------------- | ------------- | --------------- | ------------- | ----------------------- |
| `_parameters`| `Parameter` 实例        | �?           | �?             | 通常 True     | `Linear.weight`         |
| `_modules`   | `Module` 实例           | 递归子模�?   | 递归子模�?     | -             | `MLP.fc1`               |
| `_buffers`   | `Tensor`（手�?register）| �?           | �?             | 通常 False    | `BatchNorm.running_mean`|

`_buffers` �?minitorch 里只声明了字典和 `register_buffer` 接口，本章末尾会讨论它的设计。`BatchNorm` 的完整实现留到后续章节�?

### 5.2.6 递归遍历：为什么用生成�?

`parameters()` 要收集所有嵌套子模块的参数。两种写法：

```python
# 写法 A：列�?
def parameters(self):
    result = list(self._parameters.values())
    for m in self._modules.values():
        result.extend(m.parameters())
    return iter(result)

# 写法 B：生成器（minitorch 采用�?
def parameters(self):
    yield from self._parameters.values()
    for m in self._modules.values():
        yield from m.parameters()
```

生成器的优势�?

1. **惰�?*：不一次性物化整个列表。大模型参数几亿个，列表本身也是内存�?
2. **可组�?*：`yield from` 自然处理嵌套递归，代码简洁�?
3. **语义清晰**：`parameters()` 返回的是"迭代�?，暗�?遍历一次就�?，符�?PyTorch 习惯�?

代价：生成器只能迭代一次。如果用户要多次遍历，得 `list(model.parameters())`。PyTorch 也是这样�?

### 5.2.7 train/eval 的递归传播

很多层的行为依赖模式�?

- `Dropout`：train 时随机置零，eval 时恒等�?
- `BatchNorm`：train 时用 batch 统计并更�?running_mean，eval 时用 running_mean�?
- `Linear`：不依赖模式，但仍要递归（因为可能套�?`Sequential` 里）�?

所�?`train()` 必须递归�?

```python
def train(self, mode=True):
    self.training = mode
    for m in self._modules.values():
        m.train(mode)   # �?递归
    return self
```

返回 `self` 是为了链式调用：`model.train().to(device)`（虽�?minitorch 没有 `.to`）�?

### 5.2.8 容器：Sequential vs ModuleList

`Sequential`：按顺序执行，前一个的输出是后一个的输入�?

```python
net = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
out = net(x)   # 等价�?Linear(8,2)(ReLU()(Linear(4,8)(x)))
```

`ModuleList`：只是把一组模块存起来�?*�?*自动 forward。用户自己决定怎么调�?

```python
layers = ModuleList([Linear(4, 8), Linear(8, 2)])
out = layers[1](layers[0](x))   # 用户自己写执行顺�?
```

两者的 `__init__` 都用 `setattr(self, str(i), m)` 把模块注册成 `"0"`、`"1"`、`"2"`... 这样的属性名。这�?`__setattr__` 会自动把它们塞进 `_modules`，参数收集就能递归到�?

### 5.2.9 hooks：在 forward 前后插入逻辑

```python
m.register_forward_pre_hook(lambda mod, inp: print("before", inp))
m.register_forward_hook(lambda mod, inp, out: print("after", out))
m(x)
# 输出�?
# before (x,)
# after <forward 结果>
```

hooks 存在 `_forward_pre_hooks` �?`_forward_hooks` 字典里，键是 `id(hook)`（避免重复注册同一个函数）。`__call__` 在调 `forward` 前后遍历这些字典调一遍�?

hooks 的用途：

- **调试**：打印每层输入输出形状�?
- **可视�?*：取中间激活做特征图可视化�?
- **梯度剪裁**：在 backward hook 里剪裁梯度（minitorch 暂未实现 backward hook）�?
- **forward hook �?feature**：CNN 可视化里取某一层输出�?

---

## 5.3 设计决策与权�?

| 决策                                | 选择                              | 理由                                                | 代价                                              |
| ----------------------------------- | --------------------------------- | --------------------------------------------------- | ------------------------------------------------- |
| 拦截赋值的方式                      | 重写 `__setattr__`                | 用户写法最自然：`self.w = Parameter(...)`           | 所有赋值都�?`__setattr__`，有微小开销           |
| 区分参数/模块/普通属�?             | `isinstance` 判断                 | 类型即语义，无需额外标记                            | `Parameter` 必须显式包装，不能直接传 Tensor       |
| 注册表数据结�?                     | 三个 `dict`                       | O(1) 查找，键名即属性名                            | 三个 dict 占内存（但相对参数本身可忽略�?        |
| `__getattr__` 而非 `__getattribute__`| 只在查找失败时触�?              | 不影�?`self.training` 等快速路�?                  | 需保证 `_parameters` 等本身在 `__dict__` �?      |
| `parameters()` 返回类型             | 生成�?                           | 惰性、省内存、可递归                                | 只能迭代一次，多次用要 `list()`                  |
| `state_dict` 键命�?                | `prefix + name`，递归�?`.`       | 唯一且可读，匹配 PyTorch                            | 嵌套深时键名长（`block1.layer3.fc2.weight`�?    |
| `train()` 返回�?                   | 返回 `self`                       | 支持链式 `model.train().to(device)`                 | 容易误以为返回新对象，实际是 in-place             |
| `Sequential` 模块命名               | �?`str(i)`�?0"�?1"�?          | 简单、有�?                                         | 不能用名字索引（PyTorch 也这样，�?`OrderedDict`）|
| hooks �?key                        | `id(hook)`                        | 同一函数不重复注�?                                 | lambda 每次新建 id 不同，会重复注册              |
| `__call__` �?`forward`             | 不直接调 `forward`，走 `__call__` | hooks 只在 `__call__` �?                           | 直接 `model.forward(x)` 会绕�?hooks             |
| `_buffers` 是否自动注册             | 手动 `register_buffer`            | 区分普�?Tensor �?buffer                           | 用户要记得调 `register_buffer`                   |
| `__repr__` 实现                     | 列出参数和子模块                  | 调试友好                                            | 大模�?repr 很长                                  |

---

## 5.4 代码逐行实现

### 5.4.1 `parameter.py`：Parameter �?

```python
class Parameter(Tensor):
    def __init__(self, data, requires_grad: bool = True):
        if isinstance(data, Tensor):
            # 已经�?Tensor：复用其 storage/shape/strides，强�?requires_grad=True
            super().__init__(
                data.storage, data.shape, data.strides, data.storage_offset, requires_grad=True
            )
        else:
            # �?list/ndarray：先�?ndarray，再�?Storage
            arr = np.asarray(data)
            storage = Storage.from_numpy(arr)
            super().__init__(
                storage, arr.shape, _compute_contiguous_strides(arr.shape), 0, requires_grad=True
            )
        self.requires_grad = requires_grad   # �?允许用户�?False 冻结参数
```

逐行解读�?

- **`class Parameter(Tensor)`**：继�?Tensor，所�?`Parameter` 既能参与所有张量运算，又能�?`isinstance` 识别�?
- **`if isinstance(data, Tensor)`**：支持两种构造方式：`Parameter(some_tensor)` �?`Parameter([[1,2],[3,4]])`。前者复用底�?Storage（零拷贝），后者新建�?
- **`requires_grad=True` 默认**：参数默认要算梯度。这与普�?Tensor 默认 `False` 形成对比——这�?`Parameter` 存在的核心意义�?
- **`_compute_contiguous_strides`**：从 shape 算出连续布局�?strides，详见第一章�?
- **最后一�?`self.requires_grad = requires_grad`**：允�?`Parameter(data, requires_grad=False)` 冻结参数（迁移学习常用）�?

!!! warning "为什么不直接�?Tensor�?"
如果直接 `self.weight = Tensor(...)`，`__setattr__` 不会把它放进 `_parameters`（因为不�?`Parameter` 实例），优化器就收集不到。`Parameter` �?*类型本身**就是信号。这�?类型即语�?的典型应用�?

### 5.4.2 `module.py`：`__init__`

```python
class Module:
    def __init__(self):
        self._parameters: dict[str, Parameter] = {}   # �?参数注册�?
        self._modules: dict[str, Module] = {}         # �?子模块注册表
        self._buffers: dict[str, Tensor] = {}         # �?状态注册表
        self.training: bool = True                    # �?train/eval 标志
        self._forward_pre_hooks: dict[int, Callable] = {}  # �?�?hook
        self._forward_hooks: dict[int, Callable] = {}      # �?�?hook
```

注意一个微妙点：这些赋�?*会触�?`__setattr__`**。但此时 `_parameters` 等还不存在（`__dict__` 里没有），所�?`__setattr__` 里的 `isinstance(value, Parameter)` 判断：`{}` 不是 `Parameter`，不�?`Module`，走 `super().__setattr__`，正常写�?`__dict__`�?*初始化顺序很重要**：必须先�?`_parameters` 等空字典，后续赋值才能往里塞�?

如果反过来先 `self.weight = Parameter(...)`，此�?`__setattr__` �?`self.__dict__.setdefault("_parameters", {})` 会自动建一个空字典再塞——所�?`setdefault` 的设计就是为了应对这种顺序问题。这�?PyTorch 源码里也有的防御性写法�?

### 5.4.3 `module.py`：`__setattr__` 拦截

```python
def __setattr__(self, name: str, value: Any) -> None:
    if isinstance(value, Parameter):
        # 是参�?�?�?_parameters，不�?__dict__
        self.__dict__.setdefault("_parameters", {})[name] = value
    elif isinstance(value, Module):
        # 是子模块 �?�?_modules
        self.__dict__.setdefault("_modules", {})[name] = value
    else:
        # 普通�?�?走默认，�?__dict__
        super().__setattr__(name, value)
```

逐行解读�?

- **`isinstance(value, Parameter)`**：因�?`Parameter` 继承 `Tensor`，所�?`isinstance(param, Tensor)` 也是 True。但这里先判 `Parameter`，所以参数不会走 `Tensor` 分支�?*判断顺序很重�?*：先具体后一般�?
- **`self.__dict__.setdefault("_parameters", {})`**：直接操�?`__dict__` 而非 `self._parameters`，是为了**避免递归**。如果写 `self._parameters[name] = value`，会再次触发 `__setattr__`（给 `_parameters` 赋值？不，是读 `self._parameters`，读会触�?`__getattr__`...）。直接操�?`__dict__` 绕过属性协议�?
- **`setdefault`**：如�?`_parameters` 不存在就建空字典。这处理�?用户�?`super().__init__()` 之前就赋值参�?的边界情况�?
- **`super().__setattr__`**：走 `object.__setattr__`，即 `self.__dict__[name] = value`�?

!!! tip "为什么不直接 `self.__dict__[name] = value`�?"
等价。但�?`super().__setattr__` �?正统"，未来如�?MRO 中间插入了别�?`__setattr__`（如某个 mixin），能正确转发。这是防御性编程�?

### 5.4.4 `module.py`：`__getattr__` 反查

```python
def __getattr__(self, name: str) -> Any:
    # ⚠️ 只在 self.__dict__ 没找�?name 时才被调�?
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

逐行解读�?

- **`__getattr__` 只在正常查找失败时调�?*：所�?`self.training`（在 `__dict__`）不会进这里，直接走快速路径。这是性能关键�?
- **`self.__dict__.get("_parameters", {})`**：用 `.get` 而非 `self._parameters`，避免递归。如�?`_parameters` 还没初始化（`__init__` 没调），返回 `{}` 而非抛错�?
- **查找顺序**：`_parameters` �?`_modules` �?`_buffers`。实际中名字不会冲突（用户不会同�?`self.x = Parameter()` �?`self.x = Module()`），但顺序定义了优先级�?
- **�?`AttributeError`**：必须抛，否�?Python 内部很多机制（如 `hasattr`、pickle）会失效�?

!!! warning "一个经典坑"
�?`__getattr__` 里写 `self._parameters` �?*无限递归**：`self._parameters` 触发 `__getattr__("_parameters")`，里面又 `self._parameters`... 所以必须用 `self.__dict__.get("_parameters")`。这是写 `__getattr__` 的铁律�?

### 5.4.5 `module.py`：`__call__` �?hooks

```python
def __call__(self, *args, **kwargs) -> Any:
    for hook in self._forward_pre_hooks.values():
        hook(self, args)                          # �?hook：可改输入（这里不改�?
    result = self.forward(*args, **kwargs)        # 真正的前�?
    for hook in self._forward_hooks.values():
        hook(self, args, result)                  # �?hook：可观察输出
    return result
```

逐行解读�?

- **`__call__` 而非直接 `forward`**：用户写 `model(x)` 触发 `__call__`，`__call__` �?`forward`。这�?hooks 有统一入口。如果用户直�?`model.forward(x)`，会绕过 hooks——所以文档都�?别直接调 forward"�?
- **`hook(self, args)`**：签名是 `(module, input)`。input �?tuple（因�?`*args`）�?
- **`hook(self, args, result)`**：签名是 `(module, input, output)`�?
- **遍历 `.values()`**：hooks 存在 dict 里，键是 `id(hook)`。遍�?values 不关心键�?

### 5.4.6 `module.py`：递归 `parameters()` �?`named_parameters()`

```python
def parameters(self) -> Iterator[Parameter]:
    yield from self._parameters.values()          # 自己的直接参�?
    for m in self._modules.values():              # 遍历子模�?
        yield from m.parameters()                 # 递归子模块的参数

def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
    for name, p in self._parameters.items():
        yield prefix + name, p                    # 自己的参数，加前缀
    for name, m in self._modules.items():
        yield from m.named_parameters(prefix + name + ".")  # 递归，前缀�?"子模块名."
```

逐行解读�?

- **`yield from`**：把一个可迭代对象的每个元素逐个 yield 出来，等价于 `for x in iter: yield x`，但更高效（CPython 优化）�?
- **递归终止**：叶子模块（�?`Linear`）的 `_modules` 为空，循环不进，自然终止�?
- **`prefix` 累积**：`named_parameters` 在递归时把当前模块名加进前缀。`MLP.fc1.weight` 就是这么来的：`MLP` �?`fc1.named_parameters("fc1.")`，`fc1` yield `("fc1.weight", w)`�?

### 5.4.7 `module.py`：`state_dict` �?`load_state_dict`

```python
def state_dict(self, prefix: str = "") -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, p in self._parameters.items():
        state[prefix + name] = p                  # 参数�?state
    for name, b in self._buffers.items():
        state[prefix + name] = b                  # buffer 也进 state
    for name, m in self._modules.items():
        state.update(m.state_dict(prefix + name + "."))  # 递归子模�?
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

逐行解读�?

- **`state_dict` 包含参数�?buffer**：但**�?*包含子模块结构。所以加载时模型结构必须匹配�?
- **`load_state_dict` �?in-place**：直接改 `p._storage._data`，不替换 `Parameter` 对象。这样优化器里对 `p` 的引用仍然有效�?
- **`src._numpy_view().ravel()`**：把源张量展平成一�?numpy 视图，再 `[:]` 拷贝到目�?storage。`ravel` 处理形状不一致（实际应一致，但防御性）�?
- **不报错如�?key 不在 state_dict**：`if key in state_dict` 静默跳过。PyTorch 会报 `missing keys`，minitorch 简化了�?

### 5.4.8 `module.py`：`train` / `eval` / `zero_grad`

```python
def train(self, mode: bool = True) -> Module:
    self.training = mode
    for m in self._modules.values():
        m.train(mode)                             # 递归子模�?
    return self                                   # 返回 self 支持链式

def eval(self) -> Module:
    return self.train(False)                      # eval 就是 train(False)

def zero_grad(self) -> None:
    for p in self.parameters():
        p.grad = None                             # �?None 而非 0，省内存且符�?PyTorch 语义
```

逐行解读�?

- **`train` 递归**：必须递归，否则子模块�?`Dropout` 不知道要切模式�?
- **`return self`**：链式调�?`model.train().to(device)`�?
- **`zero_grad` �?`None`**：PyTorch 0.4+ 改成 `None` 而非零张量，原因是：�?省内存（不存零张量）；② 优化器能区分"这一步没算梯�?�?梯度是零"。minitorch 跟随这一设计�?

### 5.4.9 `module.py`：hooks 注册

```python
def register_forward_pre_hook(self, hook: Callable) -> None:
    self._forward_pre_hooks[id(hook)] = hook      # �?id(hook) �?key

def register_forward_hook(self, hook: Callable) -> None:
    self._forward_hooks[id(hook)] = hook
```

逐行解读�?

- **`id(hook)` �?key**：避免重复注册同一个函数对象。`id` 返回内存地址，同一对象 `id` 相同，后注册会覆盖前注册�?
- **lambda 每次新建 id 不同**：所�?`m.register_forward_hook(lambda ...)` 多次调用会注册多个不�?hook。这是已知行为，用户应注意�?

### 5.4.10 `containers.py`：Sequential �?ModuleList

```python
class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, str(i), m)              # 把模块注册成属�?"0"�?1"...

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)                              # 顺序执行，前一个输出是后一个输�?
        return x


class ModuleList(Module):
    def __init__(self, modules: list[Module] | None = None):
        super().__init__()
        if modules is not None:
            for i, m in enumerate(modules):
                setattr(self, str(i), m)          # 同样注册�?"0"�?1"...

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)                              # 也提供默�?forward（顺序执行）
        return x
```

逐行解读�?

- **`setattr(self, str(i), m)`**：等价于 `self."0" = m`（Python 不允许直接写 `self.0 = m`，但 `setattr` 可以）。这触发 `Module.__setattr__`，把 `m` 塞进 `_modules["0"]`�?
- **`Sequential.forward` 顺序执行**：这�?Sequential 的语义。如果用户想要分�?跳连，得自己�?`Module` 子类�?
- **`ModuleList` �?`__len__` �?`__iter__`**：可�?`len(ml)`、`for layer in ml`、`ml[0]`（后�?minitorch 未实现，PyTorch 通过 `__getitem__` 实现）�?
- **`ModuleList.forward` 也顺序执�?*：这�?minitorch 的简化。PyTorch �?`ModuleList` **没有** `forward`，调用会报错——因�?ModuleList 的语义是"存储"而非"执行"。minitorch 加了默认 forward 是为了简化教学�?

!!! warning "�?PyTorch 的差�?"
PyTorch �?`ModuleList` 没有 `forward`，调 `ml(x)` �?`NotImplementedError`。minitorch 给了默认顺序执行，这是教学简化。生产代码里 `ModuleList` 通常用在 `Module` 子类�?`forward` 里手动控制执行顺序�?

### 5.4.11 `linear.py`：Linear �?

```python
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        bound = 1.0 / (in_features**0.5)          # Kaiming uniform 的简化版
        w = np.random.uniform(-bound, bound, (out_features, in_features))
        self.weight = Parameter(Tensor.from_numpy(w))   # �?触发 __setattr__，进 _parameters
        if bias:
            b = np.random.uniform(-bound, bound, (out_features,))
            self.bias = Parameter(Tensor.from_numpy(b))
        else:
            self.bias = None                      # None 不触发注�?

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose()         # x @ W^T，形�?(N, in) @ (in, out) = (N, out)
        if self.bias is not None:
            out = out + self.bias                 # 广播�?bias
        return out
```

逐行解读�?

- **`bound = 1 / sqrt(in_features)`**：这�?PyTorch `Linear` 的默认初始化（Kaiming uniform 的一个特例）。让权重方差 �?`1/in`，保证前向方差稳定�?
- **`weight` 形状 `(out, in)`**：注意是 `(out, in)` 不是 `(in, out)`。所以前向要 `x @ weight.T`。这�?PyTorch 的约定，历史原因�?
- **`self.weight = Parameter(...)`**：这一行触�?`Module.__setattr__`，把 `weight` 塞进 `_parameters["weight"]`。用户后�?`self.weight` 读取时，`__getattr__` �?`_parameters` 取出�?
- **`self.bias = None`**：`None` 不是 `Parameter` 也不�?`Module`，走 `super().__setattr__`，进 `__dict__`。所�?`self.bias` 读取时直接从 `__dict__` 取，不触�?`__getattr__`�?
- **`forward` �?`self.weight`**：触�?`__getattr__`（因�?weight 不在 `__dict__`，在 `_parameters`），取出 `Parameter`。`Parameter` �?`Tensor` 子类，能参与 `@` �?`transpose`�?

### 5.4.12 `module.py`：`__repr__` 调试输出

```python
def __repr__(self) -> str:
    lines = [f"{type(self).__name__("]                # �?类名开�?
    for name, p in self._parameters.items():
        lines.append(f"  {name}: {p}")                # �?缩进列参�?
    for name, m in self._modules.items():
        lines.append(f"  {name}: {m}")                # �?缩进列子模块（递归 repr�?
    lines.append(")")
    return "\n".join(lines)
```

逐行解读�?

- **`type(self).__name__`**：取类名字符串，避免硬编码。子类自动用子类名�?
- **两空格缩�?*：`  {name}:` 让嵌套结构可读。PyTorch �?repr 更精细（�?`extra_repr`、参数形状而非值），minitorch 简化�?
- **`{m}` 触发子模块的 `__repr__`**：递归。所�?`print(MLP())` 会展开所有嵌套层�?
- **只列 `_parameters` �?`_modules`**：不�?`_buffers` 和普通属性。PyTorch �?buffer，minitorch 简化�?

### 5.4.13 `module.py`：`register_parameter` �?`register_buffer`

```python
def register_parameter(self, name: str, param: Parameter) -> None:
    self._parameters[name] = param                   # 直接塞，不走 __setattr__

def register_buffer(self, name: str, tensor: Tensor) -> None:
    self._buffers[name] = tensor                     # buffer 不进 _parameters
```

逐行解读�?

- **`register_parameter`**：等价于 `setattr(self, name, param)`，但更显式。PyTorch 内部用这�?API 注册参数，避�?`__setattr__` 的隐式魔法�?
- **`register_buffer`**：注�?buffer。buffer 不是参数（不�?`parameters()`、不算梯度），但要进 `state_dict`（要保存/加载）。典型例子：`BatchNorm.running_mean`、`running_var`�?
- **直接操作 `_buffers` 字典**：不�?`__setattr__`（因�?`__setattr__` 只识�?`Parameter`/`Module`，普�?`Tensor` 会进 `__dict__` 而非 `_buffers`）。所�?*必须**�?`register_buffer` 才能注册 buffer�?

!!! tip "为什�?buffer 不直�?`self.running_mean = tensor`�?"
如果 `self.running_mean = some_tensor`，`__setattr__` 判断 `some_tensor` 不是 `Parameter` 也不�?`Module`，走默认分支�?`__dict__`。这�?`state_dict` 收集不到它（`state_dict` 只遍�?`_parameters`/`_buffers`/`_modules`），保存模型时会丢。所�?buffer 必须�?`register_buffer` 显式注册�?`_buffers` 字典�?

!!! warning "minitorch �?`state_dict` 漏了 buffer 的加�?"
�?`load_state_dict` 的实现：它只遍历 `_parameters`�?*�?*遍历 `_buffers`。所以即�?`state_dict` 里有 buffer 的键，`load_state_dict` 也不会灌�?buffer。这�?minitorch 的一个简化缺陷（PyTorch �?`load_state_dict` 会处�?buffer）。如果要支持 BatchNorm 的完整保存加载，得在 `load_state_dict` 里加一段遍�?`_buffers` 的逻辑�?

### 5.4.14 三大注册表的对比总结

�?`_parameters`、`_modules`、`_buffers` 放在一起对比，记住一个口诀�?

> **参数要梯度、模块要递归、buffer 要保存�?*

| 操作                | `_parameters` | `_modules` | `_buffers` | `__dict__`（普通属性）|
| ------------------- | ------------- | ---------- | ---------- | --------------------- |
| `parameters()` 收集 | �?           | 递归子模�?| �?        | �?                   |
| `state_dict()` 保存 | �?           | 递归子模�?| �?        | �?                   |
| `train()` 递归      | �?           | �?        | �?        | �?                   |
| `zero_grad()` �?   | �?           | 递归子模�?| �?        | �?                   |
| 算梯�?             | 是（默认�?   | -          | 否（默认�?| �?                   |

这张表是本章最值得记住的：它解释了为什么要把不同东西放进不同字典——因为每种操作要"挑�?的对象不同，分开存就不用每次全量扫描判断�?

---

## 5.5 完整示例

### 5.5.1 定义并使用一�?MLP

```python
import numpy as np
from minitorch import Tensor
from minitorch.nn import Linear, Module, Sequential

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)      # �?触发 __setattr__，fc1 �?_modules
        self.fc2 = Linear(8, 2)      # �?fc2 �?_modules

    def forward(self, x):
        return self.fc2(self.fc1(x))

np.random.seed(0)
model = MLP()
x = Tensor.from_numpy(np.random.randn(3, 4))
out = model(x)
print(out.shape)                     # (3, 2)
```

### 5.5.2 查看注册�?

```python
print(model._modules)                # {'fc1': Linear(...), 'fc2': Linear(...)}
print(model._parameters)             # {} �?MLP 自己没有直接参数
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

### 5.5.4 state_dict 保存与加�?

```python
sd = model.state_dict()
print(sd.keys())
# dict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])

# 新建一个同结构模型，加载权�?
model2 = MLP()
model2.load_state_dict(sd)

# 验证输出一�?
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

### 5.5.8 端到端：训练一�?MLP 回归（与第七章呼应）

```python
from minitorch.optim import SGD
from minitorch.nn import MSELoss

np.random.seed(42)
X = np.random.randn(32, 4)
W_true = np.random.randn(4, 1)
Y = X @ W_true + 0.1 * np.random.randn(32, 1)

model = MLP_regression()   # 假设�?4�?�? �?MLP
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

**症状**：`AttributeError: 'BadMLP' object has no attribute '_parameters'`�?

**原因**：`_parameters` 等字典在 `Module.__init__` 里创建。没调就不存在。虽�?`__setattr__` 用了 `setdefault` 能建 `_parameters`，但 `__getattr__` 里读 `_modules` 会失败�?

**解决**：永远在 `__init__` 第一行调 `super().__init__()`�?

### 陷阱 2：把 Tensor �?Parameter

```python
class Wrong(Module):
    def __init__(self):
        super().__init__()
        self.weight = Tensor.from_numpy(np.random.randn(3, 3))   # �?不是 Parameter!
```

**症状**：`list(model.parameters())` 是空的，优化器没参数可更新�?

**原因**：`__setattr__` 判断 `isinstance(value, Parameter)`，普�?`Tensor` 不通过，走默认分支�?`__dict__`，不被注册�?

**解决**：`self.weight = Parameter(Tensor.from_numpy(...))`�?

### 陷阱 3：在 `__getattr__` 里访�?`self.xxx`

（这是实现者的陷阱，不是用户的。）如果你修�?`__getattr__`，里面写 `self._parameters`，会无限递归。必须用 `self.__dict__.get("_parameters", {})`�?

### 陷阱 4：直接调 `forward` 绕过 hooks

```python
model.forward(x)   # �?不触�?hooks
model(x)           # �?触发 hooks
```

**解决**：永远用 `model(x)`�?

### 陷阱 5：`parameters()` 是生成器，只能迭代一�?

```python
params = model.parameters()
print(len(list(params)))   # 4
print(len(list(params)))   # 0 �?已经耗尽!
```

**解决**：`params = list(model.parameters())` 物化成列表，或每次重新调 `model.parameters()`�?

### 陷阱 6：`Sequential` 不能用名字索�?

```python
net = Sequential(Linear(4, 8), Linear(8, 2))
net["0"]          # �?minitorch 不支持，�?TypeError
net[0]            # �?minitorch 也不支持（PyTorch 支持 __getitem__�?
```

**解决**：用 `net._modules["0"]`（不推荐，访问私有），或改用 `ModuleList` + 自己�?forward�?

### 陷阱 7：`bias=None` �?`self.bias` �?None 但在 `__dict__` �?

```python
lin = Linear(3, 2, bias=False)
print(lin.bias)                  # None
print("_parameters" in lin.__dict__ and "bias" in lin._parameters)  # False
print("bias" in lin.__dict__)    # True �?None 进了 __dict__
```

这不影响功能，但要知�?`None` 走的是默�?`__setattr__` 路径�?

### 陷阱 8：lambda hook 重复注册

```python
for _ in range(3):
    lin.register_forward_hook(lambda m, i, o: None)
print(len(lin._forward_hooks))   # 3 �?每次 lambda 都是新对象，id 不同
```

**解决**：把 lambda 提出来赋给变量，或用具名函数�?

---

## 5.7 与真�?PyTorch 对照

| minitorch                                       | PyTorch                                                | 差异说明                                  |
| ----------------------------------------------- | ------------------------------------------------------ | ----------------------------------------- |
| `Module.__setattr__` 拦截 Parameter/Module      | �?                                                    | 一�?                                     |
| `_parameters`/`_modules`/`_buffers` 三表        | �?                                                    | 一�?                                     |
| `__getattr__` 反查三表                          | �?                                                    | 一�?                                     |
| `parameters()` 返回生成�?                      | �?                                                    | 一�?                                     |
| `named_parameters(prefix="")`                   | �?                                                    | 一�?                                     |
| `state_dict()` 含参数和 buffer                  | �?                                                    | 一�?                                     |
| `load_state_dict` 静默跳过缺失 key              | �?`RuntimeError: missing/unexpected keys`             | minitorch 简化，不严格检�?               |
| `train()` 返回 self                             | �?                                                    | 一�?                                     |
| `zero_grad()` �?`grad = None`                  | 同（0.4+�?                                            | 一�?                                     |
| `register_forward_pre_hook` �?`id(hook)` �?key| �?`RemovableHandle` 对象，返�?handle �?`remove()`   | minitorch 不能移除单个 hook               |
| `Sequential` �?`str(i)` 命名                   | 同（PyTorch 1.x 也用 "0"�?1"�?                        | 一�?                                     |
| `ModuleList` 有默�?`forward`（顺序执行）       | `ModuleList` �?`forward`，调用报�?                   | **差异**：minitorch 教学简�?             |
| `ModuleList` 不支�?`__getitem__`               | 支持，返�?`ModuleList` 子集                           | minitorch 未实�?                         |
| `Linear` 初始�?`uniform(-1/sqrt(in), 1/sqrt(in))`| �?                                                    | 一致（Kaiming uniform 特例�?             |
| `Linear.weight` 形状 `(out, in)`                | �?                                                    | 一�?                                     |
| `__repr__` 列出参数和子模块                     | 更精细，递归 repr，带缩进                              | minitorch 简�?                           |
| �?`__getstate__`/`__setstate__`（pickle�?     | 实现，支�?`pickle.dump(model)`                        | minitorch 未实�?                         |
| �?`to(device)`/`cuda()`/`cpu()`                | �?                                                    | minitorch 单设�?                         |
| �?`extra_repr`                                 | 有，子类可扩�?repr                                    | minitorch 简�?                           |
| �?`apply(fn)` 递归对子模块调函�?              | �?                                                    | minitorch 未实现，�?`train()` 内部用了类似逻辑 |
| �?backward hooks                               | �?`register_full_backward_hook` �?                    | minitorch 未实�?                         |

### 5.7.1 关键差异详解：`ModuleList.forward`

PyTorch �?`ModuleList` 故意不实�?`forward`，因�?`ModuleList` 的语义是"参数容器"而非"执行单元"。用户应该在 `Module` 子类�?`forward` 里手动写执行顺序�?

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

minitorch �?`ModuleList` 加了默认 `forward`（顺序执行），是为了�?`Sequential` �?`ModuleList` 在简单场景下行为一致，降低初学者认知负担。但**生产代码**应遵�?PyTorch 语义：`ModuleList` 不直�?call�?

### 5.7.2 关键差异详解：hooks 的移�?

PyTorch �?`register_forward_hook` 返回一�?`RemovableHandle`，调 `handle.remove()` 能移除该 hook。minitorch �?`id(hook)` �?key，没�?handle，只�?`m._forward_hooks.clear()` 全清。这是工程上的简化�?

---

## 5.8 历史背景

### 5.8.1 PyTorch 之前：Lua Torch �?`nn.Module`

PyTorch 的前身是 Lua Torch。Lua Torch 也有 `nn.Module`，但 Lua 没有重写 `__setattr__` 的优雅机制（Lua �?metatable `__newindex` 类似但语义不同）。参数注册更多靠显式 `self.weight = nn.Parameter(...)` 加上框架内部的注册逻辑�?

### 5.8.2 PyTorch 0.1：`__setattr__` trick 的引�?

PyTorch 0.1�?017 年初）确立了当前�?`nn.Module` 设计：用 `__setattr__` 拦截 `Parameter` �?`Module`，自动注册到三个字典。这一设计来自 Soumith Chintala 等人，核心目标是"让用户代码尽可能自然"�?

对比 TF1 �?`tf.Variable`：用户必须显�?`tf.get_variable("weight", ...)`，且要在 `tf.variable_scope` 下。PyTorch �?`self.weight = Parameter(...)` 更接�?写普�?Python �?的直觉�?

### 5.8.3 `zero_grad` 从零张量�?None 的演�?

PyTorch 0.3 之前，`zero_grad` �?`grad` 设成零张量�?.4 改成 `None`。原因：

1. **省内�?*：不存零张量�?
2. **语义清晰**：`None` 表示"没算梯度"，零张量表示"算了，恰好是�?。优化器能区分�?
3. **避免累加脏数�?*：`backward` 是累加的，如�?`zero_grad` 设零张量，下�?`backward` 在零基础上累加；如果 `None`，`backward` 第一次直接赋值。后者更安全�?

minitorch 跟随 0.4+ �?`None` 语义�?

### 5.8.4 `ModuleList` vs `Sequential` 的分�?

早期 PyTorch 只有 `Sequential`。但用户越来越需�?存一组模块但执行顺序自定�?（如 ResNet 的残差块、Transformer 的多头注意力）。于是引�?`ModuleList`（只存不执行）、`ModuleDict`（按名存）、`ParameterList`、`ParameterDict` 等容器。minitorch 只实现了前两个�?

### 5.8.5 hooks 的演�?

早期 PyTorch �?hooks �?`dict` 存，key �?`id`。后来引�?`RemovableHandle` �?hook 可移除（避免内存泄漏——hook 闭包捕获大对象时不释放）。再后来�?.10+）引�?`register_full_backward_hook` 统一各种 backward hook 的混�?API。minitorch 只实现了 forward hooks 的最简版本�?

---

## 5.9 练习�?

### 练习 1：实�?`ModuleDict`

实现一�?`ModuleDict(Module)`，接�?`dict[str, Module]`，支持按名索引、`__len__`、`__iter__`，且参数能被递归收集�?

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
    关键点：`setattr(self, name, m)` 用名字而非下标，所�?`state_dict` 键会�?`block_a.weight` 而非 `0.weight`�?
???

### 练习 2：为什�?`__setattr__` 里用 `self.__dict__.setdefault` 而非 `self._parameters`

用代码演示如果写�?`self._parameters[name] = value` 会出什么问题�?

??? 解答
    `self._parameters` 会触�?`__getattr__("_parameters")`。如�?`_parameters` 已在 `__dict__`，`__getattr__` 不被调用，直接从 `__dict__` 取，没问题。但如果 `__init__` 还没执行（`_parameters` 不存在），`__getattr__` �?`self.__dict__.get("_parameters", {})` 返回 `{}`（一个临时空字典），往里塞 name→value�?*但这个临时字典不�?`self.__dict__["_parameters"]`**，赋值丢失。所以必须用 `self.__dict__.setdefault("_parameters", {})` 直接操作 `__dict__`�?
???

### 练习 3：实�?`apply(fn)` 递归对子模块调函�?

PyTorch �?`model.apply(init_fn)` 常用于初始化。实现一个�?

??? 解答
    ```python
    def apply(self, fn):
        fn(self)
        for m in self._modules.values():
            m.apply(fn)
        return self
    ```
    用法：`model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)`�?
???

### 练习 4：`state_dict` 为什么不存模块结�?

如果 `state_dict` 只存参数不存结构，加载时怎么保证结构匹配�?

??? 解答
    `state_dict` 只存 `{�? 张量}`，不�?模型有几个层、每层多�?。加载时模型结构由用户代码决定：`model = MLP(); model.load_state_dict(sd)`。如�?`MLP` 结构和保存时不一致（�?`fc1` �?`Linear(4,8)` 变成 `Linear(4,16)`），`load_state_dict` �?`p._storage._data[:] = src...` 会形状不匹配报错（或静默错误）。PyTorch 会检�?`missing_keys` �?`unexpected_keys` 报警告。这种设计让"模型结构"�?参数"解耦：结构在代码里，参数在文件里�?
???

### 练习 5：hooks 的执行顺�?

如果注册了多�?pre_hook，执行顺序是什么？为什么用 `dict` 而非 `list`�?

??? 解答
    顺序�?`dict.values()` 的迭代顺序，Python 3.7+ �?*插入顺序**。所以先注册先执行。用 `dict` 而非 `list` 是为了用 `id(hook)` �?key 去重——同一个函数对象重复注册不会重复执行。代价是 lambda 每次新建对象，`id` 不同，会重复注册（已知行为）�?
???

---

## 5.10 关键测试解读

`tests/test_module.py` 的每个测试都在防御一�?bug�?

### `test_parameter_requires_grad`

```python
p = Parameter(Tensor.from_numpy(np.array([1.0, 2.0])))
assert p.requires_grad
```

**防御**：`Parameter` 默认 `requires_grad=True`。如果某次重构把 `super().__init__` �?`requires_grad` 写成 `False`，这个测试会挂�?

### `test_module_setattr_register`

```python
m = Linear(3, 2)
assert "weight" in m._parameters
assert "bias" in m._parameters
```

**防御**：`self.weight = Parameter(...)` 确实进了 `_parameters` 而非 `__dict__`。如�?`__setattr__` �?`isinstance` 判断写反或漏了，参数会进错地方�?

### `test_nested_parameters_recursive`

```python
m = _MLP()   # fc1: Linear(4,8), fc2: Linear(8,2)
params = list(m.parameters())
assert len(params) == 4
```

**防御**：递归收集。`MLP` 自己 `_parameters` 是空的，参数全在子模块里。如�?`parameters()` 没递归，会返回 0 个�?

### `test_named_parameters_prefix`

```python
names = dict(m.named_parameters())
assert "fc1.weight" in names
assert "fc2.bias" in names
```

**防御**：前缀拼接正确。如�?`prefix + name + "."` 写错（如漏了 `.`），键会�?`fc1weight` 而非 `fc1.weight`�?

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

**防御**：保存再加载后，模型行为不变。这是端到端测试，覆�?`state_dict` + `load_state_dict` + 数据拷贝的正确性�?

### `test_train_eval_propagates`

```python
m = _MLP()
assert m.training and m.fc1.training and m.fc2.training
m.eval()
assert not m.training and not m.fc1.training and not m.fc2.training
```

**防御**：`eval()` 递归切到所有子模块。如�?`train` 没递归，`fc1.training` 会还�?True�?

### `test_forward_hook_order`

```python
calls = []
m.register_forward_pre_hook(lambda mod, inp: calls.append("pre"))
m.register_forward_hook(lambda mod, inp, out: calls.append("post"))
m(x)
assert calls == ["pre", "post"]
```

**防御**：pre_hook �?forward 前，post_hook �?forward 后，顺序正确。如�?`__call__` 里两�?hook 顺序写反，会挂�?

### `test_zero_grad`

```python
(m(x).sum()).backward()
m.zero_grad()
for p in m.parameters():
    assert p.grad is None
```

**防御**：`zero_grad` �?`grad` 设成 `None` 而非零张量。如果设成零张量，`p.grad is None` 会是 False�?

### `test_linear_backward`

```python
(m(x).sum()).backward()
assert m.weight.grad is not None
assert m.bias.grad is not None
```

**防御**：`Linear` 的参数确实收到了梯度。这依赖整个 autograd 链路（`@`、`transpose`、`+`、`sum`）正确，是集成测试�?

---

## 5.11 优劣势总结

### 优势

1. **用户代码自然**：`self.weight = Parameter(...)` 就是普�?Python 类写法，无需框架 DSL�?
2. **类型即语�?*：`Parameter` / `Module` / `Tensor` 三种类型自动分流，无需额外标记�?
3. **递归免费**：`parameters()`、`state_dict()`、`train()` 自动递归，嵌套模块无需特殊处理�?
4. **扩展性强**：新层只需继承 `Module` 实现 `forward`，注�?收集/保存全自动�?
5. **�?autograd 解�?*：`Module` 只管组织参数，`Tensor` 管自动微分，职责清晰�?

### 劣势

1. **`__setattr__` 全局拦截**：所有赋值都过一�?`isinstance`，有微小开销。大模型 `__init__` 时几千个赋值，累积可感知�?
2. **`__getattr__` 的递归陷阱**：实现者稍不留神就无限递归。这�?Python 属性协议的固有复杂度�?
3. **`state_dict` 不存结构**：模型结构必须在代码里定义，加载时结构必须匹配。改结构要写迁移逻辑�?
4. **hooks 不可移除**：lambda 重复注册、无法精确删一�?hook，工程上不便�?
5. **`ModuleList` 语义偏离**：minitorch 给了默认 forward，与 PyTorch 语义不一致，可能误导初学者�?
6. **�?pickle 支持**：不能直�?`pickle.dump(model)`，要自己�?`state_dict`�?

---

## 5.12 下一章预�?

本章我们解决�?参数怎么组织"。下一�?**第六�?优化器系�?* 将回答：

- 有了参数和梯度，怎么更新？`SGD` �?`p -= lr * grad` 背后有什么数学？
- 为什么需要动量？Nesterov 动量和普通动量差在哪�?
- `Adam` 为什么对每个参数有自适应步长？一阶矩、二阶矩、bias correction 各解决什么问题？
- `param_groups` 怎么让不同层用不同学习率�?
- 优化器的 `state` 为什么用 `id(p)` �?key？参数对象和优化器状态怎么关联�?
- `LR Scheduler` 怎么在训练过程中调学习率？`CosineAnnealingLR` 的曲线长什么样�?

我们将从梯度下降的数学推导开始，一步步加出动量、Nesterov、Adam，并对照 minitorch �?`optim/sgd.py` �?`optim/adam.py` 逐行实现�?
