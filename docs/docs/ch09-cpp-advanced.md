# 第九章 C++ 高级特性：Autograd 引擎 / 多线程 / Double Backward / Allocator / Profiler / Hooks / Anomaly / Checkpointing

> 第八章搭好了 C++ 核心的"地基"——Storage、TensorImpl、基本算子、pybind11 绑定。
> 本章在这块地基上盖"上层建筑"：**Autograd 引擎**（建图 + 反向传播）、**多线程并行**、
> **double backward**（高阶导数）、**自定义 Allocator**（内存池）。
> 这些特性让 minitorch 的 C++ 核心从"能跑"升级到"能和真实 PyTorch 对照工程细节"。
> 每一节都先讲原理、再逐行解读实现、最后对照 `torch/csrc/autograd/` 和 `c10/Allocator`。

---

## 9.1 本章目标

读完本章后，你应当能够：

1. 说出 C++ Autograd 的 **Node 虚函数设计**：为什么用 `virtual apply()` 替代 `std::function`，每个算子怎么定义自己的 Node 子类。
2. 解释 **AccumulateGrad** 的作用：叶子张量的梯度怎么累加，`add_inplace` 怎么避免重复分配。
3. 手写一个 **grad_mode RAII 守卫**：`thread_local` 标志 + 构造时保存/析构时恢复，对应 `torch.no_grad()`。
4. 画出 **多线程 Engine** 的调度流程：ThreadPool + 原子依赖计数 + 并行执行无依赖 Node。
5. 解释 **double backward** 的原理：`create_graph=true` 时为什么不启用 NoGradGuard，MulNode 为什么用 `autograd::mul` 而非 `ops::mul`。
6. 说出 **broadcast_to 保留 grad_fn** 的 bug：为什么广播张量需要复制原始张量的 `grad_fn` 和 `is_leaf`。
7. 手写一个 **自定义 Allocator**：Allocator 接口 + DefaultAllocator 统计 + PoolAllocator 内存池。
8. 说出 **Profiler** 的设计：在 `run_backward` 中记录每个 Node 的耗时和内存分配，对应 `torch.autograd.profiler`。
9. 解释 **梯度钩子** 的机制：`register_hook` 在 `AccumulateGrad::apply` 中调用，可修改梯度，对应 `tensor.register_hook()`。
10. 说出 **Anomaly Detection** 的原理：`check_anomaly` 在每次梯度生成后检测 NaN/Inf，对应 `torch.autograd.detect_anomaly()`。
11. 解释 **Gradient Checkpointing** 的用重计算换内存策略：前向 NoGrad、backward 重执行前向、detached 副本避免覆盖原图，对应 `torch.utils.checkpoint.checkpoint()`。
12. 对照真实 PyTorch：`torch::autograd::Node` vs 我们的 `Node`、`Engine::execute_with_thread_pool` vs 我们的 `run_backward_mt`、`c10::Allocator` vs 我们的 `Allocator`。

---

## 9.2 C++ Autograd 引擎

### 9.2.1 为什么 Autograd 要用 C++ 写

第八章的 Storage/TensorImpl/ops 是"前向计算"。autograd 是"反向计算"——从输出出发，沿计算图逆向传播梯度。在阶段一（纯 Python）中，autograd 引擎用 Python 写，跑得对但慢：

- 每个 Node 的 `backward_fn` 是 Python lambda，调用开销大。
- 拓扑排序用 Python 递归 DFS，深图栈溢出。
- 梯度累加每次创建新 list，GC 压力大。

C++ 重写后：

- Node 用虚函数 `apply()`，编译器可内联优化。
- 拓扑排序用 `std::function` + 显式栈，无栈溢出。
- 梯度累加用 `add_inplace` 直接改内存，零分配。

### 9.2.2 Node 基类：虚函数设计

```cpp
// autograd/node.h
class Node {
public:
    std::vector<NodePtr> next_edges;   // 指向前驱 Node（梯度传播路径）
    std::string name;                  // 调试用："Add" / "Mul" / "AccumulateGrad"
    TensorImplPtr output;              // 此 Node 产生的前向输出（用于 retain_grad）

    Node() = default;
    virtual ~Node() = default;

    // 纯虚函数：子类必须实现，接收上游梯度，返回对各输入的梯度
    virtual std::vector<TensorImplPtr> apply(TensorImplPtr grad) = 0;

    bool is_accumulate_grad() const { return name == "AccumulateGrad"; }
};
```

**为什么用虚函数而不是 `std::function`？**

旧方案是 `std::function<std::vector<TensorImplPtr>(TensorImplPtr)> backward_fn`，每个 Node 存一个 lambda。问题：

1. `std::function` 有类型擦除 + 堆分配开销（小对象优化不一定命中）。
2. lambda 捕获 `shared_ptr` 容易循环引用。
3. 调试时看不到 lambda 内部，只有一堆 `std::function` 指针。

新方案用虚函数：

1. 每个 Node 子类（AddNode, MulNode, ...）在自己的 `apply()` 里直接写 backward 逻辑，编译器可内联。
2. 子类成员变量就是 backward 需要的上下文（shape、mask、原始输入），不用 lambda 捕获。
3. 调试时 `node->name` 直接告诉你是什么算子，gdb 里能 downcast 看成员。

### 9.2.3 Node 子类：每个算子一个

以 `MulNode` 为例：

```cpp
// autograd/ops.cpp
class MulNode : public Node {
public:
    std::vector<int64_t> a_shape, b_shape;  // 原始输入形状（用于 reduce_grad）
    TensorImplPtr orig_a, orig_b;           // 原始输入（非广播后的！）

    MulNode(std::vector<int64_t> a, std::vector<int64_t> b,
            TensorImplPtr a_, TensorImplPtr b_)
        : a_shape(std::move(a)), b_shape(std::move(b)),
          orig_a(std::move(a_)), orig_b(std::move(b_)) { name = "Mul"; }

    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        // ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
        // 注意：这里用 autograd::mul 而非 ops::mul（double backward 需要）
        auto grad_a = mul(grad, orig_b);
        auto grad_b = mul(grad, orig_a);
        return {ops::reduce_grad(grad_a, a_shape),
                ops::reduce_grad(grad_b, b_shape)};
    }
};
```

**关键细节：存原始输入 `orig_a`/`orig_b`，而非广播后的 `ba`/`bb`。**

前向计算时 `mul(a, b)` 先广播：`auto [ba, bb] = ops::broadcast_tensors(a, b)`，然后 `ops::mul(ba, bb)`。如果 Node 存广播后的 `ba`/`bb`，`collect_edges` 会为广播张量创建 `AccumulateGrad`，梯度累加到广播张量而非原始 `a`/`b`——**梯度丢失 bug**。

修复：Node 存原始 `a`/`b`，`collect_edges({a, b})` 正确为原始张量建 `AccumulateGrad`。

所有 Node 子类一览：

| Node 子类 | 成员 | apply 逻辑 |
|-----------|------|-----------|
| `AddNode` | `a_shape, b_shape` | `{reduce_grad(grad, a_shape), reduce_grad(grad, b_shape)}` |
| `SubNode` | `a_shape, b_shape` | `{reduce_grad(grad, a_shape), reduce_grad(neg(grad), b_shape)}` |
| `MulNode` | `a_shape, b_shape, orig_a, orig_b` | `{reduce_grad(mul(grad,orig_b), a_shape), reduce_grad(mul(grad,orig_a), b_shape)}` |
| `DivNode` | `a_shape, b_shape, orig_a, orig_b` | `{reduce_grad(div(grad,bb), ...), reduce_grad(neg(div(mul(grad,ba),mul(bb,bb))), ...)}` |
| `NegNode` | 无 | `{neg(grad)}` |
| `ReluNode` | `mask` | `{mul(grad, mask)}` |
| `MatmulNode` | `a_copy, b_copy, a_was_1d, b_was_1d` | `grad_a = matmul(g, b^T), grad_b = matmul(a^T, g)` |
| `SumNode` | `a_shape, dim, keepdim` | `{broadcast_to(grad, a_shape)}` |
| `MeanNode` | `a_shape, inv_n` | `{broadcast_to(mul(grad, inv_n), a_shape)}` |
| `TransposeNode` | `dim0, dim1` | `{grad.transpose(dim0, dim1)}` |

### 9.2.4 AccumulateGrad：叶子梯度累加

```cpp
// autograd/node.cpp
class AccumulateGrad : public Node {
public:
    TensorImplPtr variable;   // 指向叶子张量

    explicit AccumulateGrad(TensorImplPtr var) : variable(std::move(var)) {
        name = "AccumulateGrad";
    }

    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        if (!variable->grad()) {
            // 第一次：直接设置
            variable->set_grad(grad);
        } else if (grad && grad->requires_grad()) {
            // double backward：grad 本身有图，用 autograd::add 建新图
            variable->set_grad(autograd::add(variable->grad(), grad));
        } else {
            // 普通累加：原地加，零分配
            auto existing = variable->grad();
            ops::add_inplace(existing, grad);
            variable->set_grad(existing);
        }
        return {};  // AccumulateGrad 是终点，不继续传播
    }
};
```

三个分支：

1. **第一次累加**：叶子还没有 grad，直接 `set_grad`。
2. **double backward**：`grad` 自身 `requires_grad`（因为 `create_graph=true`），用 `autograd::add` 而非 `ops::add_inplace`，这样加法也建图，二阶导能继续反向。
3. **普通累加**：`add_inplace` 直接改 `existing` 的内存，不分配新 TensorImpl。

### 9.2.5 add_inplace：原地梯度累加

```cpp
// aten/ops.h
void add_inplace(TensorImplPtr& target, const TensorImplPtr& source);
```

普通加法 `ops::add(a, b)` 创建新 TensorImpl + 新 Storage。但在梯度累加场景，`target` 是已有的梯度张量，我们只想把 `source` 加进去，不需要新对象：

```cpp
// 概念上等价于：
for (size_t i = 0; i < target->numel(); ++i)
    target_data[i] += source_data[i];
```

好处：**零分配**。反向传播中每个 Node 的梯度都可能被多次累加（多后继汇合），用 `add_inplace` 省掉 N 次内存分配 + N 次 shared_ptr 引用计数原子操作。

### 9.2.6 grad_mode：RAII 守卫

```cpp
// autograd/grad_mode.h
inline thread_local bool grad_mode_enabled = true;

inline bool is_grad_enabled() { return grad_mode_enabled; }

class NoGradGuard {
public:
    NoGradGuard() : prev_(grad_mode_enabled) { grad_mode_enabled = false; }
    ~NoGradGuard() { grad_mode_enabled = prev_; }
    NoGradGuard(const NoGradGuard&) = delete;   // 不可拷贝
    NoGradGuard& operator=(const NoGradGuard&) = delete;
private:
    bool prev_;
};
```

**`thread_local`**：每个线程独立标志。多线程 Engine 中，一个线程 `NoGradGuard` 不影响其他线程。

**RAII**：构造时保存旧值并设 false，析构时恢复。即使 `apply()` 抛异常，析构也会执行，grad_mode 不会卡在 false。

**用法**：`run_backward` 在 `create_graph=false` 时用 `NoGradGuard` 包裹整个反向传播，确保 backward 中的前向计算（如 `mul(grad, orig_b)`）不建图：

```cpp
void run_backward(...) {
    std::unique_ptr<NoGradGuard> no_grad;
    if (!create_graph) no_grad = std::make_unique<NoGradGuard>();
    // ... 反向传播 ...
}  // no_grad 析构，恢复 grad_mode
```

### 9.2.7 建图算子：autograd 命名空间

`autograd::add` / `autograd::mul` 等 = `ops::add` / `ops::mul` + 建图：

```cpp
// autograd/ops.cpp
TensorImplPtr mul(const TensorImplPtr& a, const TensorImplPtr& b) {
    auto [ba, bb] = ops::broadcast_tensors(a, b);
    auto result = ops::mul(ba, bb);                    // 前向计算

    if (!is_grad_enabled() || !any_requires_grad({a, b}))
        return result;                                 // 不建图

    auto node = std::make_shared<MulNode>(a->shape(), b->shape(), a, b);
    attach_node(result, node, collect_edges({a, b}));  // 挂 grad_fn + next_edges
    return result;
}
```

`collect_edges` 决定梯度往哪传：

```cpp
static std::vector<NodePtr> collect_edges(const std::vector<TensorImplPtr>& inputs) {
    std::vector<NodePtr> edges;
    for (const auto& t : inputs) {
        if (t && t->requires_grad()) {
            if (t->grad_fn()) {
                edges.push_back(t->grad_fn());         // 非叶子：接它的 grad_fn
            } else {
                edges.push_back(std::make_shared<AccumulateGrad>(t));  // 叶子：新建 AccumulateGrad
            }
        } else {
            edges.push_back(nullptr);                  // 不需要梯度
        }
    }
    return edges;
}
```

### 9.2.8 单线程 run_backward

```cpp
void run_backward(NodePtr root, TensorImplPtr root_grad,
                  bool retain_graph, bool retain_grad, bool create_graph) {
    std::unique_ptr<NoGradGuard> no_grad;
    if (!create_graph) no_grad = std::make_unique<NoGradGuard>();

    auto topo = topological_sort(root);        // DFS 后序 → 逆序 = 拓扑序

    std::unordered_map<Node*, TensorImplPtr> grad_map;
    grad_map[root.get()] = root_grad;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {  // 逆拓扑序
        auto& node = *it;
        auto grad_it = grad_map.find(node.get());
        if (grad_it == grad_map.end() || !grad_it->second) continue;

        TensorImplPtr grad = grad_it->second;

        if (node->is_accumulate_grad()) {
            node->apply(grad);                 // 累加到叶子
            continue;
        }

        if (retain_grad && node->output) { ... }  // 保留中间梯度

        auto grads = node->apply(grad);        // 调虚函数
        if (grads.empty()) continue;

        for (size_t i = 0; i < node->next_edges.size() && i < grads.size(); ++i) {
            auto& edge = node->next_edges[i];
            auto& g = grads[i];
            if (!edge || !g) continue;
            // 梯度汇合：多后继 → 同一前驱，累加
            auto prev_it = grad_map.find(edge.get());
            if (prev_it == grad_map.end() || !prev_it->second) {
                grad_map[edge.get()] = g;
            } else if (create_graph) {
                grad_map[edge.get()] = autograd::add(prev_it->second, g);
            } else {
                ops::add_inplace(prev_it->second, g);
            }
        }
    }

    if (!retain_graph) {
        for (auto& node : topo) node->next_edges.clear();  // 释放图
    }
}
```

流程：拓扑排序 → 逆序遍历 → 每个 Node 调 `apply(grad)` 得到对各输入的梯度 → 沿 `next_edges` 传给前驱 → 多后继汇合时累加。

---

## 9.3 多线程 Engine

### 9.3.1 为什么要多线程

计算图中，**无依赖关系的 Node 可以并行执行**。例如：

```
        loss
       /    \
      L1     L2
     /  \   /  \
    a    b  c   d
```

`L1` 和 `L2` 互不依赖（都不在对方的路径上），可以并行 backward。`a`/`b` 必须等 `L1` 完成，`c`/`d` 必须等 `L2`。

PyTorch 的 `Engine::execute_with_thread_pool` 正是这么做的：用线程池 + 依赖计数，Node 的所有后继都完成后才调度它。

### 9.3.2 ThreadPool 类

```cpp
// c10/thread_pool.h
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) : stop_(false), active_(0) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
    }

    void submit(std::function<void()> task) {
        { std::lock_guard<std::mutex> lock(mutex_); tasks_.push(std::move(task)); ++active_; }
        cv_.notify_one();
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return active_ == 0 && tasks_.empty(); });
    }

    static size_t default_num_threads() {
        unsigned n = std::thread::hardware_concurrency();
        return n > 0 ? n : 4;
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --active_;
                if (active_ == 0 && tasks_.empty()) done_cv_.notify_all();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;        // worker 等任务
    std::condition_variable done_cv_;   // 主线程等全部完成
    bool stop_;
    std::atomic<int> active_;
};
```

经典生产者-消费者：

- **worker 线程**：`cv_.wait` 等任务队列非空 → 取任务 → 执行 → `active_--` → 如果全完成则 `done_cv_.notify_all`。
- **主线程**：`submit` 投任务 → `wait_all` 等 `active_ == 0 && tasks_.empty()`。

### 9.3.3 run_backward_mt：依赖计数调度

```cpp
void run_backward_mt(NodePtr root, TensorImplPtr root_grad,
                     bool retain_graph, bool retain_grad,
                     int num_threads, bool create_graph) {
    auto topo = topological_sort(root);

    // Node* → index（用 vector<atomic> 替代 unordered_map<atomic>）
    std::unordered_map<Node*, size_t> node_index;
    for (size_t i = 0; i < topo.size(); ++i)
        node_index[topo[i].get()] = i;

    // 入度 = 有多少后继会向此 Node 投递梯度
    std::vector<std::atomic<int>> dep_count(topo.size());
    for (auto& dc : dep_count) dc.store(0);
    for (auto& node : topo)
        for (auto& edge : node->next_edges)
            if (edge) dep_count[node_index[edge.get()]]++;

    // 共享状态
    std::mutex grad_mutex;
    std::unordered_map<Node*, TensorImplPtr> grad_map;
    grad_map[root.get()] = root_grad;

    ThreadPool pool(num_threads > 0 ? num_threads : ThreadPool::default_num_threads());
    std::atomic<size_t> remaining(topo.size());

    // 递归任务函数
    std::function<void(NodePtr)> process_node = [&](NodePtr node) {
        std::unique_ptr<NoGradGuard> no_grad;
        if (!create_graph) no_grad = std::make_unique<NoGradGuard>();

        // 1. 取梯度（加锁）
        TensorImplPtr grad;
        { std::lock_guard<std::mutex> lock(grad_mutex);
          auto it = grad_map.find(node.get());
          if (it != grad_map.end()) grad = it->second; }

        if (grad) {
            if (node->is_accumulate_grad()) {
                node->apply(grad);
            } else {
                // retain_grad ...
                auto grads = node->apply(grad);
                if (!grads.empty()) {
                    std::lock_guard<std::mutex> lock(grad_mutex);
                    // 梯度投递到 next_edges（加锁）
                    for (size_t i = 0; i < node->next_edges.size() && i < grads.size(); ++i) {
                        auto& edge = node->next_edges[i];
                        auto& g = grads[i];
                        if (!edge || !g) continue;
                        auto it = grad_map.find(edge.get());
                        if (it == grad_map.end() || !it->second) {
                            grad_map[edge.get()] = g;
                        } else if (create_graph) {
                            grad_map[edge.get()] = autograd::add(it->second, g);
                        } else {
                            ops::add_inplace(it->second, g);
                        }
                    }
                }
            }
        }

        // 2. 通知前驱：此来源已处理，dep_count--，归零则调度
        for (auto& edge : node->next_edges) {
            if (!edge) continue;
            size_t idx = node_index[edge.get()];
            if (dep_count[idx].fetch_sub(1) == 1)   // 原子减，返回旧值
                pool.submit([&, edge] { process_node(edge); });
        }

        remaining.fetch_sub(1);
    };

    // 启动：root 的 dep_count 应为 0（没有后继向它投递）
    pool.submit([&] { process_node(root); });
    pool.wait_all();

    if (!retain_graph)
        for (auto& node : topo) node->next_edges.clear();
}
```

**调度逻辑**：

1. **入度计算**：`dep_count[i]` = 有多少个 Node 的 `next_edges` 指向 `topo[i]`。即 `topo[i]` 需要等多少个后继完成后才能执行。
2. **root 启动**：root 的 `dep_count` 为 0（没人向它投递），直接 submit。
3. **process_node**：取梯度 → 调 `apply` → 把梯度投递到 `next_edges` → 对每个前驱 `dep_count--`，如果归零就 submit 它。
4. **`fetch_sub(1) == 1`**：原子减返回旧值。旧值为 1 表示减后为 0，即所有后继已完成，可以调度。

**线程安全**：

- `grad_map` 用 `grad_mutex` 保护。梯度投递和累加都在锁内。
- `dep_count` 是 `vector<atomic<int>>`，原子操作不需要锁。
- `NoGradGuard` 是 `thread_local`，每个 worker 线程独立。

!!! warning "为什么用 vector<atomic> 而不是 unordered_map<atomic>？"
    `std::atomic` 不可拷贝/移动，不能直接放在 `unordered_map` 中。用 `vector<atomic<int>>` + `unordered_map<Node*, size_t>` 做索引，是 C++ 中的标准做法。

### 9.3.4 与 PyTorch Engine 对照

| 我们的 `run_backward_mt` | 真实 `torch::autograd::Engine` |
|--------------------------|-------------------------------|
| `ThreadPool` 固定线程数 | `ThreadPool` 可动态调整 |
| `vector<atomic<int>>` 依赖计数 | `std::atomic<int>` per NodeTask |
| `grad_mutex` 全局锁 | per-device 队列 + `ReadyQueue` |
| 拓扑序 + 依赖计数 | `GraphTask` + `NodeTask` 调度 |
| 无 CUDA stream 调度 | 按 device/stream 调度 |

PyTorch 的 Engine 更复杂：多设备（CPU + 多 GPU）、CUDA stream 同步、优先级队列。但**核心思路一致**：依赖计数 + 线程池 + 原子调度。

---

## 9.4 Double Backward：高阶导数

### 9.4.1 什么是 double backward

一阶导：`y = x^2` → `dy/dx = 2x`。
二阶导：`d²y/dx² = 2`。

在 autograd 中，二阶导 = 对一阶导再 backward 一次。但一阶导 `2x` 本身是一个**计算图**（`x → mul(2, x)`），要对它 backward，这个图必须存在——即 `create_graph=true`。

### 9.4.2 create_graph 参数

```cpp
// c10/tensor.h
void backward(TensorImplPtr gradient = nullptr,
              bool retain_graph = false,
              bool retain_grad = false,
              bool create_graph = false);
```

`create_graph=true` 时：

1. **不启用 NoGradGuard**：backward 中的前向计算（如 `mul(grad, orig_b)`）会建图。
2. **MulNode::apply 用 `autograd::mul`**：建图乘法，而非 `ops::mul`（不建图）。
3. **AccumulateGrad::apply 用 `autograd::add`**：梯度累加也建图。
4. **梯度汇合用 `autograd::add`**：多后继梯度求和也建图。

这样，反向传播本身产生的新计算（梯度计算）也被记录成图，可以再 backward。

### 9.4.3 MulNode 用 autograd::mul

```cpp
class MulNode : public Node {
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        auto grad_a = mul(grad, orig_b);    // autograd::mul，不是 ops::mul
        auto grad_b = mul(grad, orig_a);
        return {ops::reduce_grad(grad_a, a_shape),
                ops::reduce_grad(grad_b, b_shape)};
    }
};
```

`autograd::mul` 在 `is_grad_enabled()` 时会建图（挂 MulNode）。`ops::mul` 永远不建图。

当 `create_graph=true`：`NoGradGuard` 不启用 → `is_grad_enabled()=true` → `autograd::mul` 建图 → `grad_a` 有 `grad_fn` → 可以再 backward。

当 `create_graph=false`：`NoGradGuard` 启用 → `is_grad_enabled()=false` → `autograd::mul` 不建图 → `grad_a` 无 `grad_fn` → 不能再 backward（但省内存）。

### 9.4.4 broadcast_to 保留 grad_fn 的 bug

**问题**：`y = x * w`（x 是 `[3]`，w 是 `[1]` 广播到 `[3]`）。backward 时 `grad_x = grad_y * w_broadcast`。但 `w_broadcast` 是广播张量，它的 `grad_fn` 为空、`is_leaf=false`。`collect_edges` 为它创建 `AccumulateGrad(w_broadcast)`，梯度累加到 `w_broadcast` 而非原始 `w`——**w 的梯度丢失**。

**修复**：`broadcast_to` 保留原始张量的 `grad_fn` 和 `is_leaf`：

```cpp
// aten/ops.cpp
TensorImplPtr broadcast_to(const TensorImplPtr& a,
                           const std::vector<int64_t>& target_shape) {
    // ... 计算 new_strides ...
    auto result = std::make_shared<TensorImpl>(a->storage(), target_shape, new_strides,
                                               a->storage_offset(), a->requires_grad());
    result->set_grad_fn(a->grad_fn());    // ← 保留！
    result->set_is_leaf(a->is_leaf());    // ← 保留！
    return result;
}
```

这样 `w_broadcast->grad_fn()` 返回 `w` 的 `grad_fn`（或空，表示叶子），`collect_edges` 正确找到原始 `w` 的 `AccumulateGrad`。

### 9.4.5 端到端示例

```python
import minitorch as mt

x = mt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * x          # y = x²
g = mt.Tensor([1.0, 1.0, 1.0])
y.backward(g, create_graph=True)   # 一阶：x.grad = 2x = [2, 4, 6]
# 此时 x.grad 本身是一个图（2x），可以再 backward

x.grad.backward(mt.Tensor([1.0, 1.0, 1.0]))  # 二阶：d(2x)/dx = 2
# x.grad 现在变成 [2, 2, 2]（二阶导数值）
```

一阶 backward 时，`create_graph=True`：

1. `MulNode::apply` 用 `autograd::mul(grad, orig_x)` → `grad_x = 1 * x`，建图。
2. `AccumulateGrad::apply` 用 `autograd::add` → `x.grad = 0 + grad_x = x`，建图。
3. 最终 `x.grad` 的计算图：`x → mul(1, x) → add(0, ...) → x.grad`。

二阶 backward 时，对 `x.grad` 的图再 backward，得到 `d(x)/dx = 1`，乘以 `[1,1,1]` 得 `[1,1,1]`... 但因为 `y = x²`，`dy/dx = 2x`，`d²y/dx² = 2`。实际结果取决于 `g` 和二阶 backward 的 gradient 参数。

---

## 9.5 自定义 Allocator

### 9.5.1 为什么需要自定义 Allocator

默认 `new`/`delete` 的问题：

1. **碎片化**：频繁分配/释放不同大小，内存碎片化。
2. **开销**：每次 `new` 调 malloc，有系统调用开销。
3. **不可控**：无法统计分配量、无法限制内存、无法对齐。

PyTorch 的 `c10::Allocator` 解决这些：`CPUAllocator`、`CUDAAllocator`、`PinnedMemoryAllocator`... Storage 通过 Allocator 分配内存。

### 9.5.2 Allocator 接口

```cpp
// c10/allocator.h
class Allocator {
public:
    virtual ~Allocator() = default;

    virtual double* allocate(size_t size) = 0;           // 分配 size 个 double
    virtual void deallocate(double* ptr, size_t size) = 0; // 释放

    // 统计
    virtual size_t total_allocated() const = 0;   // 当前分配量
    virtual size_t peak_allocated() const = 0;    // 峰值分配量
    virtual size_t num_allocations() const = 0;   // 分配次数

    virtual std::string name() const = 0;
};
```

### 9.5.3 DefaultAllocator：带统计的 malloc/free

```cpp
class DefaultAllocator : public Allocator {
public:
    double* allocate(size_t size) override {
        if (size == 0) return nullptr;
        double* ptr = new double[size]();
        current_ += size;
        peak_ = std::max(peak_.load(), current_.load());
        ++num_allocs_;
        return ptr;
    }

    void deallocate(double* ptr, size_t size) override {
        if (!ptr) return;
        delete[] ptr;
        current_ -= size;
    }

    size_t total_allocated() const override { return current_; }
    size_t peak_allocated() const override { return peak_; }
    size_t num_allocations() const override { return num_allocs_; }
    std::string name() const override { return "DefaultAllocator"; }

private:
    std::atomic<size_t> current_{0};
    std::atomic<size_t> peak_{0};
    std::atomic<size_t> num_allocs_{0};
};
```

`std::atomic<size_t>` 保证多线程下统计正确。`peak_ = std::max(peak_.load(), current_.load())` 不是原子操作，但峰值统计允许偶尔不精确。

### 9.5.4 PoolAllocator：内存池

```cpp
class PoolAllocator : public Allocator {
public:
    explicit PoolAllocator(size_t pool_threshold = 1024 * 1024)
        : threshold_(pool_threshold) {}

    double* allocate(size_t size) override {
        if (size == 0) return nullptr;
        std::lock_guard<std::mutex> lock(mutex_);

        // 1. 在空闲列表中找恰好匹配的块
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (it->second == size) {
                double* ptr = it->first;
                free_blocks_.erase(it);
                current_ += size;
                peak_ = std::max(peak_, current_);
                ++num_allocs_;
                ++pool_hits_;           // 命中！
                return ptr;             // 重用，不调 new
            }
        }

        // 2. 没找到，分配新内存
        double* ptr = new double[size]();
        current_ += size;
        peak_ = std::max(peak_, current_);
        ++num_allocs_;
        ++pool_misses_;          // 未命中
        return ptr;
    }

    void deallocate(double* ptr, size_t size) override {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        current_ -= size;

        // 池子未超限 → 放入空闲列表（不 delete）
        if (pooled_bytes_ + size <= threshold_) {
            free_blocks_.emplace_back(ptr, size);
            pooled_bytes_ += size;
        } else {
            delete[] ptr;        // 超限 → 真正释放
        }
    }

    size_t pool_hits() const { return pool_hits_; }
    size_t pool_misses() const { return pool_misses_; }
    size_t pooled_bytes() const { return pooled_bytes_; }

private:
    std::mutex mutex_;
    std::vector<std::pair<double*, size_t>> free_blocks_;  // 空闲块列表
    size_t current_ = 0, peak_ = 0, num_allocs_ = 0;
    size_t pooled_bytes_ = 0, pool_hits_ = 0, pool_misses_ = 0;
    size_t threshold_;    // 池子上限
};
```

**策略**：

- **allocate**：先在 `free_blocks_` 找大小恰好匹配的块，找到就重用（`pool_hits_++`），找不到才 `new`（`pool_misses_++`）。
- **deallocate**：如果池中总字节未超 `threshold_`，放入 `free_blocks_`（不 delete）；否则真正 `delete[]`。

**效果**：训练循环中反复创建/释放相同大小的张量（如每步的梯度），`pool_hits_` 远大于 `pool_misses_`，省掉大部分 `new`/`delete` 调用。

### 9.5.5 全局 Allocator 管理

```cpp
// c10/allocator.cpp
static std::shared_ptr<Allocator>& global_allocator_ref() {
    static std::shared_ptr<Allocator> instance = std::make_shared<DefaultAllocator>();
    return instance;
}

Allocator& get_global_allocator() { return *global_allocator_ref(); }
void set_global_allocator(std::shared_ptr<Allocator> alloc) {
    global_allocator_ref() = std::move(alloc);
}
```

`static` 局部变量保证线程安全初始化（C++11 保证）。`set_global_allocator` 允许运行时切换：

```python
# Python 端
from minitorch import _cpp_ext
_cpp_ext.set_global_allocator(_cpp_ext.PoolAllocator(1024 * 1024))
# 之后所有 Storage 分配都走内存池
```

### 9.5.6 Storage 改用 Allocator

```cpp
// c10/storage.cpp
Storage::Storage(size_t size) : data_(nullptr), size_(size) {
    if (size > 0) {
        data_ = get_global_allocator().allocate(size);  // ← 通过 allocator
    }
}

Storage::~Storage() {
    if (data_) {
        get_global_allocator().deallocate(data_, size_);  // ← 通过 allocator
    }
}
```

Storage 不再直接 `new`/`delete`，全部通过 `get_global_allocator()`。切换 allocator 时，Storage 代码一行不改。

### 9.5.7 与 c10::Allocator 对照

| 我们的 `Allocator` | 真实 `c10::Allocator` |
|--------------------|------------------------|
| `double* allocate(size_t)` | `DataPtr allocate(size_t)` |
| 只支持 double | 任意 dtype + device |
| `DefaultAllocator` / `PoolAllocator` | `CPUAllocator` / `CUDAAllocator` / `PinnedMemoryAllocator` |
| 全局 `set_global_allocator` | `c10::SetAllocator` per device |
| 无对齐保证 | 按 CUDA 对齐要求（256B） |
| 无 OOM 处理 | `OutOfMemoryError` + caching allocator 回收 |

PyTorch 的 `c10::CUDAAllocator` 更复杂：caching allocator 维护按大小分桶的空闲列表（类似 jemalloc），跨 stream 同步，OOM 时回收。但**接口一致**：`allocate`/`deallocate` + 统计。

---

## 9.6 Autograd Profiler

### 9.6.1 为什么需要 Profiler

训练大模型时，backward 耗时往往占 60-80%。我们需要知道：

- 每个 Node 执行多久？
- 哪个算子是瓶颈？
- backward 期间分配了多少内存？

PyTorch 提供 `torch.autograd.profiler`，我们在 C++ 层实现一个轻量版。

### 9.6.2 Profiler 设计

```cpp
// autograd/profiler.h
struct ProfileEvent {
    std::string node_name;       // "MulNode", "AddNode", ...
    double duration_us;          // 微秒
    size_t memory_before;        // 执行前已分配字节
    size_t memory_after;         // 执行后已分配字节
    int thread_id;               // 线程 ID
};

class Profiler {
public:
    void start();
    void stop();
    bool enabled() const;
    void record(const std::string& name, double us,
                size_t mem_before, size_t mem_after, int tid);
    const std::vector<ProfileEvent>& events() const;
private:
    bool enabled_ = false;
    std::vector<ProfileEvent> events_;
};

Profiler& get_global_profiler();
```

### 9.6.3 集成到 run_backward

在 `run_backward` 的每个 Node 执行前后记录：

```cpp
auto& profiler = get_global_profiler();
auto t0 = std::chrono::high_resolution_clock::now();
size_t mem_before = get_global_allocator().total_allocated();

// ... node->apply(grad) ...

auto t1 = std::chrono::high_resolution_clock::now();
double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
if (profiler.enabled())
    profiler.record(node->name, us, mem_before,
                    get_global_allocator().total_allocated(), 0);
```

### 9.6.4 Python 端使用

```python
from minitorch import _cpp_ext

_cpp_ext.profiler_start()
x = _cpp_ext.TensorImpl([1.0, 2.0, 3.0], [3], True)
y = _cpp_ext.autograd_sum(_cpp_ext.autograd_mul(x, x), -1, False)
y.backward(_cpp_ext.TensorImpl([1.0], [1], False))
_cpp_ext.profiler_stop()

for event in _cpp_ext.profiler_events():
    print(f"{event[0]}: {event[1]:.1f} us, "
          f"mem: {event[2]} → {event[3]} bytes")
```

### 9.6.5 与 PyTorch 对照

| 我们的实现 | 真实 PyTorch | 文件 |
|-----------|-------------|------|
| `Profiler` | `torch::autograd::Profiler` | `torch/csrc/autograd/profiler.cpp` |
| `ProfileEvent` | `profiler::Event` | 同上 |
| `profiler_start/stop` | `torch.autograd.profiler.profile()` | `torch/autograd/profiler.py` |

---

## 9.7 梯度钩子（Backward Hooks）

### 9.7.1 什么是梯度钩子

`register_hook(fn)` 在叶子张量上注册一个回调，backward 时梯度到达该张量后、累加之前调用 `fn(grad)`。钩子可以：

- **监控**：打印梯度范数、检测梯度爆炸/消失
- **修改**：梯度裁剪、自定义梯度变换
- **调试**：记录每步梯度分布

对应 PyTorch 的 `tensor.register_hook()`。

### 9.7.2 实现

在 `TensorImpl` 中添加钩子字段：

```cpp
// c10/tensor.h
using HookFn = std::function<TensorImplPtr(TensorImplPtr)>;
// ...
HookFn backward_hook_;     // 梯度钩子
void register_hook(HookFn fn) { backward_hook_ = std::move(fn); }
void clear_hook() { backward_hook_ = nullptr; }
HookFn backward_hook() const { return backward_hook_; }
```

在 `AccumulateGrad::apply` 中调用钩子：

```cpp
// autograd/node.cpp
std::vector<TensorImplPtr> AccumulateGrad::apply(TensorImplPtr grad) {
    // 调用 backward hook（如果注册了）
    if (variable->backward_hook()) {
        auto hooked = variable->backward_hook()(grad);
        if (hooked) grad = hooked;
    }
    // ... 累加梯度 ...
}
```

### 9.7.3 Python 端使用

```python
x = _cpp_ext.TensorImpl([1.0, 2.0, 3.0], [3], True)

# 注册钩子：梯度翻倍
x.register_hook(lambda g: _cpp_ext.autograd_mul(
    g, _cpp_ext.TensorImpl([2.0], [1], False).expand([3])
))

y = _cpp_ext.autograd_mul(x, x)  # dy/dx = 2x
y.backward(_cpp_ext.TensorImpl([1.0, 1.0, 1.0], [3], False))
# x.grad = 2x * 2 = 4x = [4, 8, 12]
```

### 9.7.4 pybind11 绑定

钩子函数从 Python 传入，需用 `py::gil_scoped_acquire` 确保 GIL：

```cpp
.def("register_hook", [](TensorImplPtr& t, py::object fn) {
    t->register_hook([fn](TensorImplPtr grad) -> TensorImplPtr {
        py::gil_scoped_acquire acquire;
        py::object result = fn(grad);
        if (result.is_none()) return nullptr;
        return result.cast<TensorImplPtr>();
    });
})
```

---

## 9.8 Anomaly Detection

### 9.8.1 什么是 Anomaly Detection

训练中出现 NaN/Inf 梯度时，默认不会报错——NaN 会静默传播，最终模型权重全部变 NaN，难以定位根因。

`anomaly_check` 在每次梯度生成时检测 NaN/Inf，立即抛异常，帮助定位产生 NaN 的那个 Node。

对应 PyTorch 的 `torch.autograd.detect_anomaly()`。

### 9.8.2 实现

在 `grad_mode.h` 中添加全局标志：

```cpp
inline thread_local bool anomaly_check_enabled = false;

inline bool is_anomaly_check_enabled() { return anomaly_check_enabled; }
inline void set_anomaly_check_enabled(bool v) { anomaly_check_enabled = v; }

class AnomalyGuard {
public:
    AnomalyGuard() : prev_(anomaly_check_enabled) { anomaly_check_enabled = true; }
    ~AnomalyGuard() { anomaly_check_enabled = prev_; }
private:
    bool prev_;
};
```

在 `autograd/engine.cpp` 中检测：

```cpp
static bool has_nan_or_inf(const TensorImplPtr& t) {
    if (!t) return false;
    auto data = t->to_vector();
    for (auto v : data) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

static void check_anomaly(const TensorImplPtr& grad) {
    if (is_anomaly_check_enabled() && has_nan_or_inf(grad)) {
        throw std::runtime_error("Anomaly detected: NaN or Inf in gradient");
    }
}
```

在 `run_backward` 中，每个梯度生成后调用 `check_anomaly(grad)`。

### 9.8.3 Python 端使用

```python
_cpp_ext.set_anomaly_check_enabled(True)

x = _cpp_ext.TensorImpl([0.0], [1], True)
y = _cpp_ext.autograd_sqrt(x)  # dy/dx = 1/(2*sqrt(0)) = inf
try:
    y.backward(_cpp_ext.TensorImpl([1.0], [1], False))
except RuntimeError as e:
    print(e)  # "Anomaly detected: NaN or Inf in gradient"
```

### 9.8.4 与 PyTorch 对照

PyTorch 的 `detect_anomaly()` 上下文管理器更复杂：它还会记录前向执行栈，在异常消息中显示产生 NaN 的 Python 代码行。我们的实现只检测+抛异常，不含栈追踪。

---

## 9.9 Gradient Checkpointing

### 9.9.1 用重计算换内存

深层网络训练时，前向产生大量中间激活（每层输出），backward 需要这些激活来计算梯度。激活全部保存在内存中，显存成为瓶颈。

**Gradient checkpointing** 的思路：前向时**不保存**中间激活，只保存输入；backward 时**重新执行前向**以重算激活，再计算梯度。牺牲一次前向计算换取大幅内存节省。

对应 `torch.utils.checkpoint.checkpoint`。

### 9.9.2 CheckpointNode 设计

```cpp
// autograd/checkpoint.h
using CheckpointFn = std::function<TensorImplPtr(std::vector<TensorImplPtr>)>;

class CheckpointNode : public Node {
public:
    CheckpointFn fn;                    // 用户函数
    std::vector<TensorImplPtr> inputs;  // 保存的输入

    CheckpointNode(CheckpointFn fn, std::vector<TensorImplPtr> inputs);
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override;
};
```

### 9.9.3 前向：NoGrad 执行

```cpp
TensorImplPtr checkpoint(CheckpointFn fn, std::vector<TensorImplPtr> inputs) {
    // 前向在 NoGrad 下执行（不建图、不保存中间激活）
    TensorImplPtr output;
    {
        NoGradGuard no_grad;
        output = fn(inputs);
    }

    // 创建 CheckpointNode 并接入主图
    auto node = std::make_shared<CheckpointNode>(std::move(fn), inputs);
    for (auto& inp : inputs) {
        if (inp->grad_fn()) {
            node->next_edges.push_back(inp->grad_fn());
        } else if (inp->requires_grad()) {
            node->next_edges.push_back(std::make_shared<AccumulateGrad>(inp));
        } else {
            node->next_edges.push_back(nullptr);
        }
    }
    output->set_grad_fn(node);
    return output;
}
```

### 9.9.4 Backward：重计算

```cpp
std::vector<TensorImplPtr> CheckpointNode::apply(TensorImplPtr grad) {
    EnableGradGuard enable_grad;  // 重计算需要建图

    // 创建 detached 副本（避免覆盖原始图）
    std::vector<TensorImplPtr> detached;
    for (const auto& inp : inputs) {
        auto d = make_tensor(inp->to_vector(), inp->shape(), inp->requires_grad());
        d->set_is_leaf(true);
        detached.push_back(d);
    }

    // 重执行前向 → 局部 backward
    auto output = fn(detached);
    output->backward(grad);

    // 返回各输入的梯度
    std::vector<TensorImplPtr> result;
    for (const auto& d : detached) {
        result.push_back(d->grad());
    }
    return result;
}
```

关键点：
1. **detached 副本**：重计算用输入的数据副本，不触碰原始图的 `grad_fn`，避免覆盖。
2. **EnableGradGuard**：重计算时需要 grad enabled 以建局部图；之后 `backward` 内部的 `NoGradGuard` 会关闭 grad。
3. **局部 backward**：在重计算的局部图上反向传播，得到输入梯度，返回给主 Engine 继续传播。

### 9.9.5 Python 端使用

```python
x = _cpp_ext.TensorImpl([1.0, 2.0, 3.0], [3], True)

def fn(inputs):
    v = inputs[0]
    return _cpp_ext.autograd_sum(_cpp_ext.autograd_mul(v, v), -1, False)

y = _cpp_ext.checkpoint(fn, [x])
y.backward(_cpp_ext.TensorImpl([1.0], [1], False))
# x.grad = 2x = [2, 4, 6]  ← 与不使用 checkpoint 结果一致
```

### 9.9.6 内存 vs 计算的权衡

| 方式 | 前向内存 | backward 内存 | backward 计算 |
|------|---------|-------------|--------------|
| 普通 autograd | 保存所有激活 | 释放激活 | 1× 前向 |
| checkpoint | 只保存输入 | 重算激活 | 2× 前向 |

对 N 层网络，每 k 层设一个 checkpoint，内存从 O(N) 降到 O(N/k)，计算量增加约 k/(k+1) 倍。

### 9.9.7 与 PyTorch 对照

| 我们的实现 | 真实 PyTorch | 文件 |
|-----------|-------------|------|
| `checkpoint()` | `torch.utils.checkpoint.checkpoint()` | `torch/utils/checkpoint.py` |
| `CheckpointNode` | `CheckpointFunction` | 同上 |
| detached 副本 | `tensor.detach()` | 同上 |

PyTorch 的实现还支持 `use_reentrant` 参数（两种模式）、多输出、CPU offload 等。我们的实现是 reentrant 模式的基础版。

---

## 9.10 与 PyTorch 对照

### 9.10.1 Autograd 对照

| 我们的实现 | 真实 PyTorch | 文件 |
|-----------|-------------|------|
| `Node` 虚函数 | `torch::autograd::Node` | `torch/csrc/autograd/function.h` |
| `AccumulateGrad` | `AccumulateGrad` | `torch/csrc/autograd/variable.cpp` |
| `NoGradGuard` | `at::AutoGradMode` | `c10/core/GradMode.h` |
| `run_backward` | `Engine::execute` | `torch/csrc/autograd/engine.cpp` |
| `run_backward_mt` | `Engine::execute_with_thread_pool` | 同上 |
| `thread_local grad_mode_enabled` | `c10::AutoGradMode::grad_mode` | `c10/core/GradMode.h` |
| `Profiler` | `torch::autograd::Profiler` | `torch/csrc/autograd/profiler.cpp` |
| `register_hook` | `Tensor::register_hook()` | `torch/csrc/autograd/variable.cpp` |
| `anomaly_check_enabled` | `detect_anomaly()` | `torch/autograd/anomaly_mode.py` |
| `checkpoint()` | `torch.utils.checkpoint.checkpoint()` | `torch/utils/checkpoint.py` |

### 9.10.2 Allocator 对照

| 我们的实现 | 真实 PyTorch | 文件 |
|-----------|-------------|------|
| `Allocator` 接口 | `c10::Allocator` | `c10/core/Allocator.h` |
| `DefaultAllocator` | `c10::CPUAllocator` | `c10/core/CPUAllocator.cpp` |
| `PoolAllocator` | `c10::cuda::CUDACachingAllocator` | `c10/cuda/CUDACachingAllocator.cpp` |
| `get_global_allocator` | `c10::GetAllocator(device)` | `c10/core/Allocator.h` |
| `Storage` 用 allocator | `c10::StorageImpl` 用 allocator | `c10/core/StorageImpl.h` |

---

## 9.11 关键测试解读

### 9.11.1 多线程 backward 正确性

```python
def test_backward_mt_matches_single_thread():
    x = mt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * x).sum()
    y.backward()
    grad_single = x.grad.clone()

    x2 = mt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y2 = (x2 * x2).sum()
    y2.backward_mt(num_threads=4)
    grad_mt = x2.grad.clone()

    assert grad_single.allclose(grad_mt)
```

验证多线程结果与单线程一致。

### 9.11.2 Double backward

```python
def test_double_backward():
    x = mt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * x
    y.backward(mt.Tensor([1.0, 1.0, 1.0]), create_graph=True)
    # x.grad = 2x = [2, 4, 6]
    assert x.grad.data == [2.0, 4.0, 6.0]

    x.grad.backward(mt.Tensor([1.0, 1.0, 1.0]))
    # d(2x)/dx = 2
    assert x.grad.data == [2.0, 2.0, 2.0]
```

### 9.11.3 Allocator 统计

```python
def test_allocator_stats():
    alloc = _cpp_ext.DefaultAllocator()
    _cpp_ext.set_global_allocator(alloc)
    t = _cpp_ext.TensorImpl([1.0, 2.0, 3.0], [3])
    assert alloc.num_allocations() >= 1
    assert alloc.total_allocated() >= 3
```

### 9.11.4 PoolAllocator 命中率

```python
def test_pool_allocator_reuse():
    alloc = _cpp_ext.PoolAllocator(1024 * 1024)
    _cpp_ext.set_global_allocator(alloc)
    for _ in range(100):
        t = _cpp_ext.TensorImpl([1.0] * 1000, [1000])
        # t 析构 → 内存回池
    assert alloc.pool_hits() > 0   # 有重用
```

### 9.11.5 Profiler 事件记录

```python
def test_profiler_events():
    _cpp_ext.profiler_start()
    x = _cpp_ext.TensorImpl([1.0, 2.0, 3.0], [3], True)
    y = _cpp_ext.autograd_sum(_cpp_ext.autograd_mul(x, x), -1, False)
    y.backward(_cpp_ext.TensorImpl([1.0], [1], False))
    _cpp_ext.profiler_stop()
    events = _cpp_ext.profiler_events()
    assert len(events) > 0
    # 每个 event: (name, duration_us, mem_before, mem_after, thread_id)
    assert all(e[1] >= 0 for e in events)  # duration >= 0
```

### 9.11.6 梯度钩子

```python
def test_hook_modify_grad():
    x = _cpp_ext.TensorImpl([1.0, 2.0, 3.0], [3], True)
    # 钩子使梯度翻倍
    x.register_hook(lambda g: _cpp_ext.autograd_mul(
        g, _cpp_ext.TensorImpl([2.0], [1], False).expand([3])
    ))
    y = _cpp_ext.autograd_mul(x, x)  # dy/dx = 2x
    y.backward(_cpp_ext.TensorImpl([1.0, 1.0, 1.0], [3], False))
    assert x.grad.to_vector() == [4.0, 8.0, 12.0]  # 2x * 2 = 4x
```

### 9.11.7 Anomaly Detection

```python
def test_anomaly_detect_inf_grad():
    _cpp_ext.set_anomaly_check_enabled(True)
    try:
        x = _cpp_ext.TensorImpl([0.0], [1], True)
        y = _cpp_ext.autograd_sqrt(x)  # dy/dx = 1/(2*sqrt(0)) = inf
        with pytest.raises(RuntimeError, match="Anomaly detected"):
            y.backward(_cpp_ext.TensorImpl([1.0], [1], False))
    finally:
        _cpp_ext.set_anomaly_check_enabled(False)
```

### 9.11.8 Gradient Checkpointing

```python
def test_checkpoint_vs_no_checkpoint():
    # 普通前向
    x1 = _cpp_ext.TensorImpl([0.5, 1.5, 2.0], [3], True)
    y1 = _cpp_ext.autograd_sum(_cpp_ext.autograd_mul(x1, x1), -1, False)
    y1.backward(_cpp_ext.TensorImpl([1.0], [1], False))

    # checkpoint 前向
    x2 = _cpp_ext.TensorImpl([0.5, 1.5, 2.0], [3], True)
    def fn(inputs):
        v = inputs[0]
        return _cpp_ext.autograd_sum(_cpp_ext.autograd_mul(v, v), -1, False)
    y2 = _cpp_ext.checkpoint(fn, [x2])
    y2.backward(_cpp_ext.TensorImpl([1.0], [1], False))

    assert x2.grad.to_vector() == x1.grad.to_vector()  # 结果一致
```

---

## 9.12 下一章预告

本章的 C++ 核心只跑在 CPU 上。下一章（Ch10）我们给它接上 **CUDA**——让算子在 GPU 上跑，并引入 **dispatcher** 机制：同一个算子名（`"add"`）按张量的 device 自动路由到 CPU kernel 或 CUDA kernel。

你会看到：

- CUDA 编程模型：thread / block / grid，`__global__` kernel，`cudaMalloc`/`cudaMemcpy`。
- dispatcher 的 dispatch table 设计：`op_name → (device → kernel)`。
- 一个 `add` kernel 怎么写、怎么 launch、怎么同步。
- CPU 和 CUDA 算子注册到同一张表，调用点零改动。
- 对照真实 PyTorch 的 `c10::DispatchTable` 和 `aten/native/cuda/`。

Ch10 的代码在本环境无 GPU 也能完整阅读——CUDA 源文件（`.cu`）已写好，配上 GPU 机器 + `-DMINITORCH_ENABLE_CUDA=ON` 即可编译运行。