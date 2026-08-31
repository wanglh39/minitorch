// Engine：反向传播调度引擎（Ch8 C++ autograd）
//
// 对应阶段一的 src/minitorch/autograd/engine.py。
// 从输出 Node 出发，DFS 后序逆序做拓扑排序，按序逐个调用 backward_fn。
// 梯度沿 next_edges 传给前驱，多后继梯度求和。
// AccumulateGrad 累加到 variable.grad。
// 对应真实 PyTorch 的 torch/csrc/autograd/engine.cpp。

#pragma once

#include "autograd/node.h"

namespace minitorch {

void run_backward(NodePtr root,
                   TensorImplPtr root_grad,
                   bool retain_graph = false,
                   bool retain_grad = false,
                   bool create_graph = false);

// 多线程版反向传播（Ch8 第二批③）
// 用 ThreadPool 并行执行无依赖的 Node。
// num_threads <= 0 时自动取 hardware_concurrency。
void run_backward_mt(NodePtr root,
                     TensorImplPtr root_grad,
                     bool retain_graph = false,
                     bool retain_grad = false,
                     int num_threads = 0,
                     bool create_graph = false);

} // namespace minitorch