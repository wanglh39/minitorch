// Gradient Checkpointing 实现（Ch9 C++ 高级特性）

#include "autograd/checkpoint.h"
#include "autograd/grad_mode.h"
#include "autograd/engine.h"
#include "autograd/ops.h"

namespace minitorch {

CheckpointNode::CheckpointNode(CheckpointFn fn, std::vector<TensorImplPtr> inputs)
    : fn(std::move(fn)), inputs(std::move(inputs)) {
    name = "CheckpointNode";
}

std::vector<TensorImplPtr> CheckpointNode::apply(TensorImplPtr grad) {
    // 重执行前向：需要 grad enabled 以重建局部计算图
    EnableGradGuard enable_grad;

    // 创建 detached 副本：避免覆盖原始图的 grad_fn
    std::vector<TensorImplPtr> detached;
    detached.reserve(inputs.size());
    for (const auto& inp : inputs) {
        auto d = make_tensor(inp->to_vector(), inp->shape(), inp->requires_grad());
        d->set_is_leaf(true);
        detached.push_back(d);
    }

    // 重计算输出
    auto output = fn(detached);
    if (!output || !output->grad_fn()) return {};

    // 局部 backward：在重计算的图上反向传播
    if (!grad) {
        grad = make_tensor({1.0}, {});
    }
    output->backward(grad, /*retain_graph=*/false,
                     /*retain_grad=*/false, /*create_graph=*/false);

    // 收集各输入的梯度
    std::vector<TensorImplPtr> result;
    result.reserve(detached.size());
    for (const auto& d : detached) {
        result.push_back(d->grad());
    }
    return result;
}

TensorImplPtr checkpoint(CheckpointFn fn, std::vector<TensorImplPtr> inputs) {
    // 检查是否有输入需要梯度
    bool any_requires_grad = false;
    for (const auto& inp : inputs) {
        if (inp->requires_grad()) {
            any_requires_grad = true;
            break;
        }
    }

    // 前向在 NoGrad 下执行（不建图、不保存中间激活）
    TensorImplPtr output;
    {
        NoGradGuard no_grad;
        output = fn(inputs);
    }

    if (!any_requires_grad) return output;

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
    output->set_is_leaf(false);
    output->set_requires_grad(true);

    return output;
}

} // namespace minitorch