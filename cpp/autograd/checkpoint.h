// Gradient Checkpointing（Ch9 C++ 高级特性）
//
// 对应 torch.utils.checkpoint.checkpoint。
// 前向时不保存中间激活，backward 时重新执行前向以重计算梯度。
// 用重计算换内存：适合深层网络/大激活场景。
//
// 设计要点：
//   - checkpoint(fn, inputs)：前向在 NoGrad 下执行，只保存输入
//   - CheckpointNode::apply(grad)：重执行前向（建图）→ 局部 backward → 返回输入梯度
//   - 重计算时用 detached 副本，避免覆盖原始图

#pragma once

#include "c10/tensor.h"
#include "autograd/node.h"
#include <functional>

namespace minitorch {

// 用户函数类型：接受一组张量，返回一个张量
using CheckpointFn = std::function<TensorImplPtr(std::vector<TensorImplPtr>)>;

// CheckpointNode：backward 时重执行前向
class CheckpointNode : public Node {
public:
    CheckpointFn fn;
    std::vector<TensorImplPtr> inputs;

    CheckpointNode(CheckpointFn fn, std::vector<TensorImplPtr> inputs);

    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override;
};

// checkpoint：前向在 NoGrad 下执行，返回带 CheckpointNode 的输出
TensorImplPtr checkpoint(CheckpointFn fn, std::vector<TensorImplPtr> inputs);

} // namespace minitorch