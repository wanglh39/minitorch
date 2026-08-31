// Node：计算图节点（Ch8 C++ autograd）
//
// 对应阶段一的 src/minitorch/autograd/function.py 中的 Node/AccumulateGrad。
// Node 基类用虚函数 apply() 替代 std::function，避免堆分配开销。
// 每个算子定义自己的 Node 子类（AddNode, MulNode 等）。
// 对应真实 PyTorch 的 torch/csrc/autograd/function.h。

#pragma once

#include "c10/tensor.h"
#include <vector>
#include <string>
#include <memory>

namespace minitorch {

class Node {
public:
    std::vector<NodePtr> next_edges;
    std::string name;
    TensorImplPtr output;

    Node() = default;
    virtual ~Node() = default;

    virtual std::vector<TensorImplPtr> apply(TensorImplPtr grad) = 0;

    bool is_accumulate_grad() const { return name == "AccumulateGrad"; }
};

class AccumulateGrad : public Node {
public:
    TensorImplPtr variable;

    explicit AccumulateGrad(TensorImplPtr var);
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override;
};

} // namespace minitorch
