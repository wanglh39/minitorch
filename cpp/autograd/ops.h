// AutogradOps：建图算子（Ch8 C++ autograd）
//
// 对应阶段一的 src/minitorch/ops/arithmetic.py。
// 每个函数先调用 ops:: 做前向计算，再在 requires_grad 时构建 Node 挂到输出.grad_fn。
// backward 用 ops:: 底层方法避免递归建图。

#pragma once

#include "c10/tensor.h"

namespace minitorch::autograd {

TensorImplPtr add(const TensorImplPtr& a, const TensorImplPtr& b);
TensorImplPtr sub(const TensorImplPtr& a, const TensorImplPtr& b);
TensorImplPtr mul(const TensorImplPtr& a, const TensorImplPtr& b);
TensorImplPtr div(const TensorImplPtr& a, const TensorImplPtr& b);
TensorImplPtr neg(const TensorImplPtr& a);
TensorImplPtr relu(const TensorImplPtr& a);
TensorImplPtr matmul(const TensorImplPtr& a, const TensorImplPtr& b);
TensorImplPtr sum(const TensorImplPtr& a, int64_t dim = -1, bool keepdim = false);
TensorImplPtr mean(const TensorImplPtr& a, int64_t dim = -1, bool keepdim = false);
TensorImplPtr transpose(const TensorImplPtr& a, int64_t dim0 = 1, int64_t dim1 = 0);

// ── 逐元素数学函数 ──────────────────────────
TensorImplPtr exp(const TensorImplPtr& a);
TensorImplPtr log(const TensorImplPtr& a);
TensorImplPtr sqrt(const TensorImplPtr& a);
TensorImplPtr abs_val(const TensorImplPtr& a);
TensorImplPtr pow_scalar(const TensorImplPtr& a, double exponent);
TensorImplPtr clamp(const TensorImplPtr& a, double min_val, double max_val);
TensorImplPtr sigmoid(const TensorImplPtr& a);
TensorImplPtr tanh(const TensorImplPtr& a);

// ── 损失函数 ──────────────────────────────────
TensorImplPtr log_softmax(const TensorImplPtr& a, int64_t dim = -1);
TensorImplPtr softmax(const TensorImplPtr& a, int64_t dim = -1);
TensorImplPtr nll_loss(const TensorImplPtr& log_probs, const TensorImplPtr& target);
TensorImplPtr cross_entropy(const TensorImplPtr& logits, const TensorImplPtr& target, int64_t dim = -1);
TensorImplPtr mse_loss(const TensorImplPtr& pred, const TensorImplPtr& target);

// ── 归约算子（可微）────────────────────────────
TensorImplPtr max(const TensorImplPtr& a, int64_t dim = -1, bool keepdim = false);
TensorImplPtr min(const TensorImplPtr& a, int64_t dim = -1, bool keepdim = false);

} // namespace minitorch::autograd