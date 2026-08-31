// 算子声明（Ch8 C++ 核心重写）
//
// 对应阶段一的 src/minitorch/ops/arithmetic.py。
// 纯计算，不建图（autograd 在 autograd/ 中）。

#pragma once

#include "c10/tensor.h"
#include <vector>

namespace minitorch::ops {

// ── 逐元素二元算子 ────────────────────────────
TensorImplPtr add(const TensorImplPtr& a, const TensorImplPtr& b);

// 原地加法：target += source（要求同 shape，不做广播）
// 若 target 的 storage 被共享（use_count > 1），先 contiguous 拷贝再原地加。
// 用于梯度累加，避免每次分配新 TensorImpl + Storage。
void add_inplace(TensorImplPtr& target, const TensorImplPtr& source);
void sub_inplace(TensorImplPtr& target, const TensorImplPtr& source);
void mul_inplace(TensorImplPtr& target, const TensorImplPtr& source);
void div_inplace(TensorImplPtr& target, const TensorImplPtr& source);
TensorImplPtr sub(const TensorImplPtr& a, const TensorImplPtr& b);
TensorImplPtr mul(const TensorImplPtr& a, const TensorImplPtr& b);
TensorImplPtr div(const TensorImplPtr& a, const TensorImplPtr& b);

// ── 一元算子 ────────────────────────────────
TensorImplPtr neg(const TensorImplPtr& a);
TensorImplPtr relu(const TensorImplPtr& a);

// ── 逐元素数学函数 ──────────────────────────
TensorImplPtr exp(const TensorImplPtr& a);
TensorImplPtr log(const TensorImplPtr& a);
TensorImplPtr sqrt(const TensorImplPtr& a);
TensorImplPtr abs_val(const TensorImplPtr& a);
TensorImplPtr pow_scalar(const TensorImplPtr& a, double exponent);
TensorImplPtr clamp(const TensorImplPtr& a, double min_val, double max_val);
TensorImplPtr sigmoid(const TensorImplPtr& a);
TensorImplPtr tanh(const TensorImplPtr& a);

// ── 归约 ────────────────────────────────────
TensorImplPtr sum(const TensorImplPtr& a, int64_t dim = -1, bool keepdim = false);
TensorImplPtr mean(const TensorImplPtr& a, int64_t dim = -1, bool keepdim = false);

// ── 矩阵乘法 ────────────────────────────────
TensorImplPtr matmul(const TensorImplPtr& a, const TensorImplPtr& b);

// ── 广播辅助 ────────────────────────────────
std::pair<TensorImplPtr, TensorImplPtr> broadcast_tensors(
    const TensorImplPtr& a, const TensorImplPtr& b);

TensorImplPtr broadcast_to(const TensorImplPtr& a,
                            const std::vector<int64_t>& target_shape);

// ── 广播反向：reduce grad 到原 shape ──────────
TensorImplPtr reduce_grad(const TensorImplPtr& grad,
                          const std::vector<int64_t>& target_shape);

// ── 逐元素比较（用于 relu backward）────────────
TensorImplPtr greater_than_zero(const TensorImplPtr& a);

// ── 损失函数前向（不建图）──────────────────────
TensorImplPtr log_softmax(const TensorImplPtr& a, int64_t dim = -1);
TensorImplPtr softmax(const TensorImplPtr& a, int64_t dim = -1);
TensorImplPtr nll_loss(const TensorImplPtr& log_probs, const TensorImplPtr& target);

// ── 比较算子（返回 1.0/0.0，不可微）────────────
TensorImplPtr gt(const TensorImplPtr& a, const TensorImplPtr& b);
TensorImplPtr lt(const TensorImplPtr& a, const TensorImplPtr& b);
TensorImplPtr eq(const TensorImplPtr& a, const TensorImplPtr& b);
TensorImplPtr ge(const TensorImplPtr& a, const TensorImplPtr& b);
TensorImplPtr le(const TensorImplPtr& a, const TensorImplPtr& b);

// ── 归约算子 ──────────────────────────────────
TensorImplPtr max(const TensorImplPtr& a, int64_t dim = -1, bool keepdim = false);
TensorImplPtr min(const TensorImplPtr& a, int64_t dim = -1, bool keepdim = false);
TensorImplPtr argmax(const TensorImplPtr& a, int64_t dim = -1, bool keepdim = false);

} // namespace minitorch::ops