// 算子实现（Ch8）

#include "aten/ops.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace minitorch::ops {

// ── 广播辅助 ─────────────────────────────────────────

TensorImplPtr broadcast_to(const TensorImplPtr& a,
                            const std::vector<int64_t>& target_shape) {
    const auto& src_shape = a->shape();
    int64_t src_ndim = static_cast<int64_t>(src_shape.size());
    int64_t dst_ndim = static_cast<int64_t>(target_shape.size());

    if (dst_ndim < src_ndim) {
        throw std::runtime_error("broadcast_to: 目标维度不能少于自身");
    }

    // 左填充
    int64_t pad = dst_ndim - src_ndim;
    std::vector<int64_t> padded_shape(dst_ndim, 1);
    std::vector<int64_t> padded_strides(dst_ndim, 0);
    for (int64_t i = 0; i < src_ndim; ++i) {
        padded_shape[pad + i] = src_shape[i];
        padded_strides[pad + i] = a->strides()[i];
    }

    // 计算新 stride
    std::vector<int64_t> new_strides(dst_ndim);
    for (int64_t i = 0; i < dst_ndim; ++i) {
        if (padded_shape[i] == target_shape[i]) {
            new_strides[i] = padded_strides[i];
        } else if (padded_shape[i] == 1) {
            new_strides[i] = 0;  // 广播维度 stride=0
        } else {
            throw std::runtime_error("broadcast_to: shape 不兼容");
        }
    }

    auto result = std::make_shared<TensorImpl>(a->storage(), target_shape, new_strides,
                                               a->storage_offset(), a->requires_grad());
    result->set_grad_fn(a->grad_fn());
    result->set_is_leaf(a->is_leaf());
    return result;
}

std::pair<TensorImplPtr, TensorImplPtr> broadcast_tensors(
    const TensorImplPtr& a, const TensorImplPtr& b) {
    auto out_shape = broadcast_shapes(a->shape(), b->shape());
    return {broadcast_to(a, out_shape), broadcast_to(b, out_shape)};
}

// ── 逐元素二元算子 ────────────────────────────────────

// 通用二元算子模板
template <typename Op>
TensorImplPtr binary_op(const TensorImplPtr& a, const TensorImplPtr& b, Op op) {
    auto [ba, bb] = broadcast_tensors(a, b);
    const auto& shape = ba->shape();
    int64_t n = ba->numel();

    std::vector<double> result(static_cast<size_t>(n));
    std::vector<int64_t> indices(ba->ndim(), 0);

    for (int64_t i = 0; i < n; ++i) {
        int64_t off_a = ba->linear_offset(indices);
        int64_t off_b = bb->linear_offset(indices);
        result[static_cast<size_t>(i)] = op(
            ba->storage()->data()[static_cast<size_t>(off_a)],
            bb->storage()->data()[static_cast<size_t>(off_b)]
        );
        // 递增索引
        for (int64_t d = ba->ndim() - 1; d >= 0; --d) {
            if (++indices[d] < shape[d]) break;
            indices[d] = 0;
        }
    }

    return make_tensor(result, shape);
}

TensorImplPtr add(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x + y; });
}

void add_inplace(TensorImplPtr& target, const TensorImplPtr& source) {
    if (!target) {
        target = source;
        return;
    }
    if (!source) return;

    const auto& t_shape = target->shape();
    const auto& s_shape = source->shape();
    if (t_shape != s_shape) {
        target = add(target, source);
        return;
    }

    if (target->storage().use_count() > 1 || !target->is_contiguous()) {
        target = target->contiguous();
    }

    int64_t n = target->numel();
    double* t_data = target->storage()->data();

    if (source->is_contiguous() && source->storage_offset() == 0) {
        const double* s_data = source->storage()->data();
        for (int64_t i = 0; i < n; ++i) {
            t_data[static_cast<size_t>(i)] += s_data[static_cast<size_t>(i)];
        }
    } else {
        std::vector<int64_t> indices(target->ndim(), 0);
        for (int64_t i = 0; i < n; ++i) {
            int64_t off_s = source->linear_offset(indices);
            t_data[static_cast<size_t>(i)] +=
                source->storage()->data()[static_cast<size_t>(off_s)];
            for (int64_t d = target->ndim() - 1; d >= 0; --d) {
                if (++indices[d] < t_shape[static_cast<size_t>(d)]) break;
                indices[d] = 0;
            }
        }
    }
}

// 通用原地二元算子模板
template <typename ScalarOp, typename TensorOp>
void inplace_op(TensorImplPtr& target, const TensorImplPtr& source,
                ScalarOp scalar_op, TensorOp tensor_op) {
    if (!target || !source) return;
    const auto& t_shape = target->shape();
    const auto& s_shape = source->shape();
    if (t_shape != s_shape) {
        target = tensor_op(target, source);
        return;
    }
    if (target->storage().use_count() > 1 || !target->is_contiguous()) {
        target = target->contiguous();
    }
    int64_t n = target->numel();
    double* t_data = target->storage()->data();
    if (source->is_contiguous() && source->storage_offset() == 0) {
        const double* s_data = source->storage()->data();
        for (int64_t i = 0; i < n; ++i)
            t_data[static_cast<size_t>(i)] = scalar_op(t_data[static_cast<size_t>(i)], s_data[static_cast<size_t>(i)]);
    } else {
        std::vector<int64_t> indices(target->ndim(), 0);
        for (int64_t i = 0; i < n; ++i) {
            int64_t off_s = source->linear_offset(indices);
            t_data[static_cast<size_t>(i)] = scalar_op(t_data[static_cast<size_t>(i)],
                source->storage()->data()[static_cast<size_t>(off_s)]);
            for (int64_t d = target->ndim() - 1; d >= 0; --d) {
                if (++indices[d] < t_shape[static_cast<size_t>(d)]) break;
                indices[d] = 0;
            }
        }
    }
}

void sub_inplace(TensorImplPtr& target, const TensorImplPtr& source) {
    inplace_op(target, source, [](double a, double b) { return a - b; }, sub);
}

void mul_inplace(TensorImplPtr& target, const TensorImplPtr& source) {
    inplace_op(target, source, [](double a, double b) { return a * b; }, mul);
}

void div_inplace(TensorImplPtr& target, const TensorImplPtr& source) {
    inplace_op(target, source, [](double a, double b) { return a / b; }, div);
}

TensorImplPtr sub(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x - y; });
}

TensorImplPtr mul(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x * y; });
}

TensorImplPtr div(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x / y; });
}

// ── 一元算子 ──────────────────────────────────────────

// 通用一元算子模板
template <typename Op>
TensorImplPtr unary_op(const TensorImplPtr& a, Op op) {
    int64_t n = a->numel();
    std::vector<double> result(static_cast<size_t>(n));
    std::vector<int64_t> indices(a->ndim(), 0);
    for (int64_t i = 0; i < n; ++i) {
        result[static_cast<size_t>(i)] = op(
            a->storage()->data()[static_cast<size_t>(a->linear_offset(indices))]
        );
        for (int64_t d = a->ndim() - 1; d >= 0; --d) {
            if (++indices[d] < a->shape()[d]) break;
            indices[d] = 0;
        }
    }
    return make_tensor(result, a->shape());
}

TensorImplPtr neg(const TensorImplPtr& a) {
    return unary_op(a, [](double x) { return -x; });
}

TensorImplPtr relu(const TensorImplPtr& a) {
    return unary_op(a, [](double x) { return std::max(0.0, x); });
}

// ── 逐元素数学函数 ─────────────────────────────────────

TensorImplPtr exp(const TensorImplPtr& a) {
    return unary_op(a, [](double x) { return std::exp(x); });
}

TensorImplPtr log(const TensorImplPtr& a) {
    return unary_op(a, [](double x) { return std::log(x); });
}

TensorImplPtr sqrt(const TensorImplPtr& a) {
    return unary_op(a, [](double x) { return std::sqrt(x); });
}

TensorImplPtr abs_val(const TensorImplPtr& a) {
    return unary_op(a, [](double x) { return std::fabs(x); });
}

TensorImplPtr pow_scalar(const TensorImplPtr& a, double exponent) {
    return unary_op(a, [exponent](double x) { return std::pow(x, exponent); });
}

TensorImplPtr clamp(const TensorImplPtr& a, double min_val, double max_val) {
    return unary_op(a, [min_val, max_val](double x) {
        return std::max(min_val, std::min(max_val, x));
    });
}

TensorImplPtr sigmoid(const TensorImplPtr& a) {
    return unary_op(a, [](double x) {
        if (x >= 0) {
            double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        } else {
            double z = std::exp(x);
            return z / (1.0 + z);
        }
    });
}

TensorImplPtr tanh(const TensorImplPtr& a) {
    return unary_op(a, [](double x) { return std::tanh(x); });
}

// ── 归约 ──────────────────────────────────────────────

TensorImplPtr sum(const TensorImplPtr& a, int64_t dim, bool keepdim) {
    if (dim < 0) {
        // 全局 sum
        int64_t n = a->numel();
        double s = 0;
        std::vector<int64_t> indices(a->ndim(), 0);
        for (int64_t i = 0; i < n; ++i) {
            s += a->storage()->data()[static_cast<size_t>(a->linear_offset(indices))];
            for (int64_t d = a->ndim() - 1; d >= 0; --d) {
                if (++indices[d] < a->shape()[d]) break;
                indices[d] = 0;
            }
        }
        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(a->ndim(), 1) : std::vector<int64_t>{};
        return make_tensor({s}, out_shape);
    }

    // 沿 dim sum
    int64_t ndim = a->ndim();
    if (dim < 0) dim += ndim;
    const auto& shape = a->shape();

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }

    int64_t out_numel = 1;
    for (auto d : out_shape) out_numel *= d;
    std::vector<double> result(static_cast<size_t>(out_numel), 0.0);

    std::vector<int64_t> indices(ndim, 0);
    for (int64_t i = 0; i < a->numel(); ++i) {
        // 计算 out 索引
        std::vector<int64_t> out_idx;
        for (int64_t d = 0; d < ndim; ++d) {
            if (d == dim) {
                if (keepdim) out_idx.push_back(0);
            } else {
                out_idx.push_back(indices[d]);
            }
        }
        // out_idx -> flat offset
        int64_t out_off = 0;
        for (size_t d = 0; d < out_idx.size(); ++d) {
            int64_t stride = 1;
            for (size_t e = d + 1; e < out_idx.size(); ++e) stride *= out_shape[e];
            out_off += out_idx[d] * stride;
        }
        result[static_cast<size_t>(out_off)] += a->storage()->data()[static_cast<size_t>(a->linear_offset(indices))];
        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (++indices[d] < shape[d]) break;
            indices[d] = 0;
        }
    }
    return make_tensor(result, out_shape);
}

TensorImplPtr mean(const TensorImplPtr& a, int64_t dim, bool keepdim) {
    auto s = sum(a, dim, keepdim);
    int64_t n = a->numel();
    std::vector<double> data = s->to_vector();
    for (auto& v : data) v /= static_cast<double>(n);
    return make_tensor(data, s->shape());
}

// ── 矩阵乘法 ──────────────────────────────────────────

TensorImplPtr matmul(const TensorImplPtr& a, const TensorImplPtr& b) {
    const auto& sa = a->shape();
    const auto& sb = b->shape();

    // 1D @ 1D → scalar (dot product)
    if (sa.size() == 1 && sb.size() == 1) {
        if (sa[0] != sb[0]) throw std::runtime_error("matmul: shape 不兼容");
        double dot = 0;
        auto a_data = a->to_vector();
        auto b_data = b->to_vector();
        for (size_t i = 0; i < a_data.size(); ++i) {
            dot += a_data[i] * b_data[i];
        }
        return make_tensor({dot}, {});
    }

    // 归一化 1D 输入：a → [1,K]，b → [K,1]
    bool a_was_1d = (sa.size() == 1);
    bool b_was_1d = (sb.size() == 1);

    TensorImplPtr a_norm = a_was_1d ? a->reshape({1, sa[0]}) : a;
    TensorImplPtr b_norm = b_was_1d ? b->reshape({sb[0], 1}) : b;

    const auto& na = a_norm->shape();
    const auto& nb = b_norm->shape();

    int64_t M = na[na.size() - 2];
    int64_t K = na[na.size() - 1];
    int64_t N = nb[nb.size() - 1];

    if (K != nb[nb.size() - 2]) {
        throw std::runtime_error("matmul: shape 不兼容");
    }

    // 批量维度广播
    std::vector<int64_t> a_batch(na.begin(), na.end() - 2);
    std::vector<int64_t> b_batch(nb.begin(), nb.end() - 2);
    std::vector<int64_t> batch_shape = broadcast_shapes(a_batch, b_batch);

    std::vector<int64_t> a_target = batch_shape;
    a_target.push_back(M);
    a_target.push_back(K);
    std::vector<int64_t> b_target = batch_shape;
    b_target.push_back(K);
    b_target.push_back(N);

    auto a_bc = broadcast_to(a_norm, a_target);
    auto b_bc = broadcast_to(b_norm, b_target);

    int64_t batch_numel = 1;
    for (auto d : batch_shape) batch_numel *= d;

    std::vector<double> result(static_cast<size_t>(batch_numel * M * N), 0.0);
    auto a_data = a_bc->to_vector();
    auto b_data = b_bc->to_vector();

    for (int64_t batch = 0; batch < batch_numel; ++batch) {
        int64_t a_off = batch * M * K;
        int64_t b_off = batch * K * N;
        int64_t r_off = batch * M * N;
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                double s = 0;
                for (int64_t k = 0; k < K; ++k) {
                    s += a_data[static_cast<size_t>(a_off + i * K + k)] *
                         b_data[static_cast<size_t>(b_off + k * N + j)];
                }
                result[static_cast<size_t>(r_off + i * N + j)] = s;
            }
        }
    }

    // 输出 shape = batch + [M, N]，squeeze 1D 归一化
    std::vector<int64_t> out_shape = batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);

    if (a_was_1d) {
        out_shape.erase(out_shape.begin() + static_cast<long>(batch_shape.size()));
    }
    if (b_was_1d) {
        out_shape.pop_back();
    }

    return make_tensor(result, out_shape);
}

// ── 广播反向 ──────────────────────────────────────────

TensorImplPtr reduce_grad(const TensorImplPtr& grad,
                          const std::vector<int64_t>& target_shape) {
    auto g = grad;
    // 消除多余前导维度
    while (g->ndim() > static_cast<int64_t>(target_shape.size())) {
        g = sum(g, 0, false);
    }
    // 广播维度求和
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (g->shape()[i] != target_shape[i] && target_shape[i] == 1) {
            g = sum(g, static_cast<int64_t>(i), true);
        }
    }
    return g;
}

TensorImplPtr greater_than_zero(const TensorImplPtr& a) {
    return unary_op(a, [](double x) { return (x > 0.0) ? 1.0 : 0.0; });
}

// ── 损失函数前向 ─────────────────────────────────────

TensorImplPtr log_softmax(const TensorImplPtr& a, int64_t dim) {
    int64_t ndim = a->ndim();
    if (dim < 0) dim += ndim;
    const auto& shape = a->shape();
    int64_t dim_size = shape[static_cast<size_t>(dim)];

    // 计算不含 dim 的形状大小
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[static_cast<size_t>(i)];
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[static_cast<size_t>(i)];

    auto a_data = a->to_vector();
    std::vector<double> result(a_data.size());

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            // 求 max
            double max_val = -1e30;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (o * dim_size + d) * inner + in;
                max_val = std::max(max_val, a_data[static_cast<size_t>(idx)]);
            }
            // shifted + log_sum_exp
            double log_sum_exp = 0;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (o * dim_size + d) * inner + in;
                log_sum_exp += std::exp(a_data[static_cast<size_t>(idx)] - max_val);
            }
            log_sum_exp = std::log(log_sum_exp);
            // result = shifted - log_sum_exp
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (o * dim_size + d) * inner + in;
                result[static_cast<size_t>(idx)] =
                    a_data[static_cast<size_t>(idx)] - max_val - log_sum_exp;
            }
        }
    }
    return make_tensor(result, shape);
}

TensorImplPtr softmax(const TensorImplPtr& a, int64_t dim) {
    auto lp = log_softmax(a, dim);
    return exp(lp);
}

TensorImplPtr nll_loss(const TensorImplPtr& log_probs, const TensorImplPtr& target) {
    // log_probs: [N, C], target: [N] (integer)
    int64_t n = log_probs->shape()[0];
    int64_t c = log_probs->shape()[1];
    auto lp_data = log_probs->to_vector();
    auto tgt_data = target->to_vector();

    double loss = 0;
    for (int64_t i = 0; i < n; ++i) {
        int64_t t = static_cast<int64_t>(tgt_data[static_cast<size_t>(i)]);
        loss -= lp_data[static_cast<size_t>(i * c + t)];
    }
    loss /= n;
    return make_tensor({loss}, {});
}

// ── 比较算子 ──────────────────────────────────────────

TensorImplPtr gt(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x > y ? 1.0 : 0.0; });
}

TensorImplPtr lt(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x < y ? 1.0 : 0.0; });
}

TensorImplPtr eq(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x == y ? 1.0 : 0.0; });
}

TensorImplPtr ge(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x >= y ? 1.0 : 0.0; });
}

TensorImplPtr le(const TensorImplPtr& a, const TensorImplPtr& b) {
    return binary_op(a, b, [](double x, double y) { return x <= y ? 1.0 : 0.0; });
}

// ── 归约算子 ──────────────────────────────────────────

TensorImplPtr max(const TensorImplPtr& a, int64_t dim, bool keepdim) {
    if (dim < 0) {
        // 全局 max
        auto data = a->to_vector();
        double m = data[0];
        for (auto v : data) m = std::max(m, v);
        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(a->ndim(), 1) : std::vector<int64_t>{};
        return make_tensor({m}, out_shape);
    }

    int64_t ndim = a->ndim();
    const auto& shape = a->shape();
    int64_t dim_size = shape[static_cast<size_t>(dim)];

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape[i]);
    }

    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];

    auto a_data = a->to_vector();
    int64_t out_numel = outer * inner;
    std::vector<double> result(static_cast<size_t>(out_numel));

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            double m = -1e30;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (o * dim_size + d) * inner + in;
                m = std::max(m, a_data[static_cast<size_t>(idx)]);
            }
            result[static_cast<size_t>(o * inner + in)] = m;
        }
    }
    return make_tensor(result, out_shape);
}

TensorImplPtr min(const TensorImplPtr& a, int64_t dim, bool keepdim) {
    if (dim < 0) {
        auto data = a->to_vector();
        double m = data[0];
        for (auto v : data) m = std::min(m, v);
        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(a->ndim(), 1) : std::vector<int64_t>{};
        return make_tensor({m}, out_shape);
    }

    int64_t ndim = a->ndim();
    const auto& shape = a->shape();
    int64_t dim_size = shape[static_cast<size_t>(dim)];

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape[i]);
    }

    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];

    auto a_data = a->to_vector();
    int64_t out_numel = outer * inner;
    std::vector<double> result(static_cast<size_t>(out_numel));

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            double m = 1e30;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (o * dim_size + d) * inner + in;
                m = std::min(m, a_data[static_cast<size_t>(idx)]);
            }
            result[static_cast<size_t>(o * inner + in)] = m;
        }
    }
    return make_tensor(result, out_shape);
}

TensorImplPtr argmax(const TensorImplPtr& a, int64_t dim, bool keepdim) {
    if (dim < 0) {
        auto data = a->to_vector();
        int64_t best = 0;
        for (size_t i = 1; i < data.size(); ++i)
            if (data[i] > data[static_cast<size_t>(best)]) best = static_cast<int64_t>(i);
        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(a->ndim(), 1) : std::vector<int64_t>{};
        return make_tensor({static_cast<double>(best)}, out_shape);
    }

    int64_t ndim = a->ndim();
    const auto& shape = a->shape();
    int64_t dim_size = shape[static_cast<size_t>(dim)];

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape[i]);
    }

    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];

    auto a_data = a->to_vector();
    int64_t out_numel = outer * inner;
    std::vector<double> result(static_cast<size_t>(out_numel));

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            int64_t best = 0;
            double best_val = -1e30;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (o * dim_size + d) * inner + in;
                if (a_data[static_cast<size_t>(idx)] > best_val) {
                    best_val = a_data[static_cast<size_t>(idx)];
                    best = d;
                }
            }
            result[static_cast<size_t>(o * inner + in)] = static_cast<double>(best);
        }
    }
    return make_tensor(result, out_shape);
}

} // namespace minitorch::ops