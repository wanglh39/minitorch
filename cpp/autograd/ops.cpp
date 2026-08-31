// AutogradOps 实现（Ch8 C++ autograd）
//
// 每个算子定义自己的 Node 子类，重写 apply()。
// 替代旧的 std::function lambda 方案，避免堆分配 + 类型擦除开销。

#include "autograd/ops.h"
#include "autograd/node.h"
#include "autograd/grad_mode.h"
#include "aten/ops.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace minitorch::autograd {

// ── 辅助：收集 next_edges ─────────────────────────────

static std::vector<NodePtr> collect_edges(const std::vector<TensorImplPtr>& inputs) {
    std::vector<NodePtr> edges;
    for (const auto& t : inputs) {
        if (t && t->requires_grad()) {
            if (t->grad_fn()) {
                edges.push_back(t->grad_fn());
            } else {
                edges.push_back(std::make_shared<AccumulateGrad>(t));
            }
        } else {
            edges.push_back(nullptr);
        }
    }
    return edges;
}

static bool any_requires_grad(const std::vector<TensorImplPtr>& inputs) {
    for (const auto& t : inputs) {
        if (t && t->requires_grad()) return true;
    }
    return false;
}

static void attach_node(const TensorImplPtr& result,
                        NodePtr node,
                        std::vector<NodePtr> edges) {
    node->next_edges = std::move(edges);
    node->output = result;
    result->set_grad_fn(node);
    result->set_requires_grad(true);
    result->set_is_leaf(false);
}

// ── Node 子类定义 ─────────────────────────────────────

class AddNode : public Node {
public:
    std::vector<int64_t> a_shape, b_shape;
    AddNode(std::vector<int64_t> a, std::vector<int64_t> b)
        : a_shape(std::move(a)), b_shape(std::move(b)) { name = "Add"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        return {ops::reduce_grad(grad, a_shape),
                ops::reduce_grad(grad, b_shape)};
    }
};

class SubNode : public Node {
public:
    std::vector<int64_t> a_shape, b_shape;
    SubNode(std::vector<int64_t> a, std::vector<int64_t> b)
        : a_shape(std::move(a)), b_shape(std::move(b)) { name = "Sub"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        auto neg_grad = ops::neg(grad);
        return {ops::reduce_grad(grad, a_shape),
                ops::reduce_grad(neg_grad, b_shape)};
    }
};

class MulNode : public Node {
public:
    std::vector<int64_t> a_shape, b_shape;
    TensorImplPtr orig_a, orig_b;
    MulNode(std::vector<int64_t> a, std::vector<int64_t> b,
            TensorImplPtr a_, TensorImplPtr b_)
        : a_shape(std::move(a)), b_shape(std::move(b)),
          orig_a(std::move(a_)), orig_b(std::move(b_)) { name = "Mul"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        auto grad_a = mul(grad, orig_b);
        auto grad_b = mul(grad, orig_a);
        return {ops::reduce_grad(grad_a, a_shape),
                ops::reduce_grad(grad_b, b_shape)};
    }
};

class DivNode : public Node {
public:
    std::vector<int64_t> a_shape, b_shape;
    TensorImplPtr orig_a, orig_b;
    DivNode(std::vector<int64_t> a, std::vector<int64_t> b,
            TensorImplPtr a_, TensorImplPtr b_)
        : a_shape(std::move(a)), b_shape(std::move(b)),
          orig_a(std::move(a_)), orig_b(std::move(b_)) { name = "Div"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        auto [ba, bb] = ops::broadcast_tensors(orig_a, orig_b);
        auto grad_a = ops::div(grad, bb);
        auto grad_b = ops::neg(ops::div(ops::mul(grad, ba),
                                        ops::mul(bb, bb)));
        return {ops::reduce_grad(grad_a, a_shape),
                ops::reduce_grad(grad_b, b_shape)};
    }
};

class NegNode : public Node {
public:
    NegNode() { name = "Neg"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        return {ops::neg(grad)};
    }
};

class ReluNode : public Node {
public:
    TensorImplPtr mask;
    explicit ReluNode(TensorImplPtr m) : mask(std::move(m)) { name = "Relu"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        return {ops::mul(grad, mask)};
    }
};

class MatmulNode : public Node {
public:
    TensorImplPtr a_copy, b_copy;
    bool a_was_1d, b_was_1d;
    MatmulNode(TensorImplPtr a, TensorImplPtr b, bool a1d, bool b1d)
        : a_copy(std::move(a)), b_copy(std::move(b)),
          a_was_1d(a1d), b_was_1d(b1d) { name = "Matmul"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        auto a_mat = a_was_1d ? a_copy->reshape({1, a_copy->shape()[0]}) : a_copy;
        auto b_mat = b_was_1d ? b_copy->reshape({b_copy->shape()[0], 1}) : b_copy;

        TensorImplPtr g = grad;
        if (g->ndim() == 0) {
            g = g->reshape({1, 1});
        } else if (g->ndim() == 1) {
            if (a_was_1d && !b_was_1d) {
                g = g->reshape({1, g->shape()[0]});
            } else if (!a_was_1d && b_was_1d) {
                g = g->reshape({g->shape()[0], 1});
            } else {
                g = g->reshape({1, 1});
            }
        }

        int64_t a_ndim = a_mat->ndim();
        int64_t b_ndim = b_mat->ndim();
        auto grad_a = ops::matmul(g, b_mat->transpose(b_ndim - 2, b_ndim - 1));
        auto grad_b = ops::matmul(a_mat->transpose(a_ndim - 2, a_ndim - 1), g);

        if (a_was_1d) {
            auto shape = grad_a->shape();
            shape.erase(shape.end() - 2);
            grad_a = grad_a->reshape(shape);
        }
        if (b_was_1d) {
            auto shape = grad_b->shape();
            shape.pop_back();
            grad_b = grad_b->reshape(shape);
        }

        return {grad_a, grad_b};
    }
};

class SumNode : public Node {
public:
    std::vector<int64_t> a_shape;
    int64_t dim;
    bool keepdim;
    SumNode(std::vector<int64_t> s, int64_t d, bool k)
        : a_shape(std::move(s)), dim(d), keepdim(k) { name = "Sum"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        auto g = grad;
        if (dim >= 0 && !keepdim) {
            auto g_shape = g->shape();
            std::vector<int64_t> new_shape;
            for (int64_t i = 0; i < dim; ++i) {
                new_shape.push_back(g_shape[static_cast<size_t>(i)]);
            }
            new_shape.push_back(1);
            for (size_t i = static_cast<size_t>(dim); i < g_shape.size(); ++i) {
                new_shape.push_back(g_shape[i]);
            }
            g = g->reshape(new_shape);
        }
        g = ops::broadcast_to(g, a_shape);
        return {g};
    }
};

class MeanNode : public Node {
public:
    std::vector<int64_t> a_shape;
    double inv_n;
    MeanNode(std::vector<int64_t> s, double n)
        : a_shape(std::move(s)), inv_n(1.0 / n) { name = "Mean"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        auto scaled = ops::mul(grad, make_tensor({inv_n}, {}));
        auto g = ops::broadcast_to(scaled, a_shape);
        return {g};
    }
};

class TransposeNode : public Node {
public:
    int64_t dim0, dim1;
    TransposeNode(int64_t d0, int64_t d1) : dim0(d0), dim1(d1) { name = "Transpose"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        return {grad->transpose(dim0, dim1)};
    }
};

// ── 数学函数 Node 子类 ─────────────────────────────────

class ExpNode : public Node {
public:
    TensorImplPtr result;
    explicit ExpNode(TensorImplPtr r) : result(std::move(r)) { name = "Exp"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        return {mul(grad, result)};
    }
};

class LogNode : public Node {
public:
    TensorImplPtr orig_a;
    explicit LogNode(TensorImplPtr a) : orig_a(std::move(a)) { name = "Log"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        return {div(grad, orig_a)};
    }
};

class SqrtNode : public Node {
public:
    TensorImplPtr result;
    explicit SqrtNode(TensorImplPtr r) : result(std::move(r)) { name = "Sqrt"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        // d/dx sqrt(x) = 1 / (2 * sqrt(x)) = 1 / (2 * result)
        auto two = ops::mul(result, make_tensor({2.0}, {}));
        return {div(grad, two)};
    }
};

class AbsNode : public Node {
public:
    TensorImplPtr sign;
    explicit AbsNode(TensorImplPtr s) : sign(std::move(s)) { name = "Abs"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        return {mul(grad, sign)};
    }
};

class PowScalarNode : public Node {
public:
    TensorImplPtr orig_a;
    double exponent;
    PowScalarNode(TensorImplPtr a, double e) : orig_a(std::move(a)), exponent(e) { name = "PowScalar"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        // d/dx x^n = n * x^(n-1)
        auto grad_a = ops::mul(make_tensor({exponent}, {}),
                               ops::pow_scalar(orig_a, exponent - 1.0));
        return {mul(grad, grad_a)};
    }
};

class ClampNode : public Node {
public:
    TensorImplPtr mask;
    explicit ClampNode(TensorImplPtr m) : mask(std::move(m)) { name = "Clamp"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        return {mul(grad, mask)};
    }
};

class SigmoidNode : public Node {
public:
    TensorImplPtr result;
    explicit SigmoidNode(TensorImplPtr r) : result(std::move(r)) { name = "Sigmoid"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) = result * (1 - result)
        auto one_minus = ops::sub(make_tensor({1.0}, {}), result);
        return {mul(grad, mul(result, one_minus))};
    }
};

class TanhNode : public Node {
public:
    TensorImplPtr result;
    explicit TanhNode(TensorImplPtr r) : result(std::move(r)) { name = "Tanh"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        // d/dx tanh(x) = 1 - tanh(x)^2 = 1 - result^2
        auto sq = ops::mul(result, result);
        auto grad_a = ops::sub(make_tensor({1.0}, {}), sq);
        return {mul(grad, grad_a)};
    }
};

// ── 损失函数 Node 子类 ─────────────────────────────────

class LogSoftmaxNode : public Node {
public:
    TensorImplPtr softmax;
    int64_t dim;
    LogSoftmaxNode(TensorImplPtr s, int64_t d) : softmax(std::move(s)), dim(d) { name = "LogSoftmax"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        // grad_x = grad - softmax * sum(grad, dim, keepdim=true)
        auto sum_grad = ops::sum(grad, dim, true);
        auto grad_x = ops::sub(grad, ops::mul(softmax, sum_grad));
        return {grad_x};
    }
};

class SoftmaxNode : public Node {
public:
    TensorImplPtr result;
    int64_t dim;
    SoftmaxNode(TensorImplPtr r, int64_t d) : result(std::move(r)), dim(d) { name = "Softmax"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        // grad_x = result * (grad - sum(grad * result, dim, keepdim=true))
        auto grad_result = ops::mul(grad, result);
        auto sum_grad = ops::sum(grad_result, dim, true);
        auto grad_x = ops::mul(result, ops::sub(grad, sum_grad));
        return {grad_x};
    }
};

class NLLLossNode : public Node {
public:
    int64_t n;
    int64_t num_classes;
    std::vector<int64_t> target;
    NLLLossNode(int64_t n_, int64_t c_, std::vector<int64_t> t)
        : n(n_), num_classes(c_), target(std::move(t)) { name = "NLLLoss"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        double grad_val = grad->item();
        std::vector<double> grad_data(static_cast<size_t>(n * num_classes), 0.0);
        for (int64_t i = 0; i < n; ++i) {
            grad_data[static_cast<size_t>(i * num_classes + target[static_cast<size_t>(i)])] =
                -grad_val / n;
        }
        auto grad_lp = make_tensor(grad_data, {n, num_classes});
        return {grad_lp, nullptr};
    }
};

class MaxNode : public Node {
public:
    TensorImplPtr mask;
    std::vector<int64_t> a_shape;
    int64_t dim;
    bool keepdim;
    MaxNode(TensorImplPtr m, std::vector<int64_t> s, int64_t d, bool k)
        : mask(std::move(m)), a_shape(std::move(s)), dim(d), keepdim(k) { name = "Max"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        auto g = grad;
        if (dim >= 0 && !keepdim) {
            auto g_shape = g->shape();
            std::vector<int64_t> new_shape;
            for (int64_t i = 0; i < dim; ++i) new_shape.push_back(g_shape[i]);
            new_shape.push_back(1);
            for (size_t i = static_cast<size_t>(dim); i < g_shape.size(); ++i)
                new_shape.push_back(g_shape[i]);
            g = g->reshape(new_shape);
        }
        g = ops::broadcast_to(g, a_shape);
        return {ops::mul(g, mask)};
    }
};

class MinNode : public Node {
public:
    TensorImplPtr mask;
    std::vector<int64_t> a_shape;
    int64_t dim;
    bool keepdim;
    MinNode(TensorImplPtr m, std::vector<int64_t> s, int64_t d, bool k)
        : mask(std::move(m)), a_shape(std::move(s)), dim(d), keepdim(k) { name = "Min"; }
    std::vector<TensorImplPtr> apply(TensorImplPtr grad) override {
        auto g = grad;
        if (dim >= 0 && !keepdim) {
            auto g_shape = g->shape();
            std::vector<int64_t> new_shape;
            for (int64_t i = 0; i < dim; ++i) new_shape.push_back(g_shape[i]);
            new_shape.push_back(1);
            for (size_t i = static_cast<size_t>(dim); i < g_shape.size(); ++i)
                new_shape.push_back(g_shape[i]);
            g = g->reshape(new_shape);
        }
        g = ops::broadcast_to(g, a_shape);
        return {ops::mul(g, mask)};
    }
};

// ── 逐元素二元算子 ────────────────────────────────────

TensorImplPtr add(const TensorImplPtr& a, const TensorImplPtr& b) {
    auto [ba, bb] = ops::broadcast_tensors(a, b);
    auto result = ops::add(ba, bb);

    if (!is_grad_enabled() || !any_requires_grad({a, b})) return result;

    auto node = std::make_shared<AddNode>(a->shape(), b->shape());
    attach_node(result, node, collect_edges({a, b}));
    return result;
}

TensorImplPtr sub(const TensorImplPtr& a, const TensorImplPtr& b) {
    auto [ba, bb] = ops::broadcast_tensors(a, b);
    auto result = ops::sub(ba, bb);

    if (!is_grad_enabled() || !any_requires_grad({a, b})) return result;

    auto node = std::make_shared<SubNode>(a->shape(), b->shape());
    attach_node(result, node, collect_edges({a, b}));
    return result;
}

TensorImplPtr mul(const TensorImplPtr& a, const TensorImplPtr& b) {
    auto [ba, bb] = ops::broadcast_tensors(a, b);
    auto result = ops::mul(ba, bb);

    if (!is_grad_enabled() || !any_requires_grad({a, b})) return result;

    auto node = std::make_shared<MulNode>(a->shape(), b->shape(), a, b);
    attach_node(result, node, collect_edges({a, b}));
    return result;
}

TensorImplPtr div(const TensorImplPtr& a, const TensorImplPtr& b) {
    auto [ba, bb] = ops::broadcast_tensors(a, b);
    auto result = ops::div(ba, bb);

    if (!is_grad_enabled() || !any_requires_grad({a, b})) return result;

    auto node = std::make_shared<DivNode>(a->shape(), b->shape(), a, b);
    attach_node(result, node, collect_edges({a, b}));
    return result;
}

// ── 一元算子 ──────────────────────────────────────────

TensorImplPtr neg(const TensorImplPtr& a) {
    auto result = ops::neg(a);

    if (!is_grad_enabled() || !any_requires_grad({a})) return result;

    auto node = std::make_shared<NegNode>();
    attach_node(result, node, collect_edges({a}));
    return result;
}

TensorImplPtr relu(const TensorImplPtr& a) {
    auto result = ops::relu(a);

    if (!is_grad_enabled() || !any_requires_grad({a})) return result;

    auto mask = ops::greater_than_zero(a);
    auto node = std::make_shared<ReluNode>(mask);
    attach_node(result, node, collect_edges({a}));
    return result;
}

// ── 矩阵乘法 ──────────────────────────────────────────

TensorImplPtr matmul(const TensorImplPtr& a, const TensorImplPtr& b) {
    auto result = ops::matmul(a, b);

    if (!is_grad_enabled() || !any_requires_grad({a, b})) return result;

    auto node = std::make_shared<MatmulNode>(a, b, a->ndim() == 1, b->ndim() == 1);
    attach_node(result, node, collect_edges({a, b}));
    return result;
}

// ── 归约 ──────────────────────────────────────────────

TensorImplPtr sum(const TensorImplPtr& a, int64_t dim, bool keepdim) {
    auto result = ops::sum(a, dim, keepdim);

    if (!is_grad_enabled() || !any_requires_grad({a})) return result;

    auto node = std::make_shared<SumNode>(a->shape(), dim, keepdim);
    attach_node(result, node, collect_edges({a}));
    return result;
}

TensorImplPtr mean(const TensorImplPtr& a, int64_t dim, bool keepdim) {
    auto result = ops::mean(a, dim, keepdim);

    if (!is_grad_enabled() || !any_requires_grad({a})) return result;

    auto node = std::make_shared<MeanNode>(a->shape(), static_cast<double>(a->numel()));
    attach_node(result, node, collect_edges({a}));
    return result;
}

// ── 转置 ──────────────────────────────────────────────

TensorImplPtr transpose(const TensorImplPtr& a, int64_t dim0, int64_t dim1) {
    auto result = a->transpose(dim0, dim1);

    if (!is_grad_enabled() || !any_requires_grad({a})) return result;

    auto node = std::make_shared<TransposeNode>(dim0, dim1);
    attach_node(result, node, collect_edges({a}));
    return result;
}

// ── 逐元素数学函数 ─────────────────────────────────────

TensorImplPtr exp(const TensorImplPtr& a) {
    auto result = ops::exp(a);
    if (!is_grad_enabled() || !any_requires_grad({a})) return result;
    auto node = std::make_shared<ExpNode>(result);
    attach_node(result, node, collect_edges({a}));
    return result;
}

TensorImplPtr log(const TensorImplPtr& a) {
    auto result = ops::log(a);
    if (!is_grad_enabled() || !any_requires_grad({a})) return result;
    auto node = std::make_shared<LogNode>(a);
    attach_node(result, node, collect_edges({a}));
    return result;
}

TensorImplPtr sqrt(const TensorImplPtr& a) {
    auto result = ops::sqrt(a);
    if (!is_grad_enabled() || !any_requires_grad({a})) return result;
    auto node = std::make_shared<SqrtNode>(result);
    attach_node(result, node, collect_edges({a}));
    return result;
}

TensorImplPtr abs_val(const TensorImplPtr& a) {
    auto result = ops::abs_val(a);
    if (!is_grad_enabled() || !any_requires_grad({a})) return result;
    // sign(x) = x / |x|, 0 when x == 0
    auto sign = ops::div(a, ops::abs_val(a));
    auto node = std::make_shared<AbsNode>(sign);
    attach_node(result, node, collect_edges({a}));
    return result;
}

TensorImplPtr pow_scalar(const TensorImplPtr& a, double exponent) {
    auto result = ops::pow_scalar(a, exponent);
    if (!is_grad_enabled() || !any_requires_grad({a})) return result;
    auto node = std::make_shared<PowScalarNode>(a, exponent);
    attach_node(result, node, collect_edges({a}));
    return result;
}

TensorImplPtr clamp(const TensorImplPtr& a, double min_val, double max_val) {
    auto result = ops::clamp(a, min_val, max_val);
    if (!is_grad_enabled() || !any_requires_grad({a})) return result;
    // mask = 1 where min < x < max, else 0
    auto mask = ops::greater_than_zero(
        ops::mul(ops::sub(a, make_tensor({min_val}, {})),
                 ops::sub(make_tensor({max_val}, {}), a))
    );
    auto node = std::make_shared<ClampNode>(mask);
    attach_node(result, node, collect_edges({a}));
    return result;
}

TensorImplPtr sigmoid(const TensorImplPtr& a) {
    auto result = ops::sigmoid(a);
    if (!is_grad_enabled() || !any_requires_grad({a})) return result;
    auto node = std::make_shared<SigmoidNode>(result);
    attach_node(result, node, collect_edges({a}));
    return result;
}

TensorImplPtr tanh(const TensorImplPtr& a) {
    auto result = ops::tanh(a);
    if (!is_grad_enabled() || !any_requires_grad({a})) return result;
    auto node = std::make_shared<TanhNode>(result);
    attach_node(result, node, collect_edges({a}));
    return result;
}

// ── 损失函数 ─────────────────────────────────────────────

TensorImplPtr log_softmax(const TensorImplPtr& a, int64_t dim) {
    if (dim < 0) dim += a->ndim();
    auto result = ops::log_softmax(a, dim);
    if (!is_grad_enabled() || !any_requires_grad({a})) return result;
    auto sm = ops::exp(result);
    auto node = std::make_shared<LogSoftmaxNode>(sm, dim);
    attach_node(result, node, collect_edges({a}));
    return result;
}

TensorImplPtr softmax(const TensorImplPtr& a, int64_t dim) {
    if (dim < 0) dim += a->ndim();
    auto result = ops::softmax(a, dim);
    if (!is_grad_enabled() || !any_requires_grad({a})) return result;
    auto node = std::make_shared<SoftmaxNode>(result, dim);
    attach_node(result, node, collect_edges({a}));
    return result;
}

TensorImplPtr nll_loss(const TensorImplPtr& log_probs, const TensorImplPtr& target) {
    auto result = ops::nll_loss(log_probs, target);
    if (!is_grad_enabled() || !any_requires_grad({log_probs})) return result;

    int64_t n = log_probs->shape()[0];
    int64_t c = log_probs->shape()[1];
    auto tgt_data = target->to_vector();
    std::vector<int64_t> tgt_int(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        tgt_int[static_cast<size_t>(i)] = static_cast<int64_t>(tgt_data[static_cast<size_t>(i)]);

    auto node = std::make_shared<NLLLossNode>(n, c, std::move(tgt_int));
    attach_node(result, node, collect_edges({log_probs, target}));
    return result;
}

TensorImplPtr cross_entropy(const TensorImplPtr& logits, const TensorImplPtr& target, int64_t dim) {
    auto lp = log_softmax(logits, dim);
    return nll_loss(lp, target);
}

TensorImplPtr mse_loss(const TensorImplPtr& pred, const TensorImplPtr& target) {
    auto diff = sub(pred, target);
    auto sq = mul(diff, diff);
    return mean(sq);
}

// ── 归约算子（可微）────────────────────────────────────

static TensorImplPtr _make_extreme_mask(const TensorImplPtr& a, int64_t dim, bool is_max) {
    int64_t ndim = a->ndim();
    const auto& shape = a->shape();
    int64_t dim_size = shape[static_cast<size_t>(dim)];
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];

    auto a_data = a->to_vector();
    std::vector<double> mask(a_data.size(), 0.0);

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            int64_t best = 0;
            double best_val = is_max ? -1e30 : 1e30;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (o * dim_size + d) * inner + in;
                double v = a_data[static_cast<size_t>(idx)];
                if (is_max ? (v > best_val) : (v < best_val)) {
                    best_val = v;
                    best = d;
                }
            }
            mask[static_cast<size_t>((o * dim_size + best) * inner + in)] = 1.0;
        }
    }
    return make_tensor(mask, shape);
}

TensorImplPtr max(const TensorImplPtr& a, int64_t dim, bool keepdim) {
    if (dim < 0) dim += a->ndim();
    auto result = ops::max(a, dim, keepdim);
    if (!is_grad_enabled() || !any_requires_grad({a})) return result;
    auto mask = _make_extreme_mask(a, dim, true);
    auto node = std::make_shared<MaxNode>(mask, a->shape(), dim, keepdim);
    attach_node(result, node, collect_edges({a}));
    return result;
}

TensorImplPtr min(const TensorImplPtr& a, int64_t dim, bool keepdim) {
    if (dim < 0) dim += a->ndim();
    auto result = ops::min(a, dim, keepdim);
    if (!is_grad_enabled() || !any_requires_grad({a})) return result;
    auto mask = _make_extreme_mask(a, dim, false);
    auto node = std::make_shared<MinNode>(mask, a->shape(), dim, keepdim);
    attach_node(result, node, collect_edges({a}));
    return result;
}

} // namespace minitorch::autograd
