// TensorImpl：张量实现核心（Ch8 C++ 核心重写）
//
// 对应阶段一的 Python Tensor（src/minitorch/tensor.py）。
// 持有 shape/strides/storage_offset + shared_ptr<Storage>。
// 对应真实 PyTorch 的 at::TensorImpl / c10::TensorImpl。
//
// 设计要点：
//   - shared_ptr<Storage> 实现引用计数，多个 TensorImpl 可共享同一 Storage
//   - view 操作（transpose/slice）只改 shape/strides，不拷贝数据
//   - 算子通过 linear_offset() 计算物理偏移，访问 Storage

#pragma once

#include "c10/storage.h"
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <functional>

namespace minitorch {

// 前向声明
class TensorImpl;
using TensorImplPtr = std::shared_ptr<TensorImpl>;

class Node;
using NodePtr = std::shared_ptr<Node>;

// 计算 contiguous strides
std::vector<int64_t> compute_contiguous_strides(const std::vector<int64_t>& shape);

// 推导 shape（支持 -1）
std::vector<int64_t> infer_shape(std::vector<int64_t> shape, int64_t numel);

// 广播 shapes
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a,
                                       const std::vector<int64_t>& b);

// shape 乘积
int64_t numel(const std::vector<int64_t>& shape);

class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    // 从 Storage 创建
    TensorImpl(std::shared_ptr<Storage> storage,
               std::vector<int64_t> shape,
               std::vector<int64_t> strides,
               int64_t storage_offset = 0,
               bool requires_grad = false);

    // 从数据数组创建（便捷构造）
    TensorImpl(const std::vector<double>& data,
               std::vector<int64_t> shape,
               bool requires_grad = false);

    // 拷贝构造（浅拷贝，共享 Storage）
    TensorImpl(const TensorImpl& other);
    TensorImpl& operator=(const TensorImpl& other);

    ~TensorImpl() = default;

    // ── 属性 ──────────────────────────────────
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t storage_offset() const { return storage_offset_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return ::minitorch::numel(shape_); }
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool v) { requires_grad_ = v; }

    std::shared_ptr<Storage> storage() { return storage_; }
    std::shared_ptr<const Storage> storage() const { return storage_; }

    // ── 形状操作 ──────────────────────────────
    bool is_contiguous() const;
    TensorImplPtr contiguous() const;
    TensorImplPtr view(std::vector<int64_t> shape) const;
    TensorImplPtr reshape(std::vector<int64_t> shape) const;
    TensorImplPtr transpose(int64_t dim0 = 1, int64_t dim1 = 0) const;
    TensorImplPtr permute(std::vector<int64_t> dims) const;
    TensorImplPtr clone() const;
    TensorImplPtr expand(std::vector<int64_t> shape) const;

    // ── 数据访问 ──────────────────────────────
    int64_t linear_offset(const std::vector<int64_t>& indices) const;
    double item() const;
    std::vector<double> to_vector() const;
    void from_vector(const std::vector<double>& data);

    // ── 原地操作 ──────────────────────────────
    void set_item(const std::vector<int64_t>& indices, double value);
    void fill_(double value);
    void zero_();

    // ── Autograd ──────────────────────────────
    TensorImplPtr grad() const { return grad_; }
    void set_grad(TensorImplPtr g) { grad_ = std::move(g); }
    NodePtr grad_fn() const { return grad_fn_; }
    void set_grad_fn(NodePtr n) { grad_fn_ = std::move(n); }
    bool is_leaf() const { return is_leaf_; }
    void set_is_leaf(bool v) { is_leaf_ = v; }

    // ── Hooks ──────────────────────────────────
    using HookFn = std::function<TensorImplPtr(TensorImplPtr)>;
    void register_hook(HookFn fn) { backward_hook_ = std::move(fn); }
    void clear_hook() { backward_hook_ = nullptr; }
    HookFn backward_hook() const { return backward_hook_; }

    // 反向传播入口
    void backward(TensorImplPtr gradient = nullptr,
                  bool retain_graph = false,
                  bool retain_grad = false,
                  bool create_graph = false);

    // ── 调试 ──────────────────────────────────
    std::string repr() const;

private:
    std::shared_ptr<Storage> storage_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t storage_offset_;
    bool requires_grad_;

    // Autograd 状态
    TensorImplPtr grad_;       // 累积梯度
    NodePtr grad_fn_;          // 反向函数节点
    bool is_leaf_ = true;      // 是否叶子张量
    HookFn backward_hook_;     // 梯度钩子

    // 内部构造，不检查
    TensorImpl(std::shared_ptr<Storage> storage,
               std::vector<int64_t> shape,
               std::vector<int64_t> strides,
               int64_t storage_offset,
               bool requires_grad,
               bool /*internal_tag*/);
};

// 便捷工厂
inline TensorImplPtr make_tensor(const std::vector<double>& data,
                                  std::vector<int64_t> shape,
                                  bool requires_grad = false) {
    return std::make_shared<TensorImpl>(data, shape, requires_grad);
}

} // namespace minitorch