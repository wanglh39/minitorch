// TensorImpl 实现（Ch8）

#include "c10/tensor.h"
#include "autograd/engine.h"
#include "autograd/grad_mode.h"
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace minitorch {

// ── 工具函数 ─────────────────────────────────────────

std::vector<int64_t> compute_contiguous_strides(const std::vector<int64_t>& shape) {
    int64_t n = static_cast<int64_t>(shape.size());
    std::vector<int64_t> strides(n);
    if (n == 0) return strides;
    strides[n - 1] = 1;
    for (int64_t i = n - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

std::vector<int64_t> infer_shape(std::vector<int64_t> shape, int64_t numel) {
    int64_t known = 1;
    int neg_count = 0;
    for (auto& d : shape) {
        if (d == -1) ++neg_count;
        else known *= d;
    }
    if (neg_count > 1) throw std::runtime_error("只能有一个 -1");
    if (neg_count == 1) {
        for (auto& d : shape) {
            if (d == -1) d = numel / known;
        }
    }
    return shape;
}

std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a,
                                       const std::vector<int64_t>& b) {
    int64_t na = static_cast<int64_t>(a.size());
    int64_t nb = static_cast<int64_t>(b.size());
    int64_t n = std::max(na, nb);
    std::vector<int64_t> result(n);
    for (int64_t i = 0; i < n; ++i) {
        int64_t da = (i < na) ? a[na - 1 - i] : 1;
        int64_t db = (i < nb) ? b[nb - 1 - i] : 1;
        if (da == 1) result[n - 1 - i] = db;
        else if (db == 1) result[n - 1 - i] = da;
        else if (da == db) result[n - 1 - i] = da;
        else throw std::runtime_error("shape 无法广播");
    }
    return result;
}

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// ── TensorImpl 构造 ──────────────────────────────────

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage,
                       std::vector<int64_t> shape,
                       std::vector<int64_t> strides,
                       int64_t storage_offset,
                       bool requires_grad)
    : storage_(storage), shape_(std::move(shape)), strides_(std::move(strides)),
      storage_offset_(storage_offset), requires_grad_(requires_grad) {}

TensorImpl::TensorImpl(const std::vector<double>& data,
                       std::vector<int64_t> shape,
                       bool requires_grad)
    : shape_(std::move(shape)),
      strides_(compute_contiguous_strides(shape_)),
      storage_offset_(0),
      requires_grad_(requires_grad) {
    storage_ = make_storage(data);
}

TensorImpl::TensorImpl(const TensorImpl& other)
    : storage_(other.storage_), shape_(other.shape_), strides_(other.strides_),
      storage_offset_(other.storage_offset_), requires_grad_(other.requires_grad_) {}

TensorImpl& TensorImpl::operator=(const TensorImpl& other) {
    if (this != &other) {
        storage_ = other.storage_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        storage_offset_ = other.storage_offset_;
        requires_grad_ = other.requires_grad_;
    }
    return *this;
}

// ── 形状操作 ─────────────────────────────────────────

bool TensorImpl::is_contiguous() const {
    if (numel() <= 1) return true;
    return strides_ == compute_contiguous_strides(shape_);
}

TensorImplPtr TensorImpl::contiguous() const {
    if (is_contiguous()) {
        return std::make_shared<TensorImpl>(*this);
    }
    // materialize：按 stride 读取数据到连续 buffer
    auto n = numel();
    std::vector<double> flat(static_cast<size_t>(n));
    // 遍历所有逻辑索引
    std::vector<int64_t> indices(ndim(), 0);
    for (int64_t i = 0; i < n; ++i) {
        flat[static_cast<size_t>(i)] = storage_->data()[static_cast<size_t>(linear_offset(indices))];
        // 递增索引（行优先）
        for (int64_t d = ndim() - 1; d >= 0; --d) {
            if (++indices[d] < shape_[d]) break;
            indices[d] = 0;
        }
    }
    auto new_storage = make_storage(flat);
    return std::make_shared<TensorImpl>(new_storage, shape_,
                                        compute_contiguous_strides(shape_), 0, requires_grad_);
}

TensorImplPtr TensorImpl::view(std::vector<int64_t> shape) const {
    shape = infer_shape(std::move(shape), numel());
    if (!is_contiguous()) {
        throw std::runtime_error("view 要求 contiguous，请用 reshape() 或先 contiguous()");
    }
    auto new_strides = compute_contiguous_strides(shape);
    return std::make_shared<TensorImpl>(storage_, shape, new_strides,
                                        storage_offset_, requires_grad_);
}

TensorImplPtr TensorImpl::reshape(std::vector<int64_t> shape) const {
    shape = infer_shape(std::move(shape), numel());
    if (is_contiguous()) {
        return view(shape);
    }
    return contiguous()->view(shape);
}

TensorImplPtr TensorImpl::transpose(int64_t dim0, int64_t dim1) const {
    if (dim0 < 0) dim0 += ndim();
    if (dim1 < 0) dim1 += ndim();
    auto shape = shape_;
    auto strides = strides_;
    std::swap(shape[dim0], shape[dim1]);
    std::swap(strides[dim0], strides[dim1]);
    return std::make_shared<TensorImpl>(storage_, shape, strides,
                                        storage_offset_, requires_grad_);
}

TensorImplPtr TensorImpl::permute(std::vector<int64_t> dims) const {
    int64_t n = ndim();
    for (auto& d : dims) {
        if (d < 0) d += n;
    }
    std::vector<int64_t> new_shape, new_strides;
    for (auto d : dims) {
        new_shape.push_back(shape_[d]);
        new_strides.push_back(strides_[d]);
    }
    return std::make_shared<TensorImpl>(storage_, new_shape, new_strides,
                                        storage_offset_, requires_grad_);
}

// ── 数据访问 ─────────────────────────────────────────

int64_t TensorImpl::linear_offset(const std::vector<int64_t>& indices) const {
    int64_t offset = storage_offset_;
    for (size_t i = 0; i < indices.size() && i < strides_.size(); ++i) {
        offset += indices[i] * strides_[i];
    }
    return offset;
}

double TensorImpl::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item() 要求 numel()==1");
    }
    return storage_->data()[static_cast<size_t>(storage_offset_)];
}

std::vector<double> TensorImpl::to_vector() const {
    auto n = numel();
    std::vector<double> result(static_cast<size_t>(n));
    std::vector<int64_t> indices(ndim(), 0);
    for (int64_t i = 0; i < n; ++i) {
        result[static_cast<size_t>(i)] = storage_->data()[static_cast<size_t>(linear_offset(indices))];
        for (int64_t d = ndim() - 1; d >= 0; --d) {
            if (++indices[d] < shape_[d]) break;
            indices[d] = 0;
        }
    }
    return result;
}

void TensorImpl::from_vector(const std::vector<double>& data) {
    if (static_cast<int64_t>(data.size()) != numel()) {
        throw std::runtime_error("from_vector: 大小不匹配");
    }
    if (!is_contiguous()) {
        // 非连续：按索引写入
        std::vector<int64_t> indices(ndim(), 0);
        for (int64_t i = 0; i < numel(); ++i) {
            storage_->data()[static_cast<size_t>(linear_offset(indices))] = data[static_cast<size_t>(i)];
            for (int64_t d = ndim() - 1; d >= 0; --d) {
                if (++indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    } else {
        // 连续：直接拷贝
        std::copy(data.begin(), data.end(),
                  storage_->data() + storage_offset_);
    }
}

void TensorImpl::set_item(const std::vector<int64_t>& indices, double value) {
    storage_->data()[static_cast<size_t>(linear_offset(indices))] = value;
}

TensorImplPtr TensorImpl::clone() const {
    auto data = to_vector();
    auto result = make_tensor(data, shape_);
    result->set_requires_grad(requires_grad_);
    return result;
}

TensorImplPtr TensorImpl::expand(std::vector<int64_t> target_shape) const {
    int64_t src_ndim = static_cast<int64_t>(shape_.size());
    int64_t dst_ndim = static_cast<int64_t>(target_shape.size());
    if (dst_ndim < src_ndim) {
        throw std::runtime_error("expand: target ndim < source ndim");
    }
    int64_t pad = dst_ndim - src_ndim;
    std::vector<int64_t> padded_shape(dst_ndim, 1);
    std::vector<int64_t> padded_strides(dst_ndim, 0);
    for (int64_t i = 0; i < src_ndim; ++i) {
        padded_shape[pad + i] = shape_[i];
        padded_strides[pad + i] = strides_[i];
    }
    std::vector<int64_t> new_strides(dst_ndim);
    for (int64_t i = 0; i < dst_ndim; ++i) {
        if (padded_shape[i] == target_shape[i]) {
            new_strides[i] = padded_strides[i];
        } else if (padded_shape[i] == 1) {
            new_strides[i] = 0;
        } else {
            throw std::runtime_error("expand: shape 不兼容");
        }
    }
    return std::make_shared<TensorImpl>(storage_, target_shape, new_strides,
                                        storage_offset_, requires_grad_);
}

void TensorImpl::fill_(double value) {
    int64_t n = numel();
    std::vector<int64_t> indices(ndim(), 0);
    for (int64_t i = 0; i < n; ++i) {
        storage_->data()[static_cast<size_t>(linear_offset(indices))] = value;
        for (int64_t d = ndim() - 1; d >= 0; --d) {
            if (++indices[d] < shape_[d]) break;
            indices[d] = 0;
        }
    }
}

void TensorImpl::zero_() {
    fill_(0.0);
}

std::string TensorImpl::repr() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], data=[";
    auto data = to_vector();
    for (size_t i = 0; i < data.size() && i < 10; ++i) {
        if (i > 0) oss << ", ";
        oss << data[i];
    }
    if (data.size() > 10) oss << ", ...";
    oss << "])";
    return oss.str();
}

// ── backward 入口 ───────────────────────────────────

void TensorImpl::backward(TensorImplPtr gradient,
                          bool retain_graph,
                          bool retain_grad,
                          bool create_graph) {
    if (!grad_fn_) {
        throw std::runtime_error(
            "backward() called on a tensor with no grad_fn");
    }
    if (!gradient) {
        if (numel() != 1) {
            throw std::runtime_error(
                "grad can be implicitly created only for scalar outputs");
        }
        std::vector<double> ones(static_cast<size_t>(numel()), 1.0);
        gradient = make_tensor(ones, shape_);
    }
    run_backward(grad_fn_, gradient, retain_graph, retain_grad, create_graph);
    if (!retain_graph && !create_graph) {
        grad_fn_.reset();
    }
}

} // namespace minitorch