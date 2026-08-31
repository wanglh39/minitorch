// pybind11 绑定层（Ch8）
//
// 把 C++ TensorImpl/Storage/ops 暴露给 Python。
// Python 端通过 minitorch._C_ext 访问 C++ 核心。
// 对应真实 PyTorch 的 csrc/autograd/python_autograd.cpp + torch/csrc/Module.cpp。
//
// 设计要点：
//   - TensorImpl 用 shared_ptr 包装，pybind11 自动处理引用计数
//   - 算子返回新 TensorImplPtr（不原地修改）
//   - Python 端的 Tensor 类包装此 C++ TensorImpl

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "c10/storage.h"
#include "c10/tensor.h"
#include "c10/allocator.h"
#include "aten/ops.h"
#include "autograd/grad_mode.h"
#include "autograd/node.h"
#include "autograd/engine.h"
#include "autograd/ops.h"
#include "autograd/profiler.h"
#include "autograd/checkpoint.h"

namespace py = pybind11;
using namespace minitorch;
using namespace minitorch::ops;

PYBIND11_MODULE(_C_ext, m) {
    m.doc() = "minitorch C++ core (Ch8)";

    // ── Storage ──────────────────────────────────
    py::class_<Storage, std::shared_ptr<Storage>>(m, "Storage")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<const std::vector<double>&>(), py::arg("data"))
        .def("__len__", &Storage::size)
        .def("__getitem__", [](const Storage& s, size_t i) { return s[i]; })
        .def("__repr__", &Storage::repr)
        .def("data", [](Storage& s) {
            return py::array_t<double>(
                static_cast<py::ssize_t>(s.size()),
                s.data(),
                py::cast(&s)
            );
        });

    // ── TensorImpl ───────────────────────────────
    py::class_<TensorImpl, TensorImplPtr>(m, "TensorImpl")
        .def(py::init<const std::vector<double>&, std::vector<int64_t>, bool>(),
             py::arg("data"), py::arg("shape"), py::arg("requires_grad") = false)
        // 属性
        .def_property_readonly("shape", [](const TensorImpl& t) {
            return t.shape();
        })
        .def_property_readonly("strides", [](const TensorImpl& t) {
            return t.strides();
        })
        .def_property_readonly("ndim", &TensorImpl::ndim)
        .def_property_readonly("numel", &TensorImpl::numel)
        .def_property_readonly("storage_offset", &TensorImpl::storage_offset)
        .def_property("requires_grad",
            &TensorImpl::requires_grad,
            &TensorImpl::set_requires_grad)
        .def("is_contiguous", &TensorImpl::is_contiguous)
        // Autograd 属性
        .def_property_readonly("grad", [](const TensorImplPtr& t) {
            return t->grad();
        })
        .def_property_readonly("grad_fn", [](const TensorImplPtr& t) {
            return t->grad_fn();
        })
        .def_property_readonly("is_leaf", &TensorImpl::is_leaf)
        .def("backward", [](const TensorImplPtr& t, py::object gradient,
                            bool retain_graph, bool retain_grad, bool create_graph) {
            TensorImplPtr grad_ptr;
            if (!gradient.is_none()) {
                grad_ptr = gradient.cast<TensorImplPtr>();
            }
            t->backward(grad_ptr, retain_graph, retain_grad, create_graph);
        }, py::arg("gradient") = py::none(),
           py::arg("retain_graph") = false,
           py::arg("retain_grad") = false,
           py::arg("create_graph") = false)
        .def("backward_mt", [](const TensorImplPtr& t, py::object gradient,
                               bool retain_graph, bool retain_grad, int num_threads,
                               bool create_graph) {
            TensorImplPtr grad_ptr;
            if (!gradient.is_none()) {
                grad_ptr = gradient.cast<TensorImplPtr>();
            }
            if (!t->grad_fn()) {
                throw std::runtime_error("backward_mt: tensor has no grad_fn");
            }
            if (!grad_ptr) {
                grad_ptr = make_tensor({1.0}, {});
            }
            run_backward_mt(t->grad_fn(), grad_ptr, retain_graph, retain_grad, num_threads, create_graph);
        }, py::arg("gradient") = py::none(),
           py::arg("retain_graph") = false,
           py::arg("retain_grad") = false,
           py::arg("num_threads") = 0,
           py::arg("create_graph") = false)
        .def("zero_grad", [](TensorImplPtr& t) {
            t->set_grad(nullptr);
        })
        .def("set_grad", [](TensorImplPtr& t, py::object g) {
            if (g.is_none()) {
                t->set_grad(nullptr);
            } else {
                t->set_grad(g.cast<TensorImplPtr>());
            }
        })
        // Hooks
        .def("register_hook", [](TensorImplPtr& t, py::object fn) {
            t->register_hook([fn](TensorImplPtr grad) -> TensorImplPtr {
                py::gil_scoped_acquire acquire;
                py::object result = fn(grad);
                if (result.is_none()) return nullptr;
                return result.cast<TensorImplPtr>();
            });
        })
        .def("clear_hook", [](TensorImplPtr& t) { t->clear_hook(); })
        // 形状操作
        .def("contiguous", &TensorImpl::contiguous)
        .def("view", [](const TensorImplPtr& t, std::vector<int64_t> shape) {
            return t->view(std::move(shape));
        })
        .def("reshape", [](const TensorImplPtr& t, std::vector<int64_t> shape) {
            return t->reshape(std::move(shape));
        })
        .def("transpose", &TensorImpl::transpose,
             py::arg("dim0") = 1, py::arg("dim1") = 0)
        .def("permute", [](const TensorImplPtr& t, std::vector<int64_t> dims) {
            return t->permute(std::move(dims));
        })
        .def("clone", &TensorImpl::clone)
        .def("expand", [](const TensorImplPtr& t, std::vector<int64_t> shape) {
            return t->expand(std::move(shape));
        })
        .def("fill_", [](TensorImplPtr& t, double v) { t->fill_(v); })
        .def("zero_", [](TensorImplPtr& t) { t->zero_(); })
        // 数据访问
        .def("item", &TensorImpl::item)
        .def("to_vector", &TensorImpl::to_vector)
        .def("from_vector", &TensorImpl::from_vector)
        .def("numpy", [](const TensorImplPtr& t) {
            auto data = t->to_vector();
            auto shape = t->shape();
            std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
            py::array_t<double> arr(py_shape);
            std::copy(data.begin(), data.end(), arr.mutable_data());
            return arr;
        })
        .def("__repr__", &TensorImpl::repr)
        // 工厂方法
        .def_static("zeros", [](std::vector<int64_t> shape) {
            int64_t n = 1;
            for (auto d : shape) n *= d;
            return make_tensor(std::vector<double>(static_cast<size_t>(n), 0.0), shape);
        })
        .def_static("ones", [](std::vector<int64_t> shape) {
            int64_t n = 1;
            for (auto d : shape) n *= d;
            return make_tensor(std::vector<double>(static_cast<size_t>(n), 1.0), shape);
        })
        .def_static("from_numpy", [](py::array_t<double> arr) {
            py::buffer_info buf = arr.request();
            std::vector<int64_t> shape;
            for (auto s : buf.shape) shape.push_back(static_cast<int64_t>(s));
            std::vector<double> data(
                static_cast<double*>(buf.ptr),
                static_cast<double*>(buf.ptr) + buf.size
            );
            return make_tensor(data, shape);
        });

    // ── 算子 ─────────────────────────────────────
    // 用 lambda 包装避免与 std::div/std::sum 等名字冲突
    m.def("add", [](const TensorImplPtr& a, const TensorImplPtr& b) { return add(a, b); });
    m.def("sub", [](const TensorImplPtr& a, const TensorImplPtr& b) { return sub(a, b); });
    m.def("mul", [](const TensorImplPtr& a, const TensorImplPtr& b) { return mul(a, b); });
    m.def("div", [](const TensorImplPtr& a, const TensorImplPtr& b) { return div(a, b); });
    m.def("neg", [](const TensorImplPtr& a) { return neg(a); });
    m.def("relu", [](const TensorImplPtr& a) { return relu(a); });
    m.def("exp", [](const TensorImplPtr& a) { return exp(a); });
    m.def("log", [](const TensorImplPtr& a) { return log(a); });
    m.def("sqrt", [](const TensorImplPtr& a) { return sqrt(a); });
    m.def("abs_val", [](const TensorImplPtr& a) { return abs_val(a); });
    m.def("pow_scalar", [](const TensorImplPtr& a, double e) { return pow_scalar(a, e); },
          py::arg("a"), py::arg("exponent"));
    m.def("clamp", [](const TensorImplPtr& a, double lo, double hi) { return clamp(a, lo, hi); },
          py::arg("a"), py::arg("min_val"), py::arg("max_val"));
    m.def("sigmoid", [](const TensorImplPtr& a) { return sigmoid(a); });
    m.def("tanh", [](const TensorImplPtr& a) { return tanh(a); });
    m.def("log_softmax", [](const TensorImplPtr& a, int64_t dim) { return log_softmax(a, dim); },
          py::arg("a"), py::arg("dim") = -1);
    m.def("softmax", [](const TensorImplPtr& a, int64_t dim) { return softmax(a, dim); },
          py::arg("a"), py::arg("dim") = -1);
    m.def("nll_loss", [](const TensorImplPtr& lp, const TensorImplPtr& t) { return nll_loss(lp, t); });
    m.def("sub_inplace", [](TensorImplPtr& t, const TensorImplPtr& s) { sub_inplace(t, s); });
    m.def("mul_inplace", [](TensorImplPtr& t, const TensorImplPtr& s) { mul_inplace(t, s); });
    m.def("div_inplace", [](TensorImplPtr& t, const TensorImplPtr& s) { div_inplace(t, s); });
    m.def("gt", [](const TensorImplPtr& a, const TensorImplPtr& b) { return gt(a, b); });
    m.def("lt", [](const TensorImplPtr& a, const TensorImplPtr& b) { return lt(a, b); });
    m.def("eq", [](const TensorImplPtr& a, const TensorImplPtr& b) { return eq(a, b); });
    m.def("ge", [](const TensorImplPtr& a, const TensorImplPtr& b) { return ge(a, b); });
    m.def("le", [](const TensorImplPtr& a, const TensorImplPtr& b) { return le(a, b); });
    m.def("max", [](const TensorImplPtr& a, int64_t dim, bool keepdim) { return max(a, dim, keepdim); },
          py::arg("a"), py::arg("dim") = -1, py::arg("keepdim") = false);
    m.def("min", [](const TensorImplPtr& a, int64_t dim, bool keepdim) { return min(a, dim, keepdim); },
          py::arg("a"), py::arg("dim") = -1, py::arg("keepdim") = false);
    m.def("argmax", [](const TensorImplPtr& a, int64_t dim, bool keepdim) { return argmax(a, dim, keepdim); },
          py::arg("a"), py::arg("dim") = -1, py::arg("keepdim") = false);
    m.def("sum", [](const TensorImplPtr& a, int64_t dim, bool keepdim) { return sum(a, dim, keepdim); },
          py::arg("a"), py::arg("dim") = -1, py::arg("keepdim") = false);
    m.def("mean", [](const TensorImplPtr& a, int64_t dim, bool keepdim) { return mean(a, dim, keepdim); },
          py::arg("a"), py::arg("dim") = -1, py::arg("keepdim") = false);
    m.def("matmul", [](const TensorImplPtr& a, const TensorImplPtr& b) { return matmul(a, b); });
    m.def("broadcast_tensors", [](const TensorImplPtr& a, const TensorImplPtr& b) { return broadcast_tensors(a, b); });

    // ── Autograd 算子（建图）─────────────────────
    m.def("autograd_add", [](const TensorImplPtr& a, const TensorImplPtr& b) { return autograd::add(a, b); });
    m.def("autograd_sub", [](const TensorImplPtr& a, const TensorImplPtr& b) { return autograd::sub(a, b); });
    m.def("autograd_mul", [](const TensorImplPtr& a, const TensorImplPtr& b) { return autograd::mul(a, b); });
    m.def("autograd_div", [](const TensorImplPtr& a, const TensorImplPtr& b) { return autograd::div(a, b); });
    m.def("autograd_neg", [](const TensorImplPtr& a) { return autograd::neg(a); });
    m.def("autograd_relu", [](const TensorImplPtr& a) { return autograd::relu(a); });
    m.def("autograd_matmul", [](const TensorImplPtr& a, const TensorImplPtr& b) { return autograd::matmul(a, b); });
    m.def("autograd_sum", [](const TensorImplPtr& a, int64_t dim, bool keepdim) { return autograd::sum(a, dim, keepdim); },
          py::arg("a"), py::arg("dim") = -1, py::arg("keepdim") = false);
    m.def("autograd_mean", [](const TensorImplPtr& a, int64_t dim, bool keepdim) { return autograd::mean(a, dim, keepdim); },
          py::arg("a"), py::arg("dim") = -1, py::arg("keepdim") = false);
    m.def("autograd_transpose", [](const TensorImplPtr& a, int64_t dim0, int64_t dim1) { return autograd::transpose(a, dim0, dim1); },
          py::arg("a"), py::arg("dim0") = 1, py::arg("dim1") = 0);
    m.def("autograd_exp", [](const TensorImplPtr& a) { return autograd::exp(a); });
    m.def("autograd_log", [](const TensorImplPtr& a) { return autograd::log(a); });
    m.def("autograd_sqrt", [](const TensorImplPtr& a) { return autograd::sqrt(a); });
    m.def("autograd_abs_val", [](const TensorImplPtr& a) { return autograd::abs_val(a); });
    m.def("autograd_pow_scalar", [](const TensorImplPtr& a, double e) { return autograd::pow_scalar(a, e); },
          py::arg("a"), py::arg("exponent"));
    m.def("autograd_clamp", [](const TensorImplPtr& a, double lo, double hi) { return autograd::clamp(a, lo, hi); },
          py::arg("a"), py::arg("min_val"), py::arg("max_val"));
    m.def("autograd_sigmoid", [](const TensorImplPtr& a) { return autograd::sigmoid(a); });
    m.def("autograd_tanh", [](const TensorImplPtr& a) { return autograd::tanh(a); });
    m.def("autograd_log_softmax", [](const TensorImplPtr& a, int64_t dim) { return autograd::log_softmax(a, dim); },
          py::arg("a"), py::arg("dim") = -1);
    m.def("autograd_softmax", [](const TensorImplPtr& a, int64_t dim) { return autograd::softmax(a, dim); },
          py::arg("a"), py::arg("dim") = -1);
    m.def("autograd_nll_loss", [](const TensorImplPtr& lp, const TensorImplPtr& t) { return autograd::nll_loss(lp, t); });
    m.def("autograd_cross_entropy", [](const TensorImplPtr& logits, const TensorImplPtr& t, int64_t dim) { return autograd::cross_entropy(logits, t, dim); },
          py::arg("logits"), py::arg("target"), py::arg("dim") = -1);
    m.def("autograd_mse_loss", [](const TensorImplPtr& pred, const TensorImplPtr& t) { return autograd::mse_loss(pred, t); });
    m.def("autograd_max", [](const TensorImplPtr& a, int64_t dim, bool keepdim) { return autograd::max(a, dim, keepdim); },
          py::arg("a"), py::arg("dim") = -1, py::arg("keepdim") = false);
    m.def("autograd_min", [](const TensorImplPtr& a, int64_t dim, bool keepdim) { return autograd::min(a, dim, keepdim); },
          py::arg("a"), py::arg("dim") = -1, py::arg("keepdim") = false);

    // ── Node ─────────────────────────────────────
    py::class_<Node, NodePtr>(m, "Node")
        .def_property_readonly("name", [](const Node& n) { return n.name; })
        .def_property_readonly("next_edges", [](const Node& n) { return n.next_edges; })
        .def("__repr__", [](const Node& n) { return "Node(" + n.name + ")"; });

    // ── GradMode ─────────────────────────────────
    m.def("is_grad_enabled", &is_grad_enabled);
    m.def("set_grad_enabled", [](bool v) {
        bool prev = grad_mode_enabled;
        grad_mode_enabled = v;
        return prev;
    });

    // ── Allocator ────────────────────────────────
    py::class_<Allocator, std::shared_ptr<Allocator>>(m, "Allocator")
        .def_property_readonly("total_allocated", &Allocator::total_allocated)
        .def_property_readonly("peak_allocated", &Allocator::peak_allocated)
        .def_property_readonly("num_allocations", &Allocator::num_allocations)
        .def_property_readonly("name", &Allocator::name);

    py::class_<DefaultAllocator, Allocator, std::shared_ptr<DefaultAllocator>>(m, "DefaultAllocator")
        .def(py::init<>());

    py::class_<PoolAllocator, Allocator, std::shared_ptr<PoolAllocator>>(m, "PoolAllocator")
        .def(py::init<size_t>(), py::arg("pool_threshold") = 1024 * 1024)
        .def_property_readonly("pool_hits", &PoolAllocator::pool_hits)
        .def_property_readonly("pool_misses", &PoolAllocator::pool_misses)
        .def_property_readonly("pooled_bytes", &PoolAllocator::pooled_bytes)
        .def_property_readonly("pool_size", &PoolAllocator::pool_size);

    m.def("get_global_allocator", []() { return std::shared_ptr<Allocator>(&get_global_allocator(), [](Allocator*){}); });
    m.def("set_global_allocator", [](std::shared_ptr<Allocator> alloc) { set_global_allocator(std::move(alloc)); });

    // ── Profiler ────────────────────────────────
    m.def("profiler_start", []() { get_global_profiler().start(); });
    m.def("profiler_stop", []() { get_global_profiler().stop(); });
    m.def("profiler_enabled", []() { return get_global_profiler().enabled(); });
    m.def("profiler_events", []() {
        auto& p = get_global_profiler();
        py::list result;
        for (const auto& e : p.events()) {
            result.append(py::make_tuple(
                e.node_name, e.duration_us, e.memory_before, e.memory_after, e.thread_id
            ));
        }
        return result;
    });

    // ── Anomaly Detection ───────────────────────
    m.def("set_anomaly_check_enabled", [](bool v) {
        set_anomaly_check_enabled(v);
    }, py::arg("enabled") = true);
    m.def("is_anomaly_check_enabled", &is_anomaly_check_enabled);

    // ── Gradient Checkpointing ──────────────────
    m.def("checkpoint", [](py::object fn, std::vector<TensorImplPtr> inputs) {
        CheckpointFn cpp_fn = [fn](std::vector<TensorImplPtr> args) -> TensorImplPtr {
            py::gil_scoped_acquire acquire;
            py::list py_args;
            for (auto& a : args) py_args.append(a);
            py::object result = fn(py_args);
            return result.cast<TensorImplPtr>();
        };
        return checkpoint(std::move(cpp_fn), std::move(inputs));
    });

    // ── 版本 ─────────────────────────────────────
    m.attr("__version__") = "0.5.0";

}