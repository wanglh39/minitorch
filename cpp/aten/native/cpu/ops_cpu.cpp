// CPU 算子注册实现（Ch9）
//
// 把 Ch8 的 minitorch::ops::add 等包装成 KernelFn，注册到 Dispatcher 的 CPU 槽。
// 这层本身不做计算，只是"接线"——计算还在 ops.cpp 里。
// 这样 Ch8 的算子代码零改动就能接入 Ch9 的 dispatcher。

#include "ops_cpu.h"
#include "../../ops.h"

namespace minitorch::native::cpu {

using namespace minitorch;
using namespace minitorch::ops;

void register_all_cpu_ops() {
    auto& d = Dispatcher::instance();

    // 逐元素二元算子：用 binary_kernel 适配签名
    d.register_kernel("add", DispatchKey::CPU,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return add(a, b); }));
    d.register_kernel("sub", DispatchKey::CPU,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return sub(a, b); }));
    d.register_kernel("mul", DispatchKey::CPU,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return mul(a, b); }));
    d.register_kernel("div", DispatchKey::CPU,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return div(a, b); }));

    // 一元算子
    d.register_kernel("neg", DispatchKey::CPU,
        unary_kernel([](const TensorImplPtr& a) { return neg(a); }));
    d.register_kernel("relu", DispatchKey::CPU,
        unary_kernel([](const TensorImplPtr& a) { return relu(a); }));

    // 归约：sum/mean 带额外参数 dim/keepdim，这里注册"全局归约"版本（dim=-1）
    d.register_kernel("sum", DispatchKey::CPU,
        unary_kernel([](const TensorImplPtr& a) { return sum(a, -1, false); }));
    d.register_kernel("mean", DispatchKey::CPU,
        unary_kernel([](const TensorImplPtr& a) { return mean(a, -1, false); }));

    // 矩阵乘法
    d.register_kernel("matmul", DispatchKey::CPU,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return matmul(a, b); }));
}

} // namespace minitorch::native::cpu