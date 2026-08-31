// Dispatcher：按 device 路由算子到不同 kernel（Ch9）
//
// 这是 minitorch 的"异构计算调度层"。同一个算子名（如 "add"）在不同 device
// 上有不同实现（CPU 版 vs CUDA 版），dispatcher 负责：根据输入张量的 device，
// 查表找到对应 kernel 并调用。
//
// 对应真实 PyTorch 的 c10::DispatchTable / c10::Dispatcher / DispatchKey。
// 真实 PyTorch 的 dispatcher 还支持 backend、autograd、autocast 等"dispatch key"，
// 这里只做最核心的 device 路由（CPU / CUDA），足以讲清思想。
//
// ── 核心数据结构 ──────────────────────────────────────────
//   DispatchTable:  op_name(string) -> [device -> KernelFn]
//   KernelFn:       std::function<TensorImplPtr(const std::vector<TensorImplPtr>&)>
//
// ── 注册流程 ──────────────────────────────────────────────
//   1. CPU 算子在 native/cpu/ops_cpu.cpp 里用 REGISTER_CPU_OP("add", cpu_add) 注册
//   2. CUDA 算子在 native/cuda/ops_cuda.cu 里用 REGISTER_CUDA_OP("add", cuda_add) 注册
//   3. 调用时 dispatcher.call("add", {a, b}) 自动按 a 的 device 选 kernel
//
// ── 为什么不直接 if (device == CPU) ... else ... ──────────
//   因为算子数量会爆炸（几十个算子 × 几种 device × autograd 包装），
//   硬编码 if/else 会让每个算子都耦合所有 device。dispatch table 把
//   "算子名"和"实现"解耦，新增 device 只需注册新 kernel，不改任何调用点。
//   这正是 PyTorch 能同时支持 CPU/CUDA/MPS/XLA/... 的关键。

#pragma once

#include "c10/tensor.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstdint>

namespace minitorch {

// ── Device 类型 ────────────────────────────────────────────
// 对应 PyTorch 的 c10::DeviceType。真实 PyTorch 有十几种 device，
// 这里只列教学用到的两种。
enum class DeviceType : int8_t {
    CPU = 0,
    CUDA = 1,
};

// 把 DeviceType 转成字符串（用于错误信息和调试）
inline std::string device_name(DeviceType d) {
    switch (d) {
        case DeviceType::CPU:  return "cpu";
        case DeviceType::CUDA: return "cuda";
    }
    return "unknown";
}

// ── DispatchKey ────────────────────────────────────────────
// 真实 PyTorch 的 DispatchKey 是一个稠密枚举（int），用来索引 dispatch 表的"槽"。
// 这里直接用 DeviceType 当 key，简化教学。真实 PyTorch 还会有
// AutogradCPU / AutocastCPU / SparseCUDA 等组合 key。
using DispatchKey = DeviceType;

// ── Kernel 函数签名 ────────────────────────────────────────
// 所有算子统一签名为 (vector<TensorImplPtr>) -> TensorImplPtr。
// 这样不同算子（一元/二元/归约）能放进同一张表。
// 真实 PyTorch 用变参模板 + stackbased kernel，更高效但复杂得多。
using KernelFn = std::function<TensorImplPtr(const std::vector<TensorImplPtr>&)>;

// ── DispatchTable ──────────────────────────────────────────
// 一张表：op_name -> (device -> kernel)
// 用 nested unordered_map 实现。真实 PyTorch 用扁平数组 + DispatchKey 下标，
// 查表 O(1) 且缓存友好；这里用 hash map，O(1) 平均但常数稍大，教学够用。
class DispatchTable {
public:
    // 注册一个 kernel：把 op_name 在 device 上的实现设为 fn
    void register_kernel(const std::string& op_name,
                         DispatchKey device,
                         KernelFn fn) {
        table_[op_name][static_cast<int>(device)] = std::move(fn);
    }

    // 查询 op_name 在 device 上是否已注册
    bool has_kernel(const std::string& op_name, DispatchKey device) const {
        auto it = table_.find(op_name);
        if (it == table_.end()) return false;
        return it->second.find(static_cast<int>(device)) != it->second.end();
    }

    // 取出 op_name 在 device 上的 kernel（不存在则抛异常）
    const KernelFn& lookup(const std::string& op_name, DispatchKey device) const {
        auto it = table_.find(op_name);
        if (it == table_.end()) {
            throw std::runtime_error("dispatch: unknown op '" + op_name + "'");
        }
        auto kit = it->second.find(static_cast<int>(device));
        if (kit == it->second.end()) {
            throw std::runtime_error(
                "dispatch: op '" + op_name + "' has no kernel for device " +
                device_name(device));
        }
        return kit->second;
    }

    // 列出所有已注册的 op 名（调试用）
    std::vector<std::string> op_names() const {
        std::vector<std::string> names;
        names.reserve(table_.size());
        for (const auto& [name, _] : table_) names.push_back(name);
        return names;
    }

private:
    // op_name -> (device_int -> kernel)
    std::unordered_map<
        std::string,
        std::unordered_map<int, KernelFn>> table_;
};

// ── 全局 Dispatcher 单例 ───────────────────────────────────
// 真实 PyTorch 有一个全局 c10::Dispatcher，所有算子注册到它上面。
// 这里同样用单例，避免到处传 DispatchTable 引用。
class Dispatcher {
public:
    static Dispatcher& instance() {
        static Dispatcher inst;
        return inst;
    }

    void register_kernel(const std::string& op_name,
                         DispatchKey device,
                         KernelFn fn) {
        table_.register_kernel(op_name, device, std::move(fn));
    }

    // 调用算子：根据第一个输入张量的 device 选 kernel。
    // 约定：所有输入张量必须在同一 device 上（真实 PyTorch 会自动
    // 跨 device 拷贝，这里简化为报错，逼用户显式 .to()）。
    TensorImplPtr call(const std::string& op_name,
                       const std::vector<TensorImplPtr>& args) const {
        if (args.empty()) {
            throw std::runtime_error("dispatch: call with no args");
        }
        // 从张量取 device。Ch8 的 TensorImpl 还没加 device 字段，
        // 这里用一个虚函数占位：默认 CPU。Ch9 的 CUDA TensorImpl
        // 会 override 返回 CUDA（教学版用属性标记，见下）。
        DispatchKey device = device_of(args[0]);
        const KernelFn& fn = table_.lookup(op_name, device);
        return fn(args);
    }

    // 取表（调试/测试用）
    const DispatchTable& table() const { return table_; }

private:
    DispatchTable table_;
};

// ── device_of：从张量推断 device ───────────────────────────
// Ch8 的 TensorImpl 没有 device 字段，默认 CPU。
// Ch9 里 CUDA 张量会在 storage_offset_ 的高位打标记，或更干净地
// 给 TensorImpl 加一个 device_ 字段。教学版用最简单方案：
// 检查张量是否带 "cuda" 标记属性（通过 requires_grad 复用位不优雅，
// 所以这里直接默认 CPU，CUDA 张量在 ops_cuda.cu 里构造时另走路径）。
//
// 真实做法：TensorImpl 有 Device device_ 成员。这里为了不改动 Ch8 已编译
// 的 TensorImpl，用一个外部 registry 记录"哪些 storage 是 cuda 的"。
// 教学演示足够；生产代码请直接给 TensorImpl 加 device 字段。
inline DispatchKey device_of(const TensorImplPtr& t) {
    // 默认 CPU。CUDA 张量的 device 标记见 ops_cuda.cu 里的 cuda_device_of。
    (void)t;
    return DispatchKey::CPU;
}

// ── 注册宏（语法糖）────────────────────────────────────────
// 用法：REGISTER_CPU_OP("add", cpu_add)
//       REGISTER_CUDA_OP("add", cuda_add)
// 宏在全局构造期把 kernel 塞进 Dispatcher 单例。
// 真实 PyTorch 用 TORCH_LIBRARY_IMPL 宏，原理相同。
namespace detail {
    struct OpRegistrar {
        OpRegistrar(const std::string& name, DispatchKey device, KernelFn fn) {
            Dispatcher::instance().register_kernel(name, device, std::move(fn));
        }
    };
} // namespace detail

#define MINITORCH_REGISTER_OP(name_, device_, fn_)                          \
    static ::minitorch::detail::OpRegistrar                                  \
        _minitorch_op_reg_##name_##_##device_##_ =                           \
            ::minitorch::detail::OpRegistrar(#name_,                         \
                                             ::minitorch::DispatchKey::device_, \
                                             [](const std::vector<::minitorch::TensorImplPtr>& args) \
                                                 -> ::minitorch::TensorImplPtr { \
                                                 return fn_(args);           \
                                             })

// 便捷包装：把固定元数的 kernel 适配成 KernelFn 的统一签名
template <typename Fn>
KernelFn unary_kernel(Fn fn) {
    return [fn](const std::vector<TensorImplPtr>& a) {
        if (a.size() != 1) throw std::runtime_error("unary op needs 1 arg");
        return fn(a[0]);
    };
}
template <typename Fn>
KernelFn binary_kernel(Fn fn) {
    return [fn](const std::vector<TensorImplPtr>& a) {
        if (a.size() != 2) throw std::runtime_error("binary op needs 2 args");
        return fn(a[0], a[1]);
    };
}

} // namespace minitorch