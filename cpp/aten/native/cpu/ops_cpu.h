// CPU 算子注册到 dispatcher（Ch9）
//
// 把 Ch8 在 ops.cpp 里实现的 CPU 算子（add/sub/mul/div/neg/relu/sum/mean/matmul）
// 包装成统一签名，注册到全局 Dispatcher 的 CPU 槽。
//
// 对应真实 PyTorch 的 aten/native/ 目录 + RegisterCPU.cpp。
// 真实 PyTorch 用代码生成（gen.py）从 native_functions.yaml 批量生成注册代码，
// 这里手写注册，逻辑等价。

#pragma once

#include "../../dispatcher.h"

namespace minitorch::native::cpu {

// 注册所有 CPU 算子到 Dispatcher。在 module.cpp 的 PYBIND11_MODULE 里调用一次。
void register_all_cpu_ops();

} // namespace minitorch::native::cpu