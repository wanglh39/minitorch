# 与真实 PyTorch 的差异

!!! warning "必读"
    本页面明确列出 minitorch 相对真实 PyTorch 的简化与差异，避免产生错误心智模型。

## 原理覆盖度 vs 功能覆盖度

| 维度 | 本项目覆盖 | 真实 PyTorch |
|---|---|---|
| 原理覆盖度 | ~95%（核心设计思想） | 100% |
| 功能覆盖度 | ~5–10%（最小可用 API 集） | 100%（数百算子、上百 nn 层） |

本项目**不追求功能广度**，只实现"能跑通 MNIST + 小 Transformer"的最小必要集。

## 主要简化

| 子系统 | 本项目 | 真实 PyTorch | 简化理由 |
|---|---|---|---|
| Storage | C++ `shared_ptr<Storage>` | `intrusive_ptr<StorageImpl>` | 教学无需 intrusive 引用计数细节 |
| autograd Engine | 多线程 + 依赖计数调度 | 多线程 + ReadyQueue + CUDA stream | 并行调度属工程优化，非原理 |
| Allocator | DefaultAllocator + PoolAllocator | CUDACachingAllocator 等多策略 | 内存池属工程优化 |
| Dispatcher | Python dict 路由 + C++ dispatch table | dispatch key 链 + kernel 函数指针 | 完整 dispatch key 链放附录 A5 |
| 算子数量 | ~40 个（含 autograd 建图） | 数百个 | 算子无新原理，按需补 |
| DataLoader | 简化管道 | pin_memory/prefetch/多队列 | 管道复杂度非教学重点 |
| 量化/分布式/编译 | 附录或概念 | 完整实现 | 专题，按需选读 |
| TorchScript | 简述（已被 compile 取代） | 完整 | 时代已过 |

## 阅读建议

1. 先读本页建立边界认知。
2. 每章末尾的"与真实 PyTorch 对照"给出源码指针，可对照阅读。
3. 想深入某简化项，查附录对应章节。