// CUDA kernel 实现（Ch9，教学版）
//
// 本文件是 .cu（CUDA 源文件），由 nvcc 编译。当前教学环境无 GPU，
// 但代码完整可读，配上 GPU 机器 + -DMINITORCH_ENABLE_CUDA=ON 即可编译。
//
// 对应真实 PyTorch 的 aten/native/cuda/BinaryOps.cu / ReductionOps.cu 等。
//
// ── CUDA 编程模型速览（详见 ch09 文档）──────────────────
//   - thread：最小执行单元，每个 thread 跑一份 kernel 代码，有私有寄存器
//   - block：若干 thread 组成，block 内 thread 可通过 shared memory 协作，
//             可用 __syncthreads() 同步
//   - grid：若干 block 组成，grid 是一次 kernel launch 的全部 thread
//   - 每个 thread 用内置变量定位自己：
//       threadIdx.x/y/z   block 内的 thread 索引
//       blockIdx.x/y/z    grid 内的 block 索引
//       blockDim.x/y/z    一个 block 的 thread 数
//       gridDim.x/y/z     grid 的 block 数
//     全局一维索引：int i = blockIdx.x * blockDim.x + threadIdx.x;
//   - kernel 用 __global__ 标记（从 host 调用、在 device 执行）
//   - host↔device 内存拷贝：cudaMalloc / cudaMemcpy / cudaFree
//   - 异步：kernel launch 立即返回，要 cudaStreamSynchronize 等结果
//
// ── 本文件实现 ───────────────────────────────────────────
//   - add_cuda:   c[i] = a[i] + b[i]   （逐元素加，演示最朴素的 kernel）
//   - relu_cuda:  c[i] = max(0, a[i])  （逐元素，演示分支）
//   - mul_cuda:   c[i] = a[i] * b[i]   （练习用）
//   - sum_cuda:   全局求和，演示 reduction + shared memory（进阶）
//   以及把数据在 host/device 间搬来搬去的 wrapper 函数。

// ============================================================
// 编译守卫：只在启用 CUDA 时编译
// ============================================================
#if defined(__CUDACC__) || defined(MINITORCH_HAS_CUDA)

#include <cuda_runtime.h>
#include "../../dispatcher.h"
#include "../../ops.h"   // 复用 CPU 版做 fallback / 对照

#include <stdexcept>
#include <vector>
#include <cmath>

namespace minitorch::native::cuda {

// ============================================================
// 1. 逐元素加法 kernel
// ============================================================
// __global__ 表示这是从 host 调用、在 device 执行的 kernel。
// 三个指针都指向 device 内存（GPU 显存）。
// n 是元素总数，所有 thread 共享这个只读参数。
//
// 调用约定：launch 足够 n 个 thread，每个 thread 处理一个元素，
// 越界的 thread 直接 return（这是 CUDA 的"边界守卫"标准写法）。
__global__ void add_kernel(const double* __restrict__ a,
                           const double* __restrict__ b,
                           double* __restrict__ c,
                           int n) {
    // 计算本 thread 的全局一维索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
    // 注意：没有锁、没有同步——逐元素操作各 thread 独立，天然并行
}

// ============================================================
// 2. 逐元素 ReLU kernel（演示分支）
// ============================================================
// ReLU 在 0 处导数未定义，取 x > 0（与 PyTorch 一致）。
// GPU 上的分支（if）会走"线程级预测"，warp 内分歧会串行化，
// 但 ReLU 分歧不严重，性能影响小。
__global__ void relu_kernel(const double* __restrict__ a,
                            double* __restrict__ c,
                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = a[i];
        c[i] = x > 0.0 ? x : 0.0;
    }
}

// ============================================================
// 3. 逐元素乘法 kernel（练习用，与 add 同构）
// ============================================================
__global__ void mul_kernel(const double* __restrict__ a,
                           const double* __restrict__ b,
                           double* __restrict__ c,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

// ============================================================
// 4. 全局求和 kernel（reduction，演示 shared memory + __syncthreads）
// ============================================================
// 思路：每个 block 先把本 block 范围内的元素求和到 shared memory，
// 再让 block 内第 0 个 thread 把 block 部分和写到全局数组 partial[]。
// 最后 host 上把 partial[] 加起来（或再 launch 一个 kernel 归约）。
//
// 这是经典的"两级 reduction"。真实 PyTorch 用更精巧的版本
// （warp shuffle、避免 bank conflict），这里讲清原理即可。
__global__ void sum_kernel(const double* __restrict__ a,
                           double* __restrict__ partial,
                           int n) {
    // 动态 shared memory：大小由 launch 时的第三个参数指定
    extern __shared__ double sdata[];

    int tid = threadIdx.x;                 // block 内 thread id
    int i   = blockIdx.x * blockDim.x + tid;  // 全局索引

    // 每个 thread 先把自己负责的元素读进 shared mem（越界填 0）
    double x = (i < n) ? a[i] : 0.0;
    sdata[tid] = x;
    __syncthreads();  // 等所有 thread 都写完 sdata

    // 树形归约：步长从 blockDim/2 逐次减半
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // 每轮归约后同步
    }

    // block 的总和在 sdata[0]，由第 0 个 thread 写到 partial[blockIdx.x]
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// ============================================================
// 5. host 端 wrapper：负责内存拷贝 + launch + 同步
// ============================================================
// 这层是 host 代码（虽然写在 .cu 里），被 dispatcher 调用。
// 它把 TensorImpl 的数据搬到 GPU、launch kernel、把结果搬回 host、
// 包成新的 TensorImpl 返回。
//
// 教学版为了简单，每次调用都做 host↔device 来回拷贝（计算完搬回 CPU）。
// 真实 PyTorch 的张量常驻 GPU，不来回拷，只在需要时（.cpu()）才搬。
// 这里来回拷是为了能复用 Ch8 的 TensorImpl（它只有 host Storage）。

// 逐元素加法的 host wrapper
TensorImplPtr cuda_add(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (a->shape() != b->shape()) {
        // 教学版不做广播，要求 shape 完全一致
        throw std::runtime_error("cuda_add: shape 必须一致（教学版不支持广播）");
    }
    int n = static_cast<int>(a->numel());

    // 取 host 数据（TensorImpl 的 storage 在 host 内存）
    std::vector<double> ha = a->to_vector();
    std::vector<double> hb = b->to_vector();
    std::vector<double> hc(static_cast<size_t>(n));

    // device 指针
    double *da = nullptr, *db = nullptr, *dc = nullptr;
    cudaMalloc(&da, n * sizeof(double));
    cudaMalloc(&db, n * sizeof(double));
    cudaMalloc(&dc, n * sizeof(double));

    // host -> device
    cudaMemcpy(da, ha.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    // launch：每 block 256 个 thread，block 数 = ceil(n / 256)
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(da, db, dc, n);

    // kernel launch 是异步的，这里 cudaMemcpy 会隐式同步（它等之前的 kernel）
    cudaMemcpy(hc.data(), dc, n * sizeof(double), cudaMemcpyDeviceToHost);

    // 释放显存（教学版每次分配/释放；真实场景用 caching allocator）
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return make_tensor(hc, a->shape());
}

// ReLU 的 host wrapper
TensorImplPtr cuda_relu(const TensorImplPtr& a) {
    int n = static_cast<int>(a->numel());
    std::vector<double> ha = a->to_vector();
    std::vector<double> hc(static_cast<size_t>(n));

    double *da = nullptr, *dc = nullptr;
    cudaMalloc(&da, n * sizeof(double));
    cudaMalloc(&dc, n * sizeof(double));
    cudaMemcpy(da, ha.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(da, dc, n);

    cudaMemcpy(hc.data(), dc, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(dc);

    return make_tensor(hc, a->shape());
}

// 乘法的 host wrapper（练习）
TensorImplPtr cuda_mul(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (a->shape() != b->shape()) {
        throw std::runtime_error("cuda_mul: shape 必须一致");
    }
    int n = static_cast<int>(a->numel());
    std::vector<double> ha = a->to_vector();
    std::vector<double> hb = b->to_vector();
    std::vector<double> hc(static_cast<size_t>(n));

    double *da = nullptr, *db = nullptr, *dc = nullptr;
    cudaMalloc(&da, n * sizeof(double));
    cudaMalloc(&db, n * sizeof(double));
    cudaMalloc(&dc, n * sizeof(double));
    cudaMemcpy(da, ha.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads>>>(da, db, dc, n);

    cudaMemcpy(hc.data(), dc, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return make_tensor(hc, a->shape());
}

// 全局 sum 的 host wrapper（演示 reduction）
TensorImplPtr cuda_sum(const TensorImplPtr& a) {
    int n = static_cast<int>(a->numel());
    std::vector<double> ha = a->to_vector();

    double *da = nullptr, *dpartial = nullptr;
    cudaMalloc(&da, n * sizeof(double));
    cudaMemcpy(da, ha.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    cudaMalloc(&dpartial, blocks * sizeof(double));

    // 第三个参数 = threads * sizeof(double) 是动态 shared memory 大小
    sum_kernel<<<blocks, threads, threads * sizeof(double)>>>(da, dpartial, n);

    // 取回每个 block 的部分和，在 host 上累加
    std::vector<double> partial(static_cast<size_t>(blocks));
    cudaMemcpy(partial.data(), dpartial, blocks * sizeof(double), cudaMemcpyDeviceToHost);
    double s = 0.0;
    for (double v : partial) s += v;

    cudaFree(da); cudaFree(dpartial);
    return make_tensor({s}, {});
}

// ============================================================
// 6. 注册到 dispatcher
// ============================================================
void register_all_cuda_ops() {
    auto& d = Dispatcher::instance();
    using namespace minitorch;
    d.register_kernel("add", DispatchKey::CUDA,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return cuda_add(a, b); }));
    d.register_kernel("mul", DispatchKey::CUDA,
        binary_kernel([](const TensorImplPtr& a, const TensorImplPtr& b) { return cuda_mul(a, b); }));
    d.register_kernel("relu", DispatchKey::CUDA,
        unary_kernel([](const TensorImplPtr& a) { return cuda_relu(a); }));
    d.register_kernel("sum", DispatchKey::CUDA,
        unary_kernel([](const TensorImplPtr& a) { return cuda_sum(a); }));
}

} // namespace minitorch::native::cuda

#endif // defined(__CUDACC__) || defined(MINITORCH_HAS_CUDA)