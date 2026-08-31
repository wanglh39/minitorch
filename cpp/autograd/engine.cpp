// Engine 实现（Ch8 C++ autograd）

#include "autograd/engine.h"
#include "c10/thread_pool.h"
#include "autograd/grad_mode.h"
#include "aten/ops.h"
#include "autograd/ops.h"
#include "autograd/profiler.h"
#include "c10/allocator.h"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <functional>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cmath>

namespace minitorch {

// 检测张量是否包含 NaN/Inf
static bool has_nan_or_inf(const TensorImplPtr& t) {
    if (!t) return false;
    auto data = t->to_vector();
    for (auto v : data) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

// anomaly 检测：若开启 anomaly check 且梯度含 NaN/Inf 则抛异常
static void check_anomaly(const TensorImplPtr& grad) {
    if (is_anomaly_check_enabled() && has_nan_or_inf(grad)) {
        throw std::runtime_error("Anomaly detected: NaN or Inf in gradient");
    }
}

static std::vector<NodePtr> topological_sort(const NodePtr& root) {
    std::vector<NodePtr> topo;
    std::unordered_map<Node*, bool> visited;

    std::function<void(const NodePtr&)> dfs = [&](const NodePtr& node) {
        if (!node || visited[node.get()]) return;
        visited[node.get()] = true;
        for (const auto& edge : node->next_edges) {
            dfs(edge);
        }
        topo.push_back(node);
    };

    dfs(root);
    return topo;
}

void run_backward(NodePtr root,
                   TensorImplPtr root_grad,
                   bool retain_graph,
                   bool retain_grad,
                   bool create_graph) {
    std::unique_ptr<NoGradGuard> no_grad;
    if (!create_graph) no_grad = std::make_unique<NoGradGuard>();

    auto topo = topological_sort(root);


    std::unordered_map<Node*, TensorImplPtr> grad_map;
    grad_map[root.get()] = root_grad;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        auto& node = *it;
        auto grad_it = grad_map.find(node.get());
        if (grad_it == grad_map.end() || !grad_it->second) continue;

        TensorImplPtr grad = grad_it->second;

        auto& profiler = get_global_profiler();
        auto t0 = std::chrono::high_resolution_clock::now();
        size_t mem_before = get_global_allocator().total_allocated();

        if (node->is_accumulate_grad()) {
            node->apply(grad);
            check_anomaly(grad);
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (profiler.enabled())
                profiler.record(node->name, us, mem_before,
                                get_global_allocator().total_allocated(), 0);
            continue;
        }

        if (retain_grad && node->output) {
            if (!node->output->grad()) {
                node->output->set_grad(grad);
            } else if (create_graph) {
                node->output->set_grad(autograd::add(node->output->grad(), grad));
            } else {
                auto existing = node->output->grad();
                ops::add_inplace(existing, grad);
                node->output->set_grad(existing);
            }
        }

        auto grads = node->apply(grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (profiler.enabled())
            profiler.record(node->name, us, mem_before,
                            get_global_allocator().total_allocated(), 0);
        if (grads.empty()) continue;

        for (size_t i = 0; i < node->next_edges.size() && i < grads.size(); ++i) {
            auto& edge = node->next_edges[i];
            auto& g = grads[i];
            if (!edge || !g) continue;
            check_anomaly(g);
            auto prev_it = grad_map.find(edge.get());
            if (prev_it == grad_map.end() || !prev_it->second) {
                grad_map[edge.get()] = g;
            } else if (create_graph) {
                grad_map[edge.get()] = autograd::add(prev_it->second, g);
            } else {
                ops::add_inplace(prev_it->second, g);
            }
        }
    }

    if (!retain_graph) {
        for (auto& node : topo) {
            node->next_edges.clear();
        }
    }
}

// ── 多线程版反向传播 ──────────────────────────────────

void run_backward_mt(NodePtr root,
                     TensorImplPtr root_grad,
                     bool retain_graph,
                     bool retain_grad,
                     int num_threads,
                     bool create_graph) {
    auto topo = topological_sort(root);

    // Node* → index
    std::unordered_map<Node*, size_t> node_index;
    for (size_t i = 0; i < topo.size(); ++i) {
        node_index[topo[i].get()] = i;
    }

    // 计算每个 Node 的入度（有多少后继会向它投递梯度）
    std::vector<std::atomic<int>> dep_count(topo.size());
    for (auto& dc : dep_count) dc.store(0);
    for (auto& node : topo) {
        for (auto& edge : node->next_edges) {
            if (edge) dep_count[node_index[edge.get()]]++;
        }
    }

    // 共享状态
    std::mutex grad_mutex;
    std::unordered_map<Node*, TensorImplPtr> grad_map;
    grad_map[root.get()] = root_grad;

    size_t nthreads = num_threads > 0
        ? static_cast<size_t>(num_threads)
        : ThreadPool::default_num_threads();
    ThreadPool pool(nthreads);
    std::atomic<size_t> remaining(topo.size());

    // 处理单个 Node 的任务函数
    std::function<void(NodePtr)> process_node = [&](NodePtr node) {
        std::unique_ptr<NoGradGuard> no_grad;
        if (!create_graph) no_grad = std::make_unique<NoGradGuard>();

        // 取梯度
        TensorImplPtr grad;
        {
            std::lock_guard<std::mutex> lock(grad_mutex);
            auto it = grad_map.find(node.get());
            if (it != grad_map.end()) grad = it->second;
        }

        if (grad) {
            if (node->is_accumulate_grad()) {
                node->apply(grad);
                check_anomaly(grad);
            } else {
                if (retain_grad && node->output) {
                    std::lock_guard<std::mutex> lock(grad_mutex);
                    if (!node->output->grad()) {
                        node->output->set_grad(grad);
                    } else if (create_graph) {
                        node->output->set_grad(autograd::add(node->output->grad(), grad));
                    } else {
                        auto existing = node->output->grad();
                        ops::add_inplace(existing, grad);
                        node->output->set_grad(existing);
                    }
                }

                auto grads = node->apply(grad);
                if (!grads.empty()) {
                    std::lock_guard<std::mutex> lock(grad_mutex);
                    for (size_t i = 0; i < node->next_edges.size() && i < grads.size(); ++i) {
                        auto& edge = node->next_edges[i];
                        auto& g = grads[i];
                        if (!edge || !g) continue;
                        check_anomaly(g);
                        auto it = grad_map.find(edge.get());
                        if (it == grad_map.end() || !it->second) {
                            grad_map[edge.get()] = g;
                        } else if (create_graph) {
                            grad_map[edge.get()] = autograd::add(it->second, g);
                        } else {
                            ops::add_inplace(it->second, g);
                        }
                    }
                }
            }
        }

        // 通知 next_edges：此来源已处理
        for (auto& edge : node->next_edges) {
            if (!edge) continue;
            size_t idx = node_index[edge.get()];
            if (dep_count[idx].fetch_sub(1) == 1) {
                pool.submit([&, edge] { process_node(edge); });
            }
        }

        remaining.fetch_sub(1);
    };

    // 启动：root 的 dep_count 应为 0
    pool.submit([&] { process_node(root); });
    pool.wait_all();

    if (!retain_graph) {
        for (auto& node : topo) {
            node->next_edges.clear();
        }
    }
}

} // namespace minitorch