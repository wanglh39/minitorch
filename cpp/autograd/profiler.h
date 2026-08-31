// Profiler：autograd 性能分析器（Ch9 深入）
//
// 记录每个 Node 的执行时间和内存分配。
// 对应真实 PyTorch 的 torch/csrc/autograd/profiler.h + python_autograd.cpp。
// 用法：
//   {
//     Profiler prof;
//     prof.start();
//     ... backward() ...
//     prof.stop();
//     prof.print_summary();
//   }

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <iostream>
#include <mutex>
#include <iomanip>
#include <unordered_map>
#include <algorithm>

namespace minitorch {

struct ProfileEvent {
    std::string node_name;
    double duration_us;      // 微秒
    size_t memory_before;    // 执行前分配量
    size_t memory_after;     // 执行后分配量
    int64_t thread_id;       // 执行线程
};

class Profiler {
public:
    Profiler() : enabled_(false) {}

    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
        enabled_ = true;
    }

    void stop() {
        enabled_ = false;
    }

    bool enabled() const { return enabled_; }

    void record(const std::string& name, double duration_us,
                size_t mem_before, size_t mem_after, int64_t tid) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back({name, duration_us, mem_before, mem_after, tid});
    }

    const std::vector<ProfileEvent>& events() const { return events_; }

    void print_summary(std::ostream& os = std::cerr) const {
        std::lock_guard<std::mutex> lock(mutex_);
        os << "\n=== Autograd Profiler Summary ===\n";
        os << "Total events: " << events_.size() << "\n\n";

        double total_time = 0;
        size_t peak_mem = 0;
        std::unordered_map<std::string, double> name_to_time;
        std::unordered_map<std::string, int> name_to_count;

        for (const auto& e : events_) {
            total_time += e.duration_us;
            peak_mem = std::max(peak_mem, e.memory_after);
            name_to_time[e.node_name] += e.duration_us;
            name_to_count[e.node_name]++;
        }

        os << "Total time: " << total_time << " us\n";
        os << "Peak memory: " << peak_mem << " doubles\n\n";

        os << "Per-node breakdown:\n";
        os << "  " << std::left << std::setw(25) << "Node"
           << std::right << std::setw(8) << "Count"
           << std::setw(12) << "Time(us)"
           << std::setw(10) << "%" << "\n";
        os << "  " << std::string(55, '-') << "\n";

        for (const auto& [name, time] : name_to_time) {
            double pct = total_time > 0 ? (time / total_time * 100.0) : 0;
            os << "  " << std::left << std::setw(25) << name
               << std::right << std::setw(8) << name_to_count[name]
               << std::setw(12) << std::fixed << std::setprecision(1) << time
               << std::setw(9) << std::fixed << std::setprecision(1) << pct << "\n";
        }
        os << "\n";
    }

private:
    std::atomic<bool> enabled_;
    std::vector<ProfileEvent> events_;
    mutable std::mutex mutex_;
};

// 全局 profiler 管理
Profiler& get_global_profiler();
void set_profiler_enabled(bool v);

} // namespace minitorch