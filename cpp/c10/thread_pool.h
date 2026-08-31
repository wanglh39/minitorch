// ThreadPool：简单线程池（Ch8 C++ autograd 多线程引擎）
//
// 对应真实 PyTorch 的 torch/csrc/autograd/engine.cpp 中的线程池。
// 固定数量 worker 线程从任务队列拉取 std::function<void()> 执行。
// 主线程通过 submit() 投递任务，wait_all() 等待全部完成。

#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <vector>
#include <atomic>

namespace minitorch {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) : stop_(false), active_(0) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push(std::move(task));
            ++active_;
        }
        cv_.notify_one();
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return active_ == 0 && tasks_.empty(); });
    }

    static size_t default_num_threads() {
        unsigned n = std::thread::hardware_concurrency();
        return n > 0 ? n : 4;
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --active_;
                if (active_ == 0 && tasks_.empty()) {
                    done_cv_.notify_all();
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    bool stop_;
    std::atomic<int> active_;
};

} // namespace minitorch