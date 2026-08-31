// Allocator：内存分配器接口（Ch8 第三批⑦）
//
// 对应真实 PyTorch 的 c10::Allocator / c10::CPUAllocator。
// 提供 allocate/deallocate 接口，支持统计和内存池。
// Storage 通过全局 allocator 分配内存。

#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <string>

namespace minitorch {

class Allocator {
public:
    virtual ~Allocator() = default;

    // 分配 size 个 double 的内存，返回指针
    virtual double* allocate(size_t size) = 0;

    // 释放之前分配的内存
    virtual void deallocate(double* ptr, size_t size) = 0;

    // 统计信息
    virtual size_t total_allocated() const = 0;
    virtual size_t peak_allocated() const = 0;
    virtual size_t num_allocations() const = 0;

    virtual std::string name() const = 0;
};

// 默认分配器：直接 malloc/free，带统计
class DefaultAllocator : public Allocator {
public:
    double* allocate(size_t size) override {
        if (size == 0) return nullptr;
        double* ptr = new double[size]();
        current_ += size;
        peak_ = std::max(peak_.load(), current_.load());
        ++num_allocs_;
        return ptr;
    }

    void deallocate(double* ptr, size_t size) override {
        if (!ptr) return;
        delete[] ptr;
        current_ -= size;
    }

    size_t total_allocated() const override { return current_; }
    size_t peak_allocated() const override { return peak_; }
    size_t num_allocations() const override { return num_allocs_; }
    std::string name() const override { return "DefaultAllocator"; }

private:
    std::atomic<size_t> current_{0};
    std::atomic<size_t> peak_{0};
    std::atomic<size_t> num_allocs_{0};
};

// 内存池分配器：维护空闲块列表，重用已释放的内存
class PoolAllocator : public Allocator {
public:
    explicit PoolAllocator(size_t pool_threshold = 1024 * 1024)
        : threshold_(pool_threshold) {}

    ~PoolAllocator() override {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [ptr, size] : free_blocks_) {
            delete[] ptr;
        }
    }

    double* allocate(size_t size) override {
        if (size == 0) return nullptr;

        std::lock_guard<std::mutex> lock(mutex_);

        // 在空闲列表中找恰好匹配的块
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (it->second == size) {
                double* ptr = it->first;
                free_blocks_.erase(it);
                current_ += size;
                peak_ = std::max(peak_, current_);
                ++num_allocs_;
                ++pool_hits_;
                return ptr;
            }
        }

        // 没找到，分配新内存
        double* ptr = new double[size]();
        current_ += size;
        peak_ = std::max(peak_, current_);
        ++num_allocs_;
        ++pool_misses_;
        return ptr;
    }

    void deallocate(double* ptr, size_t size) override {
        if (!ptr) return;

        std::lock_guard<std::mutex> lock(mutex_);
        current_ -= size;

        // 如果池子未超限，放入空闲列表
        if (pooled_bytes_ + size <= threshold_) {
            free_blocks_.emplace_back(ptr, size);
            pooled_bytes_ += size;
        } else {
            delete[] ptr;
        }
    }

    size_t total_allocated() const override { return current_; }
    size_t peak_allocated() const override { return peak_; }
    size_t num_allocations() const override { return num_allocs_; }
    std::string name() const override { return "PoolAllocator"; }

    size_t pool_hits() const { return pool_hits_; }
    size_t pool_misses() const { return pool_misses_; }
    size_t pooled_bytes() const { return pooled_bytes_; }
    size_t pool_size() const { return free_blocks_.size(); }

private:
    std::mutex mutex_;
    std::vector<std::pair<double*, size_t>> free_blocks_;
    size_t current_ = 0;
    size_t peak_ = 0;
    size_t num_allocs_ = 0;
    size_t pooled_bytes_ = 0;
    size_t pool_hits_ = 0;
    size_t pool_misses_ = 0;
    size_t threshold_;
};

// 全局分配器管理
Allocator& get_global_allocator();
void set_global_allocator(std::shared_ptr<Allocator> alloc);

} // namespace minitorch