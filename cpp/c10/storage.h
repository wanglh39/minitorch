// Storage：一维数据缓冲区（Ch8 C++ 核心重写）
//
// 对应阶段一的 Python Storage（src/minitorch/storage.py）。
// 使用全局 Allocator 分配内存，shared_ptr 自动引用计数。
// 对应真实 PyTorch 的 c10::Storage / c10::StorageImpl。

#pragma once

#include "c10/allocator.h"
#include <cstddef>
#include <vector>
#include <memory>
#include <string>

namespace minitorch {

class Storage : public std::enable_shared_from_this<Storage> {
public:
    // 从给定大小创建，填充零
    explicit Storage(size_t size);

    // 从现有数据创建（拷贝）
    Storage(const std::vector<double>& data);
    Storage(std::vector<double>&& data);

    // 拷贝构造和赋值（深拷贝数据）
    Storage(const Storage& other);
    Storage& operator=(const Storage& other);

    // 移动构造
    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;

    ~Storage();

    // 访问
    double* data() { return data_; }
    const double* data() const { return data_; }
    size_t size() const { return size_; }

    // 元素访问（带边界检查）
    double& operator[](size_t idx);
    const double& operator[](size_t idx) const;

    // 调试
    std::string repr() const;

private:
    double* data_;
    size_t size_;
};

// 便捷工厂
inline std::shared_ptr<Storage> make_storage(size_t size) {
    return std::make_shared<Storage>(size);
}

inline std::shared_ptr<Storage> make_storage(const std::vector<double>& data) {
    return std::make_shared<Storage>(data);
}

} // namespace minitorch
