// Storage 实现（Ch8）

#include "c10/storage.h"
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace minitorch {

Storage::Storage(size_t size) : data_(nullptr), size_(size) {
    if (size > 0) {
        data_ = get_global_allocator().allocate(size);
    }
}

Storage::Storage(const std::vector<double>& data) : data_(nullptr), size_(data.size()) {
    if (size_ > 0) {
        data_ = get_global_allocator().allocate(size_);
        std::memcpy(data_, data.data(), size_ * sizeof(double));
    }
}

Storage::Storage(std::vector<double>&& data) : data_(nullptr), size_(data.size()) {
    if (size_ > 0) {
        data_ = get_global_allocator().allocate(size_);
        std::memcpy(data_, data.data(), size_ * sizeof(double));
    }
}

Storage::Storage(const Storage& other) : data_(nullptr), size_(other.size_) {
    if (size_ > 0) {
        data_ = get_global_allocator().allocate(size_);
        std::memcpy(data_, other.data_, size_ * sizeof(double));
    }
}

Storage& Storage::operator=(const Storage& other) {
    if (this != &other) {
        if (data_) get_global_allocator().deallocate(data_, size_);
        size_ = other.size_;
        if (size_ > 0) {
            data_ = get_global_allocator().allocate(size_);
            std::memcpy(data_, other.data_, size_ * sizeof(double));
        } else {
            data_ = nullptr;
        }
    }
    return *this;
}

Storage::Storage(Storage&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        if (data_) get_global_allocator().deallocate(data_, size_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

Storage::~Storage() {
    if (data_) {
        get_global_allocator().deallocate(data_, size_);
    }
}

double& Storage::operator[](size_t idx) {
    if (idx >= size_) {
        throw std::out_of_range("Storage index out of range: " + std::to_string(idx));
    }
    return data_[idx];
}

const double& Storage::operator[](size_t idx) const {
    if (idx >= size_) {
        throw std::out_of_range("Storage index out of range: " + std::to_string(idx));
    }
    return data_[idx];
}

std::string Storage::repr() const {
    std::ostringstream oss;
    oss << "Storage([";
    for (size_t i = 0; i < size_; ++i) {
        if (i > 0) oss << ", ";
        oss << data_[i];
    }
    oss << "])";
    return oss.str();
}

} // namespace minitorch
