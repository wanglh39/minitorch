// Allocator 全局管理实现（Ch8 第三批⑦）

#include "c10/allocator.h"
#include <memory>

namespace minitorch {

static std::shared_ptr<Allocator>& global_allocator_ref() {
    static std::shared_ptr<Allocator> instance = std::make_shared<DefaultAllocator>();
    return instance;
}

Allocator& get_global_allocator() {
    return *global_allocator_ref();
}

void set_global_allocator(std::shared_ptr<Allocator> alloc) {
    global_allocator_ref() = std::move(alloc);
}

} // namespace minitorch