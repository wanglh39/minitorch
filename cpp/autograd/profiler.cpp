// Profiler 全局管理实现（Ch9 深入）

#include "autograd/profiler.h"

namespace minitorch {

static Profiler& global_profiler_ref() {
    static Profiler instance;
    return instance;
}

Profiler& get_global_profiler() {
    return global_profiler_ref();
}

void set_profiler_enabled(bool v) {
    if (v) {
        global_profiler_ref().start();
    } else {
        global_profiler_ref().stop();
    }
}

} // namespace minitorch