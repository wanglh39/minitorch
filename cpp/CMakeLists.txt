cmake_minimum_required(VERSION 3.18)
project(minitorch_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── pybind11 ──────────────────────────────────────
# 需要先安装 pybind11: pip install pybind11
# 然后告诉 CMake pybind11 的 CMake 目录:
#   cmake .. -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
# 或用 find_package 自动发现:
find_package(pybind11 CONFIG REQUIRED)

# ── 源文件 ────────────────────────────────────────
# 三层分离：c10（核心抽象）/ aten（算子）/ autograd（自动微分）
set(MINITORCH_SOURCES
    c10/storage.cpp
    c10/allocator.cpp
    c10/tensor.cpp
    aten/ops.cpp
    autograd/profiler.cpp
    autograd/node.cpp
    autograd/engine.cpp
    autograd/ops.cpp
    autograd/checkpoint.cpp
    binding/module.cpp
)

# ── 编译扩展模块 ──────────────────────────────────
pybind11_add_module(_C_ext ${MINITORCH_SOURCES})

# ── 包含目录 ──────────────────────────────────────
# 以 cpp/ 为根，#include 使用 "c10/xxx.h" / "aten/xxx.h" / "autograd/xxx.h"
target_include_directories(_C_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ── 安装 ──────────────────────────────────────────
# 安装到 src/minitorch/ 下，让 Python 能 import minitorch._C_ext
install(TARGETS _C_ext
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/../src/minitorch
)

# ── CUDA 支持（Ch9，可选）─────────────────────────
option(MINITORCH_ENABLE_CUDA "Enable CUDA support (Ch9)" OFF)
if(MINITORCH_ENABLE_CUDA)
    enable_language(CUDA)
    # CUDA 源文件在 Ch9 添加
    # target_sources(_C_ext PRIVATE aten/native/cuda/ops_cuda.cu)
    # target_compile_definitions(_C_ext PRIVATE MINITORCH_HAS_CUDA)
    message(STATUS "CUDA support enabled (Ch9)")
endif()
