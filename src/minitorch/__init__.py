"""minitorch: 从零实现的迷你版 PyTorch，用于学习底层原理与设计原则。

渐进双层架构：
  阶段一（本包）：纯 Python 实现 autograd / nn / optim 等核心机制。
  阶段二（cpp/）：C++ 核心层 + pybind11 绑定，前端 API 不变。
"""

import ctypes as _ctypes
import os as _os

# 自动加载 C++ 扩展（Ch8）：把 MinGW 运行时 DLL 加入搜索路径
_mingw_bin = r"C:\mingw64\bin"
if _os.path.isdir(_mingw_bin):
    _os.environ["PATH"] = _mingw_bin + _os.pathsep + _os.environ.get("PATH", "")
    for _dll in ("libgcc_s_seh-1.dll", "libstdc++-6.dll"):
        _path = _os.path.join(_mingw_bin, _dll)
        if _os.path.isfile(_path):
            try:
                _ctypes.CDLL(_path)
            except OSError:
                pass

try:
    from . import _C_ext as _cpp_ext  # noqa: F401
    _has_cpp = True
except ImportError:
    _has_cpp = False

from .storage import Storage  # noqa: E402
from .tensor import Tensor  # noqa: E402

if _has_cpp:
    from .cpp_tensor import CppTensor as CppTensor  # noqa: E402

__version__ = "0.3.0" if _has_cpp else "0.1.0"

__all__ = ["Storage", "Tensor", "__version__"]
if _has_cpp:
    __all__.append("CppTensor")