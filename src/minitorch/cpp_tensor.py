"""CppTensor：C++ 后端的 Python 统一包装层。

包装 _C_ext.TensorImpl，提供与 minitorch.Tensor 相同的接口。
算子委托给 _C_ext.autograd_*（建图在 C++ 端完成）。
用户可通过 CppTensor 使用 C++ 后端，无需直接操作 _C_ext。

用法：
    from minitorch.cpp_tensor import CppTensor
    a = CppTensor([1.0, 2.0, 3.0], requires_grad=True)
    b = CppTensor([4.0, 5.0, 6.0], requires_grad=True)
    c = (a * b).sum()
    c.backward()
    print(a.grad)  # [4, 5, 6]
"""

from __future__ import annotations

import numpy as np

from . import _cpp_ext as _C


class NoGrad:
    """no_grad 上下文管理器，对应 Python 版的 no_grad。"""

    def __enter__(self):
        self._prev = _C.set_grad_enabled(False)
        return self

    def __exit__(self, *args):
        _C.set_grad_enabled(self._prev)


class CppTensor:
    """C++ 后端的 Tensor 包装类。

    内部持有 _C.TensorImpl，算子委托给 _C.autograd_*。
    接口与 minitorch.Tensor 对齐，可作为 drop-in 替换。
    """

    __slots__ = ("_impl",)

    # ── 构造 ──────────────────────────────────────

    def __init__(self, data, requires_grad: bool = False):
        if isinstance(data, CppTensor):
            self._impl = data._impl
        elif isinstance(data, _C.TensorImpl):
            self._impl = data
        elif isinstance(data, np.ndarray):
            self._impl = _C.TensorImpl.from_numpy(data)
            if requires_grad:
                self._impl.requires_grad = True
        elif isinstance(data, (list, tuple)):
            arr = np.asarray(data, dtype=np.float64)
            self._impl = _C.TensorImpl.from_numpy(arr)
            if requires_grad:
                self._impl.requires_grad = True
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

    @classmethod
    def from_numpy(cls, arr: np.ndarray, requires_grad: bool = False) -> CppTensor:
        t = cls.__new__(cls)
        t._impl = _C.TensorImpl.from_numpy(arr)
        if requires_grad:
            t._impl.requires_grad = True
        return t

    @classmethod
    def zeros(cls, *shape, requires_grad: bool = False) -> CppTensor:
        s = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        t = cls.__new__(cls)
        t._impl = _C.TensorImpl.zeros(list(s))
        if requires_grad:
            t._impl.requires_grad = True
        return t

    @classmethod
    def ones(cls, *shape, requires_grad: bool = False) -> CppTensor:
        s = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        t = cls.__new__(cls)
        t._impl = _C.TensorImpl.ones(list(s))
        if requires_grad:
            t._impl.requires_grad = True
        return t

    @classmethod
    def arange(cls, n: int, requires_grad: bool = False) -> CppTensor:
        return cls.from_numpy(np.arange(n, dtype=np.float64), requires_grad)

    # ── 属性 ──────────────────────────────────────

    @property
    def shape(self) -> tuple:
        return tuple(self._impl.shape)

    @property
    def ndim(self) -> int:
        return self._impl.ndim

    @property
    def numel(self) -> int:
        return self._impl.numel

    @property
    def dtype(self):
        return np.float64

    @property
    def requires_grad(self) -> bool:
        return self._impl.requires_grad

    @requires_grad.setter
    def requires_grad(self, v: bool):
        self._impl.requires_grad = v

    @property
    def grad(self) -> CppTensor | None:
        g = self._impl.grad
        if g is None:
            return None
        t = CppTensor.__new__(CppTensor)
        t._impl = g
        return t

    @property
    def grad_fn(self):
        return self._impl.grad_fn

    @property
    def is_leaf(self) -> bool:
        return self._impl.is_leaf

    # ── 算子 ──────────────────────────────────────

    def add(self, other: CppTensor) -> CppTensor:
        return self._wrap(_C.autograd_add(self._impl, other._impl))

    def sub(self, other: CppTensor) -> CppTensor:
        return self._wrap(_C.autograd_sub(self._impl, other._impl))

    def mul(self, other: CppTensor) -> CppTensor:
        return self._wrap(_C.autograd_mul(self._impl, other._impl))

    def div(self, other: CppTensor) -> CppTensor:
        return self._wrap(_C.autograd_div(self._impl, other._impl))

    def neg(self) -> CppTensor:
        return self._wrap(_C.autograd_neg(self._impl))

    def matmul(self, other: CppTensor) -> CppTensor:
        return self._wrap(_C.autograd_matmul(self._impl, other._impl))

    def sum(self, dim: int = -1, keepdim: bool = False) -> CppTensor:
        return self._wrap(_C.autograd_sum(self._impl, dim, keepdim))

    def mean(self, dim: int = -1, keepdim: bool = False) -> CppTensor:
        return self._wrap(_C.autograd_mean(self._impl, dim, keepdim))

    def relu(self) -> CppTensor:
        return self._wrap(_C.autograd_relu(self._impl))

    def transpose(self, dim0: int = 1, dim1: int = 0) -> CppTensor:
        return self._wrap(_C.autograd_transpose(self._impl, dim0, dim1))

    def exp(self) -> CppTensor:
        return self._wrap(_C.autograd_exp(self._impl))

    def log(self) -> CppTensor:
        return self._wrap(_C.autograd_log(self._impl))

    def sqrt(self) -> CppTensor:
        return self._wrap(_C.autograd_sqrt(self._impl))

    def abs(self) -> CppTensor:
        return self._wrap(_C.autograd_abs_val(self._impl))

    def pow(self, exponent: float) -> CppTensor:
        return self._wrap(_C.autograd_pow_scalar(self._impl, exponent))

    def clamp(self, min_val: float, max_val: float) -> CppTensor:
        return self._wrap(_C.autograd_clamp(self._impl, min_val, max_val))

    def sigmoid(self) -> CppTensor:
        return self._wrap(_C.autograd_sigmoid(self._impl))

    def tanh(self) -> CppTensor:
        return self._wrap(_C.autograd_tanh(self._impl))

    def log_softmax(self, dim: int = -1) -> CppTensor:
        return self._wrap(_C.autograd_log_softmax(self._impl, dim))

    def softmax(self, dim: int = -1) -> CppTensor:
        return self._wrap(_C.autograd_softmax(self._impl, dim))

    def nll_loss(self, target: CppTensor) -> CppTensor:
        return self._wrap(_C.autograd_nll_loss(self._impl, target._impl))

    def cross_entropy(self, target: CppTensor, dim: int = -1) -> CppTensor:
        return self._wrap(_C.autograd_cross_entropy(self._impl, target._impl, dim))

    def mse_loss(self, target: CppTensor) -> CppTensor:
        return self._wrap(_C.autograd_mse_loss(self._impl, target._impl))

    def gt(self, other: CppTensor) -> CppTensor:
        return self._wrap(_C.gt(self._impl, other._impl))

    def lt(self, other: CppTensor) -> CppTensor:
        return self._wrap(_C.lt(self._impl, other._impl))

    def eq(self, other: CppTensor) -> CppTensor:
        return self._wrap(_C.eq(self._impl, other._impl))

    def ge(self, other: CppTensor) -> CppTensor:
        return self._wrap(_C.ge(self._impl, other._impl))

    def le(self, other: CppTensor) -> CppTensor:
        return self._wrap(_C.le(self._impl, other._impl))

    def max(self, dim: int = -1, keepdim: bool = False) -> CppTensor:
        return self._wrap(_C.autograd_max(self._impl, dim, keepdim))

    def min(self, dim: int = -1, keepdim: bool = False) -> CppTensor:
        return self._wrap(_C.autograd_min(self._impl, dim, keepdim))

    def argmax(self, dim: int = -1, keepdim: bool = False) -> CppTensor:
        return self._wrap(_C.argmax(self._impl, dim, keepdim))

    @property
    def T(self) -> CppTensor:
        return self.transpose()

    # ── 运算符 ────────────────────────────────────

    def __add__(self, other):
        return self.add(self._coerce(other))

    def __sub__(self, other):
        return self.sub(self._coerce(other))

    def __mul__(self, other):
        return self.mul(self._coerce(other))

    def __truediv__(self, other):
        return self.div(self._coerce(other))

    def __matmul__(self, other):
        return self.matmul(other)

    def __neg__(self):
        return self.neg()

    def __pow__(self, exponent):
        return self.pow(exponent)

    def __abs__(self):
        return self.abs()

    def __repr__(self) -> str:
        return f"CppTensor({self.numpy()})"

    # ── 形状操作 ──────────────────────────────────

    def reshape(self, *shape) -> CppTensor:
        s = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        return self._wrap(self._impl.reshape(list(s)))

    def view(self, *shape) -> CppTensor:
        s = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        return self._wrap(self._impl.view(list(s)))

    def contiguous(self) -> CppTensor:
        return self._wrap(self._impl.contiguous())

    def clone(self) -> CppTensor:
        return self._wrap(self._impl.clone())

    def expand(self, *shape) -> CppTensor:
        s = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        return self._wrap(self._impl.expand(list(s)))

    def fill_(self, value: float):
        self._impl.fill_(value)

    def zero_(self):
        self._impl.zero_()

    def is_contiguous(self) -> bool:
        return self._impl.is_contiguous()

    def permute(self, *dims) -> CppTensor:
        d = dims[0] if len(dims) == 1 and isinstance(dims[0], (list, tuple)) else dims
        return self._wrap(self._impl.permute(list(d)))

    def unsqueeze(self, dim: int) -> CppTensor:
        s = list(self.shape)
        s.insert(dim, 1)
        return self.reshape(*s)

    def squeeze(self, dim: int = -1) -> CppTensor:
        s = list(self.shape)
        if dim < 0:
            dim += len(s)
        if s[dim] == 1:
            s.pop(dim)
        return self.reshape(*s)

    # ── Autograd ─────────────────────────────────

    def backward(self, gradient=None, retain_graph: bool = False, retain_grad: bool = False):
        g = gradient._impl if gradient is not None else None
        self._impl.backward(g, retain_graph, retain_grad)

    def zero_grad(self):
        self._impl.set_grad(None)

    # ── 数据访问 ──────────────────────────────────

    def numpy(self) -> np.ndarray:
        return self._impl.numpy()

    def to_vector(self) -> list:
        return self._impl.to_vector()

    def item(self) -> float:
        return self._impl.item()

    def tolist(self) -> list:
        return self.numpy().tolist()

    def allclose(self, other: CppTensor, tol: float = 1e-6) -> bool:
        return np.allclose(self.numpy(), other.numpy(), atol=tol)

    # ── 内部辅助 ──────────────────────────────────

    @staticmethod
    def _wrap(impl) -> CppTensor:
        t = CppTensor.__new__(CppTensor)
        t._impl = impl
        return t

    def _coerce(self, other) -> CppTensor:
        if isinstance(other, CppTensor):
            return other
        if isinstance(other, (int, float)):
            return CppTensor(np.array(other, dtype=np.float64))
        return CppTensor(other)
