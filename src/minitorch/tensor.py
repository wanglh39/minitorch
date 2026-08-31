"""Tensor：张量前端对象（Ch1）。

持有 shape/stride/storage_offset + Storage 引用。
view 系列操作只改元信息不拷贝；reshape 在非连续时 materialize。
算子为纯计算（无 autograd），自动微分在 Ch2/Ch3 引入。
对应真实 PyTorch 的 at::Tensor / c10::TensorImpl。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .storage import Storage


def _prod(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _compute_contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _infer_shape(shape: Sequence[int], numel: int) -> tuple[int, ...]:
    shape = list(shape)
    if -1 in shape:
        if shape.count(-1) != 1:
            raise ValueError("view/reshape: 只能有一个 -1")
        known = _prod([d for d in shape if d != -1])
        shape[shape.index(-1)] = numel // known
    return tuple(shape)


def _broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    if not shapes:
        return ()
    ndim = max(len(s) for s in shapes)
    aligned = [(1,) * (ndim - len(s)) + s for s in shapes]
    result = []
    for dims in zip(*aligned, strict=True):
        non_one = [d for d in dims if d != 1]
        if not non_one:
            result.append(1)
        elif len(set(non_one)) == 1:
            result.append(non_one[0])
        else:
            raise ValueError(f"shape 无法广播: {shapes}")
    return tuple(result)


class Tensor:
    """张量：Storage 的视图，持有 shape/strides/storage_offset。"""

    def __init__(
        self,
        storage: Storage,
        shape: Sequence[int],
        strides: Sequence[int],
        storage_offset: int = 0,
        requires_grad: bool = False,
    ):
        self._storage = storage
        self._shape = tuple(shape)
        self._strides = tuple(strides)
        self._storage_offset = storage_offset
        self.requires_grad = requires_grad
        self.grad: Tensor | None = None
        self.grad_fn = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def storage_offset(self) -> int:
        return self._storage_offset

    @property
    def storage(self) -> Storage:
        return self._storage

    @property
    def dtype(self) -> np.dtype:
        return self._storage.dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return _prod(self._shape)

    @property
    def T(self) -> Tensor:
        if self.ndim < 2:
            return self
        return self.transpose()

    def numel(self) -> int:
        return self.size

    def dim(self) -> int:
        return self.ndim

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Tensor:
        arr = np.asarray(arr)
        storage = Storage.from_numpy(arr)
        return cls(storage, arr.shape, _compute_contiguous_strides(arr.shape))

    @classmethod
    def zeros(cls, *shape: int, dtype=np.float64) -> Tensor:
        return cls.from_numpy(np.zeros(shape, dtype=dtype))

    @classmethod
    def ones(cls, *shape: int, dtype=np.float64) -> Tensor:
        return cls.from_numpy(np.ones(shape, dtype=dtype))

    @classmethod
    def arange(cls, stop: float, start: float = 0, step: float = 1, dtype=np.float64) -> Tensor:
        return cls.from_numpy(np.arange(start, stop, step, dtype=dtype))

    @classmethod
    def randn(cls, *shape: int, dtype=np.float64) -> Tensor:
        return cls.from_numpy(np.random.randn(*shape).astype(dtype))

    @classmethod
    def rand(cls, *shape: int, dtype=np.float64) -> Tensor:
        return cls.from_numpy(np.random.rand(*shape).astype(dtype))

    def _numpy_view(self) -> np.ndarray:
        base = self._storage.data
        byte_strides = tuple(s * base.itemsize for s in self._strides)
        return np.lib.stride_tricks.as_strided(
            base[self._storage_offset:],
            shape=self._shape if self._shape else (),
            strides=byte_strides if self._shape else (),
        )

    def numpy(self) -> np.ndarray:
        return self._numpy_view().copy()

    def is_contiguous(self) -> bool:
        if self.ndim <= 1:
            return True
        return self._strides == _compute_contiguous_strides(self._shape)

    def contiguous(self) -> Tensor:
        if self.is_contiguous():
            return self
        arr = self._numpy_view().copy()
        storage = Storage.from_numpy(arr)
        return Tensor(storage, arr.shape, _compute_contiguous_strides(arr.shape), 0, self.requires_grad)

    def view(self, *shape: int) -> Tensor:
        shape = _infer_shape(shape, self.size)
        if not self.is_contiguous():
            raise RuntimeError("view 要求 contiguous，请用 .reshape() 或先 .contiguous()")
        if _prod(shape) != self.size:
            raise RuntimeError(f"view: 元素数不匹配 {shape} vs {self.size}")
        return Tensor(
            self._storage, shape, _compute_contiguous_strides(shape), self._storage_offset, self.requires_grad
        )

    def reshape(self, *shape: int) -> Tensor:
        shape = _infer_shape(shape, self.size)
        if _prod(shape) != self.size:
            raise RuntimeError("reshape: 元素数不匹配")
        if self.is_contiguous():
            return self.view(*shape)
        return self.contiguous().view(*shape)

    def _transpose(self, dim0: int = 1, dim1: int = 0) -> Tensor:
        if dim0 < 0:
            dim0 += self.ndim
        if dim1 < 0:
            dim1 += self.ndim
        shape = list(self._shape)
        strides = list(self._strides)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
        return Tensor(self._storage, shape, strides, self._storage_offset, self.requires_grad)

    def transpose(self, dim0: int = 1, dim1: int = 0) -> Tensor:
        from .ops.arithmetic import Transpose

        return Transpose.apply(self, dim0=dim0, dim1=dim1)

    def permute(self, *dims: int) -> Tensor:
        dims = tuple(d + self.ndim if d < 0 else d for d in dims)
        if sorted(dims) != list(range(self.ndim)):
            raise RuntimeError("permute: dims 必须是 0..ndim-1 的排列")
        shape = tuple(self._shape[d] for d in dims)
        strides = tuple(self._strides[d] for d in dims)
        return Tensor(self._storage, shape, strides, self._storage_offset, self.requires_grad)

    def unsqueeze(self, dim: int) -> Tensor:
        if dim < 0:
            dim += self.ndim + 1
        shape = self._shape[:dim] + (1,) + self._shape[dim:]
        strides = self._strides[:dim] + (0,) + self._strides[dim:]
        return Tensor(self._storage, shape, strides, self._storage_offset, self.requires_grad)

    def squeeze(self, dim: int | None = None) -> Tensor:
        if dim is not None:
            if dim < 0:
                dim += self.ndim
            if self._shape[dim] != 1:
                return self
            shape = self._shape[:dim] + self._shape[dim + 1 :]
            strides = self._strides[:dim] + self._strides[dim + 1 :]
            return Tensor(self._storage, shape, strides, self._storage_offset, self.requires_grad)
        pairs = [(s, st) for s, st in zip(self._shape, self._strides, strict=True) if s != 1]
        if not pairs:
            return Tensor(self._storage, (), (), self._storage_offset, self.requires_grad)
        shape, strides = (tuple(p[0] for p in pairs), tuple(p[1] for p in pairs))
        return Tensor(self._storage, shape, strides, self._storage_offset, self.requires_grad)

    def broadcast_to(self, shape: Sequence[int]) -> Tensor:
        shape = tuple(shape)
        if len(shape) < self.ndim:
            raise RuntimeError("broadcast_to: 目标维度不能少于自身")
        pad = len(shape) - self.ndim
        self_shape = (1,) * pad + self._shape
        self_strides = (0,) * pad + self._strides
        new_strides = []
        for i, (s, t) in enumerate(zip(self_shape, shape, strict=True)):
            if s == t:
                new_strides.append(self_strides[i])
            elif s == 1:
                new_strides.append(0)
            else:
                raise RuntimeError(f"broadcast_to: {self._shape} 不能广播到 {shape}")
        return Tensor(self._storage, shape, tuple(new_strides), self._storage_offset, self.requires_grad)

    @staticmethod
    def broadcast_tensors(*tensors: Tensor) -> tuple[Tensor, ...]:
        out_shape = _broadcast_shapes(*[t.shape for t in tensors])
        return tuple(t.broadcast_to(out_shape) for t in tensors)

    def __getitem__(self, key) -> Tensor:
        arr = self._numpy_view()[key]
        return Tensor.from_numpy(np.asarray(arr))

    def __setitem__(self, key, value):
        arr = self._numpy_view()
        if isinstance(value, Tensor):
            value = value._numpy_view()
        arr[key] = value

    def _ensure_tensor(self, other) -> Tensor:
        if isinstance(other, Tensor):
            return other
        return Tensor.from_numpy(np.asarray(other))

    def _binary(self, other, op) -> Tensor:
        other = self._ensure_tensor(other)
        a, b = Tensor.broadcast_tensors(self, other)
        return Tensor.from_numpy(op(a._numpy_view(), b._numpy_view()))

    def _add(self, other) -> Tensor:
        return self._binary(other, lambda a, b: a + b)

    def _sub(self, other) -> Tensor:
        return self._binary(other, lambda a, b: a - b)

    def _mul(self, other) -> Tensor:
        return self._binary(other, lambda a, b: a * b)

    def _div(self, other) -> Tensor:
        return self._binary(other, lambda a, b: a / b)

    def _pow(self, other) -> Tensor:
        return self._binary(other, lambda a, b: a**b)

    def _neg(self) -> Tensor:
        return Tensor.from_numpy(-self._numpy_view())

    def _matmul(self, other) -> Tensor:
        other = self._ensure_tensor(other)
        return Tensor.from_numpy(self._numpy_view() @ other._numpy_view())

    def _sum(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        arr = self._numpy_view().sum(axis=dim, keepdims=keepdim)
        return Tensor.from_numpy(np.asarray(arr))

    def _mean(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        arr = self._numpy_view().mean(axis=dim, keepdims=keepdim)
        return Tensor.from_numpy(np.asarray(arr))

    def max(self, dim: int | None = None):
        arr = self._numpy_view()
        if dim is None:
            return arr.max()
        return Tensor.from_numpy(arr.max(axis=dim))

    def add(self, other) -> Tensor:
        from .ops.arithmetic import Add

        return Add.apply(self, self._ensure_tensor(other))

    def sub(self, other) -> Tensor:
        from .ops.arithmetic import Sub

        return Sub.apply(self, self._ensure_tensor(other))

    def mul(self, other) -> Tensor:
        from .ops.arithmetic import Mul

        return Mul.apply(self, self._ensure_tensor(other))

    def div(self, other) -> Tensor:
        from .ops.arithmetic import Div

        return Div.apply(self, self._ensure_tensor(other))

    def pow(self, other) -> Tensor:
        from .ops.arithmetic import Pow

        return Pow.apply(self, other)

    def neg(self) -> Tensor:
        from .ops.arithmetic import Neg

        return Neg.apply(self)

    def matmul(self, other) -> Tensor:
        from .ops.arithmetic import Matmul

        return Matmul.apply(self, self._ensure_tensor(other))

    def sum(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        from .ops.arithmetic import Sum

        return Sum.apply(self, dim=dim, keepdim=keepdim)

    def mean(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        from .ops.arithmetic import Mean

        return Mean.apply(self, dim=dim, keepdim=keepdim)

    def __add__(self, other): return self.add(other)
    def __radd__(self, other): return self._ensure_tensor(other).add(self)
    def __sub__(self, other): return self.sub(other)
    def __rsub__(self, other): return self._ensure_tensor(other).sub(self)
    def __mul__(self, other): return self.mul(other)
    def __rmul__(self, other): return self._ensure_tensor(other).mul(self)
    def __truediv__(self, other): return self.div(other)
    def __rtruediv__(self, other): return self._ensure_tensor(other).div(self)
    def __pow__(self, other): return self.pow(other)
    def __matmul__(self, other): return self.matmul(other)
    def __neg__(self): return self.neg()

    def item(self):
        return self._numpy_view().item()

    def tolist(self):
        return self._numpy_view().tolist()

    def backward(
        self,
        gradient: Tensor | None = None,
        retain_graph: bool = False,
        retain_grad: bool = False,
    ) -> None:
        from .autograd.variable import backward

        backward(self, gradient, retain_graph=retain_graph, retain_grad=retain_grad)

    def allclose(self, other, atol: float = 1e-8) -> bool:
        other = self._ensure_tensor(other)
        return bool(np.allclose(self._numpy_view(), other._numpy_view(), atol=atol))

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 0

    def __repr__(self) -> str:
        return f"Tensor({self._numpy_view().tolist()}, shape={self._shape})"
