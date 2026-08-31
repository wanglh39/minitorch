"""Storage：底层一维数据缓冲区（Ch1）。

Tensor 只持有 shape/stride/storage_offset + 对 Storage 的引用；
Storage 持有实际数据 buffer（numpy array），使 view 零拷贝成为可能。
对应真实 PyTorch 的 c10::Storage / StorageImpl。
"""

from __future__ import annotations

import numpy as np


class Storage:
    """一维数据缓冲区，可被一个或多个 Tensor 共享。"""

    __slots__ = ("_data",)

    def __init__(self, data=None, size: int = 0, dtype=np.float64):
        if data is not None:
            arr = np.asarray(data, dtype=dtype).ravel()
        else:
            arr = np.zeros(size, dtype=dtype)
        self._data = arr

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def itemsize(self) -> int:
        return self._data.itemsize

    def __len__(self) -> int:
        return self._data.size

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, value):
        self._data[idx] = value

    def resize(self, size: int) -> None:
        if size <= self._data.size:
            return
        new = np.zeros(size, dtype=self._data.dtype)
        new[: self._data.size] = self._data
        self._data = new

    def fill_(self, value) -> None:
        self._data.fill(value)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Storage:
        arr = np.asarray(arr)
        return cls(data=arr, dtype=arr.dtype)

    def __repr__(self) -> str:
        return f"Storage({self._data!r})"
