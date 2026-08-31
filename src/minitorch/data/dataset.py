"""Dataset：数据集协议（Ch10）。

设计要点：
  - Dataset: __getitem__/__len__ 协议（map-style）。
  - IterableDataset: __iter__ 协议（streaming）。
  - TensorDataset: 多个 Tensor 沿第一维对齐，__getitem__ 返回各 Tensor 同索引切片。
  - 与真实 PyTorch 的 utils/data/dataset.py 对应。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from ..tensor import Tensor


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class IterableDataset:
    def __iter__(self) -> Iterator:
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors: Tensor):
        n = tensors[0].shape[0]
        for t in tensors:
            if t.shape[0] != n:
                raise ValueError("所有 Tensor 第一维必须相同")
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]


class ArrayDataset(Dataset):
    def __init__(self, *arrays: np.ndarray):
        n = len(arrays[0])
        for a in arrays:
            if len(a) != n:
                raise ValueError("所有数组长度必须相同")
        self.arrays = arrays

    def __getitem__(self, index):
        return tuple(arr[index] for arr in self.arrays)

    def __len__(self):
        return len(self.arrays[0])


class ConcatDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = list(datasets)
        self._cum_sizes: list[int] = []
        cum = 0
        for ds in self.datasets:
            cum += len(ds)
            self._cum_sizes.append(cum)

    def __len__(self):
        return self._cum_sizes[-1] if self._cum_sizes else 0

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        for i, cum_size in enumerate(self._cum_sizes):
            if index < cum_size:
                prev = self._cum_sizes[i - 1] if i > 0 else 0
                return self.datasets[i][index - prev]
        raise IndexError(index)
