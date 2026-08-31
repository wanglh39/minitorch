"""Sampler：采样器（Ch10）。

Sampler 产出索引序列，DataLoader 据此从 Dataset 取样本。
  - SequentialSampler: 0, 1, 2, ..., n-1
  - RandomSampler: 随机排列（可 replacement / num_samples）
  - BatchSampler: 把索引列表切成 batch

对应真实 PyTorch 的 utils/data/sampler.py。
"""

from __future__ import annotations

import numpy as np


class Sampler:
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self, data_source, replacement: bool = False, num_samples: int | None = None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples if num_samples is not None else len(data_source)

    @property
    def num_samples(self):
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            yield from (int(x) for x in np.random.randint(0, n, size=self._num_samples))
        else:
            yield from (int(x) for x in np.random.permutation(n)[: self._num_samples])

    def __len__(self):
        return self._num_samples


class BatchSampler(Sampler):
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size
