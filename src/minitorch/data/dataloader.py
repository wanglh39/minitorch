"""DataLoader：数据加载器（Ch10）。

组合 Dataset + Sampler + collate_fn。
  - batch_size: 每批样本数
  - shuffle: 是否随机打乱
  - drop_last: 丢弃不完整尾批
  - collate_fn: 把 list of samples 拼成 batch tensor

教学版 num_workers=0 单进程。真实 PyTorch 多进程预取绕 GIL。
对应真实 PyTorch 的 utils/data/dataloader.py。
"""

from __future__ import annotations

import numpy as np

from .dataset import Dataset
from .sampler import BatchSampler, RandomSampler, SequentialSampler


def default_collate(batch: list) -> list:
    """把 list of samples 拼成 batch。

    sample 是 tuple/list → 转置后逐元素 stack。
    sample 是 ndarray → stack 成 batch array。
    """
    if isinstance(batch[0], tuple | list):
        transposed = zip(*batch, strict=True)
        return [default_collate(list(col)) for col in transposed]
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    return np.array(batch)


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn or default_collate

        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self):
        for indices in self.batch_sampler:
            samples = [self.dataset[i] for i in indices]
            yield self.collate_fn(samples)

    def __len__(self):
        return len(self.batch_sampler)
