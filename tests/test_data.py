"""Ch10 数据加载测试。"""

import numpy as np

from minitorch import Tensor
from minitorch.data import (
    ArrayDataset,
    BatchSampler,
    ConcatDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

# ── Sampler ──────────────────────────────────────────


def test_sequential_sampler():
    data = list(range(10))
    sampler = SequentialSampler(data)
    assert list(sampler) == list(range(10))


def test_random_sampler_covers_all():
    np.random.seed(42)
    data = list(range(20))
    sampler = RandomSampler(data)
    indices = list(sampler)
    assert sorted(indices) == list(range(20))


def test_random_sampler_with_replacement():
    np.random.seed(42)
    data = list(range(10))
    sampler = RandomSampler(data, replacement=True, num_samples=15)
    indices = list(sampler)
    assert len(indices) == 15


def test_batch_sampler():
    sampler = SequentialSampler(list(range(10)))
    bs = BatchSampler(sampler, batch_size=3, drop_last=False)
    batches = list(bs)
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert len(bs) == 4


def test_batch_sampler_drop_last():
    sampler = SequentialSampler(list(range(10)))
    bs = BatchSampler(sampler, batch_size=3, drop_last=True)
    batches = list(bs)
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert len(bs) == 3


# ── Dataset ──────────────────────────────────────────


def test_tensor_dataset():
    x = Tensor.from_numpy(np.arange(12).reshape(4, 3))
    y = Tensor.from_numpy(np.arange(4))
    ds = TensorDataset(x, y)
    assert len(ds) == 4
    sx, sy = ds[1]
    assert np.allclose(sx.numpy(), [3, 4, 5])
    assert sy.item() == 1


def test_array_dataset():
    x = np.arange(12).reshape(4, 3)
    y = np.arange(4)
    ds = ArrayDataset(x, y)
    assert len(ds) == 4
    sx, sy = ds[2]
    assert np.allclose(sx, [6, 7, 8])
    assert sy == 2


def test_concat_dataset():
    ds1 = ArrayDataset(np.arange(3))
    ds2 = ArrayDataset(np.arange(3, 7))
    cds = ConcatDataset([ds1, ds2])
    assert len(cds) == 7
    assert cds[0][0] == 0
    assert cds[3][0] == 3
    assert cds[6][0] == 6


# ── DataLoader ───────────────────────────────────────


def test_dataloader_no_shuffle():
    x = np.arange(12).reshape(4, 3)
    ds = ArrayDataset(x)
    loader = DataLoader(ds, batch_size=2)
    batches = list(loader)
    assert len(batches) == 2
    assert np.allclose(batches[0], [[0, 1, 2], [3, 4, 5]])
    assert np.allclose(batches[1], [[6, 7, 8], [9, 10, 11]])


def test_dataloader_shuffle_covers_all():
    np.random.seed(42)
    x = np.arange(20).reshape(10, 2)
    ds = ArrayDataset(x)
    loader = DataLoader(ds, batch_size=3, shuffle=True)
    batches = list(loader)
    all_rows = np.vstack([b[0] for b in batches])
    assert sorted(all_rows[:, 0].tolist()) == list(range(0, 20, 2))


def test_dataloader_drop_last():
    x = np.arange(10).reshape(5, 2)
    ds = ArrayDataset(x)
    loader = DataLoader(ds, batch_size=2, drop_last=True)
    batches = list(loader)
    assert len(batches) == 2


def test_dataloader_len():
    x = np.arange(10).reshape(5, 2)
    ds = ArrayDataset(x)
    loader = DataLoader(ds, batch_size=2)
    assert len(loader) == 3


def test_dataloader_custom_collate():
    x = np.arange(8).reshape(4, 2)
    y = np.arange(4)
    ds = ArrayDataset(x, y)

    def my_collate(batch):
        xs = np.stack([b[0] for b in batch])
        ys = np.stack([b[1] for b in batch])
        return xs, ys

    loader = DataLoader(ds, batch_size=2, collate_fn=my_collate)
    bx, by = next(iter(loader))
    assert np.allclose(bx, [[0, 1], [2, 3]])
    assert np.allclose(by, [0, 1])
