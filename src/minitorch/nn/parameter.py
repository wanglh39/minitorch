"""Parameter：模块参数（Ch5）。

继承 Tensor，requires_grad=True，注册到 Module。
对应真实 PyTorch 的 nn/parameter.py。
"""

from __future__ import annotations

import numpy as np

from ..storage import Storage
from ..tensor import Tensor, _compute_contiguous_strides


class Parameter(Tensor):
    def __init__(self, data, requires_grad: bool = True):
        if isinstance(data, Tensor):
            super().__init__(
                data.storage, data.shape, data.strides, data.storage_offset, requires_grad=True
            )
        else:
            arr = np.asarray(data)
            storage = Storage.from_numpy(arr)
            super().__init__(
                storage, arr.shape, _compute_contiguous_strides(arr.shape), 0, requires_grad=True
            )
        self.requires_grad = requires_grad

    def __repr__(self) -> str:
        return f"Parameter({self._numpy_view().tolist()}, shape={self._shape})"
