"""pytest 公共 fixture 与工具。

numerical_grad: 有限差分数值梯度，用于对照自动微分结果（Ch3 起使用）。
"""

import numpy as np
import pytest


@pytest.fixture
def numerical_grad():
    def _grad(f, x, eps=1e-6):
        x = np.asarray(x, dtype=np.float64)
        g = np.zeros_like(x)
        it = np.nditer(x, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = x[idx]
            x[idx] = orig + eps
            fp = float(np.asarray(f(x.copy())).sum())
            x[idx] = orig - eps
            fm = float(np.asarray(f(x.copy())).sum())
            x[idx] = orig
            g[idx] = (fp - fm) / (2 * eps)
        return g

    return _grad