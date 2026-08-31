"""amp：自动混合精度包（Ch11）。

子模块：
  - autocast:    前向自动转 fp16 的上下文。
  - grad_scaler: 梯度缩放，防小梯度下溢。
"""

from .autocast import Autocast, autocast_tensor, get_autocast_dtype, is_autocast_enabled
from .grad_scaler import GradScaler

__all__ = [
    "Autocast",
    "GradScaler",
    "autocast_tensor",
    "get_autocast_dtype",
    "is_autocast_enabled",
]
