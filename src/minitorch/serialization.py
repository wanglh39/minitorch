"""serialization：模型持久化（Ch11）。

state_dict 递归收集 Parameter + buffer。
save/load 用 pickle 序列化（教学用；真实 PyTorch 用 zip 格式 + 版本兼容）。
对应真实 PyTorch 的 serialization.py。
"""

from __future__ import annotations

import pickle

from .tensor import Tensor


def save(obj, path: str) -> None:
    """序列化 state_dict 或任意 pickle 可序列化对象到文件。"""
    serializable = _to_serializable(obj)
    with open(path, "wb") as f:
        pickle.dump(serializable, f)


def load(path: str):
    """从文件反序列化。"""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return _from_serializable(obj)


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, Tensor):
        return {"__tensor__": True, "data": obj.numpy(), "requires_grad": obj.requires_grad}
    return obj


def _from_serializable(obj):
    if isinstance(obj, dict):
        if obj.get("__tensor__"):
            t = Tensor.from_numpy(obj["data"])
            t.requires_grad = obj["requires_grad"]
            return t
        return {k: _from_serializable(v) for k, v in obj.items()}
    return obj


def save_state_dict(model, path: str) -> None:
    save(model.state_dict(), path)


def load_state_dict(model, path: str) -> None:
    state = load(path)
    model.load_state_dict(state)
