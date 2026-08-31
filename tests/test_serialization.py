"""Ch11 持久化测试。"""

import os

import numpy as np

from minitorch import Tensor
from minitorch.nn import Linear, Sequential
from minitorch.serialization import load, load_state_dict, save, save_state_dict


def test_save_load_roundtrip():
    model = Sequential(Linear(4, 3), Linear(3, 2))
    sd = model.state_dict()
    path = "_test_save_load.pkl"
    try:
        save(sd, path)
        loaded = load(path)
        for key in sd:
            assert np.allclose(sd[key].numpy(), loaded[key].numpy())
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_save_load_state_dict_into_model():
    model1 = Sequential(Linear(4, 3), Linear(3, 2))
    model2 = Sequential(Linear(4, 3), Linear(3, 2))
    path = "_test_state_dict.pkl"
    try:
        save_state_dict(model1, path)
        load_state_dict(model2, path)
        sd1 = model1.state_dict()
        sd2 = model2.state_dict()
        for key in sd1:
            assert np.allclose(sd1[key].numpy(), sd2[key].numpy())
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_save_load_tensor():
    t = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    path = "_test_tensor.pkl"
    try:
        save(t, path)
        loaded = load(path)
        assert np.allclose(loaded.numpy(), [1.0, 2.0, 3.0])
    finally:
        if os.path.exists(path):
            os.remove(path)
