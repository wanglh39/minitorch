"""Ch11 混合精度测试。"""

import numpy as np

from minitorch import Tensor
from minitorch.amp import Autocast, GradScaler
from minitorch.optim import SGD


def test_autocast_context():
    from minitorch.amp.autocast import is_autocast_enabled

    assert not is_autocast_enabled()
    with Autocast(enabled=True):
        assert is_autocast_enabled()
    assert not is_autocast_enabled()


def test_autocast_tensor_dtype():
    from minitorch.amp.autocast import autocast_tensor

    t = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    with Autocast(enabled=True):
        t_cast = autocast_tensor(t)
        assert t_cast.dtype == np.float16
    assert t.dtype == np.float64


def test_grad_scaler_scale():
    scaler = GradScaler(init_scale=128.0)
    loss = Tensor.from_numpy(np.array([1.0, 2.0]))
    scaled = scaler.scale(loss)
    assert np.allclose(scaled.numpy(), [128.0, 256.0])


def test_grad_scaler_skips_inf():
    p = Tensor.from_numpy(np.array([1.0]))
    p.requires_grad = True
    p.grad = Tensor.from_numpy(np.array([float("inf")]))
    opt = SGD([p], lr=0.1)
    scaler = GradScaler(init_scale=128.0)
    scaler.unscale_(opt)
    assert scaler._found_inf
    original = p.numpy().copy()
    scaler.step(opt)
    assert np.allclose(p.numpy(), original)


def test_grad_scaler_normal_step():
    p = Tensor.from_numpy(np.array([1.0]))
    p.requires_grad = True
    p.grad = Tensor.from_numpy(np.array([0.5]))
    opt = SGD([p], lr=0.1)
    scaler = GradScaler(init_scale=128.0)
    scaler.unscale_(opt)
    assert not scaler._found_inf
    scaler.step(opt)
    assert np.allclose(p.numpy(), [1.0 - 0.1 * 0.5 / 128.0])


def test_grad_scaler_update_backoff():
    scaler = GradScaler(init_scale=128.0, backoff_factor=0.5)
    scaler._found_inf = True
    scaler.update()
    assert scaler.get_scale() == 64.0


def test_grad_scaler_update_growth():
    scaler = GradScaler(init_scale=128.0, growth_factor=2.0, growth_interval=3)
    scaler._found_inf = False
    scaler.update()
    scaler.update()
    assert scaler.get_scale() == 128.0
    scaler.update()
    assert scaler.get_scale() == 256.0
