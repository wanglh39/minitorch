"""Ch6 优化器系统测试。"""

import math

import numpy as np

from minitorch import Tensor
from minitorch.optim import SGD, Adam, CosineAnnealingLR, LambdaLR, StepLR

# ── SGD ──────────────────────────────────────────────


def test_sgd_basic_update():
    p = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    p.requires_grad = True
    p.grad = Tensor.from_numpy(np.array([0.1, 0.2, 0.3]))
    opt = SGD([p], lr=0.5)
    opt.step()
    assert np.allclose(p.numpy(), [0.95, 1.9, 2.85])


def test_sgd_momentum():
    p = Tensor.from_numpy(np.array([1.0]))
    p.requires_grad = True
    p.grad = Tensor.from_numpy(np.array([1.0]))
    opt = SGD([p], lr=0.1, momentum=0.9)
    opt.step()
    assert np.allclose(p.numpy(), [0.9])
    p.grad = Tensor.from_numpy(np.array([1.0]))
    opt.step()
    assert np.allclose(p.numpy(), [0.9 - 0.1 * (0.9 + 1.0)])


def test_sgd_nesterov():
    p = Tensor.from_numpy(np.array([1.0]))
    p.requires_grad = True
    p.grad = Tensor.from_numpy(np.array([1.0]))
    opt = SGD([p], lr=0.1, momentum=0.9, nesterov=True)
    opt.step()
    buf1 = 1.0
    nesterov_grad1 = 1.0 + 0.9 * buf1
    assert np.allclose(p.numpy(), [1.0 - 0.1 * nesterov_grad1])
    p.grad = Tensor.from_numpy(np.array([1.0]))
    opt.step()
    buf2 = 0.9 * buf1 + 1.0
    nesterov_grad2 = 1.0 + 0.9 * buf2
    assert np.allclose(p.numpy(), [1.0 - 0.1 * nesterov_grad1 - 0.1 * nesterov_grad2])


def test_sgd_weight_decay():
    p = Tensor.from_numpy(np.array([2.0]))
    p.requires_grad = True
    p.grad = Tensor.from_numpy(np.array([1.0]))
    opt = SGD([p], lr=0.1, weight_decay=0.5)
    opt.step()
    effective_grad = 1.0 + 0.5 * 2.0
    assert np.allclose(p.numpy(), [2.0 - 0.1 * effective_grad])


# ── Adam ─────────────────────────────────────────────


def test_adam_converges_on_quadratic():
    p = Tensor.from_numpy(np.array([5.0, -3.0]))
    p.requires_grad = True
    opt = Adam([p], lr=0.1)

    for _ in range(500):
        p.grad = Tensor.from_numpy(2.0 * p.numpy())
        opt.step()
    assert np.allclose(p.numpy(), [0.0, 0.0], atol=1e-5)


def test_adam_bias_correction_first_step():
    p = Tensor.from_numpy(np.array([0.0]))
    p.requires_grad = True
    p.grad = Tensor.from_numpy(np.array([1.0]))
    opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    opt.step()
    m_hat = 1.0
    v_hat = 1.0
    expected = 0.0 - 0.1 * m_hat / (math.sqrt(v_hat) + 1e-8)
    assert np.allclose(p.numpy(), [expected])


def test_adam_state_persists():
    p = Tensor.from_numpy(np.array([1.0]))
    p.requires_grad = True
    opt = Adam([p], lr=0.1)
    p.grad = Tensor.from_numpy(np.array([1.0]))
    opt.step()
    state = opt.state[id(p)]
    assert state["step"] == 1
    assert state["exp_avg"].shape == (1,)
    p.grad = Tensor.from_numpy(np.array([0.5]))
    opt.step()
    assert state["step"] == 2


# ── Param Groups ─────────────────────────────────────


def test_param_group_independent_lr():
    p1 = Tensor.from_numpy(np.array([1.0]))
    p1.requires_grad = True
    p2 = Tensor.from_numpy(np.array([1.0]))
    p2.requires_grad = True
    p1.grad = Tensor.from_numpy(np.array([1.0]))
    p2.grad = Tensor.from_numpy(np.array([1.0]))
    opt = SGD(
        [
            {"params": [p1], "lr": 0.1},
            {"params": [p2], "lr": 0.01},
        ],
        lr=0.05,
    )
    opt.step()
    assert np.allclose(p1.numpy(), [0.9])
    assert np.allclose(p2.numpy(), [0.99])


def test_zero_grad_clears_grad():
    p = Tensor.from_numpy(np.array([1.0]))
    p.requires_grad = True
    p.grad = Tensor.from_numpy(np.array([0.5]))
    opt = SGD([p], lr=0.1)
    opt.zero_grad()
    assert p.grad is None


# ── LR Schedulers ────────────────────────────────────


def test_lambda_lr():
    p = Tensor.from_numpy(np.array([1.0]))
    p.requires_grad = True
    opt = SGD([p], lr=1.0)
    sched = LambdaLR(opt, lr_lambda=lambda epoch: 0.95**epoch)
    assert np.isclose(opt.param_groups[0]["lr"], 1.0)
    sched.step()
    assert np.isclose(opt.param_groups[0]["lr"], 0.95)
    sched.step()
    assert np.isclose(opt.param_groups[0]["lr"], 0.95**2)


def test_step_lr():
    p = Tensor.from_numpy(np.array([1.0]))
    p.requires_grad = True
    opt = SGD([p], lr=1.0)
    sched = StepLR(opt, step_size=2, gamma=0.1)
    assert np.isclose(opt.param_groups[0]["lr"], 1.0)
    sched.step()
    assert np.isclose(opt.param_groups[0]["lr"], 1.0)
    sched.step()
    assert np.isclose(opt.param_groups[0]["lr"], 0.1)
    sched.step()
    assert np.isclose(opt.param_groups[0]["lr"], 0.1)
    sched.step()
    assert np.isclose(opt.param_groups[0]["lr"], 0.01)


def test_cosine_annealing_lr_curve():
    p = Tensor.from_numpy(np.array([1.0]))
    p.requires_grad = True
    opt = SGD([p], lr=1.0)
    T_max = 10
    sched = CosineAnnealingLR(opt, T_max=T_max, eta_min=0.0)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(T_max):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert np.isclose(lrs[0], 1.0)
    assert np.isclose(lrs[T_max], 0.0, atol=1e-10)
    for i in range(T_max):
        expected = 0.0 + (1.0 - 0.0) * (1 + math.cos(math.pi * (i + 1) / T_max)) / 2
        assert np.isclose(lrs[i + 1], expected)


def test_scheduler_with_param_groups():
    p1 = Tensor.from_numpy(np.array([1.0]))
    p1.requires_grad = True
    p2 = Tensor.from_numpy(np.array([1.0]))
    p2.requires_grad = True
    opt = SGD(
        [{"params": [p1], "lr": 1.0}, {"params": [p2], "lr": 2.0}],
        lr=1.0,
    )
    sched = StepLR(opt, step_size=1, gamma=0.5)
    sched.step()
    assert np.isclose(opt.param_groups[0]["lr"], 0.5)
    assert np.isclose(opt.param_groups[1]["lr"], 1.0)
