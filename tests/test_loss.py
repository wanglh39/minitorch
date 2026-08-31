"""Ch7 损失函数测试。"""

import numpy as np

from minitorch import Tensor
from minitorch.nn import CrossEntropyLoss, MSELoss
from minitorch.nn import functional as F

# ── MSELoss ──────────────────────────────────────────


def test_mse_loss_forward():
    pred = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    target = Tensor.from_numpy(np.array([1.0, 0.0, 4.0]))
    loss = F.mse_loss(pred, target)
    expected = np.mean((np.array([1, 2, 3]) - np.array([1, 0, 4])) ** 2)
    assert np.isclose(loss.item(), expected)


def test_mse_loss_backward():
    pred = Tensor.from_numpy(np.array([1.0, 2.0, 3.0]))
    pred.requires_grad = True
    target = Tensor.from_numpy(np.array([0.0, 0.0, 0.0]))
    loss = F.mse_loss(pred, target)
    loss.backward()
    expected = 2 * (np.array([1, 2, 3]) - np.array([0, 0, 0])) / 3
    assert np.allclose(pred.grad.numpy(), expected)


def test_mse_loss_module():
    crit = MSELoss()
    pred = Tensor.from_numpy(np.array([1.0, 2.0]))
    target = Tensor.from_numpy(np.array([0.0, 0.0]))
    loss = crit(pred, target)
    assert np.isclose(loss.item(), 2.5)


# ── ReLU ─────────────────────────────────────────────


def test_relu_forward():
    x = Tensor.from_numpy(np.array([-1.0, 0.0, 1.0, 2.0]))
    out = F.relu(x)
    assert np.allclose(out.numpy(), [0, 0, 1, 2])


def test_relu_backward():
    x = Tensor.from_numpy(np.array([-1.0, 0.5, 1.0]))
    x.requires_grad = True
    out = F.relu(x)
    out.backward(Tensor.from_numpy(np.ones(3)))
    assert np.allclose(x.grad.numpy(), [0, 1, 1])


# ── Softmax / LogSoftmax ─────────────────────────────


def test_softmax_sums_to_one():
    x = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    out = F.softmax(x, dim=-1)
    assert np.allclose(out.numpy().sum(axis=-1), [1.0, 1.0])


def test_log_softmax_numerical_stability():
    x = Tensor.from_numpy(np.array([1000.0, 1001.0, 1002.0]))
    out = F.log_softmax(x, dim=-1)
    assert np.all(np.isfinite(out.numpy()))
    manual = np.array([1000, 1001, 1002]) - 1002 - np.log(np.sum(np.exp(np.array([0, 1, 2]) - 2)))
    assert np.allclose(out.numpy(), manual)


def test_log_softmax_backward():
    x = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]]))
    x.requires_grad = True
    out = F.log_softmax(x, dim=-1)
    out.backward(Tensor.from_numpy(np.ones((1, 3))))
    assert np.allclose(x.grad.numpy().sum(), 0.0, atol=1e-10)


# ── CrossEntropy ─────────────────────────────────────


def test_cross_entropy_forward():
    logits = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]))
    target = Tensor.from_numpy(np.array([2, 0]))
    loss = F.cross_entropy(logits, target)
    x = np.array([[1, 2, 3], [1, 1, 1]])
    log_probs = x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
    manual = -np.mean([log_probs[0, 2], log_probs[1, 0]])
    assert np.isclose(loss.item(), manual)


def test_cross_entropy_numerical_stability():
    logits = Tensor.from_numpy(np.array([[1000.0, 1001.0, 1002.0]]))
    target = Tensor.from_numpy(np.array([0]))
    loss = F.cross_entropy(logits, target)
    assert np.isfinite(loss.item())


def test_cross_entropy_backward():
    logits = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]))
    logits.requires_grad = True
    target = Tensor.from_numpy(np.array([2, 0]))
    loss = F.cross_entropy(logits, target)
    loss.backward()
    sm = np.exp(np.array([[1, 2, 3], [0, 1, 2]]))
    sm = sm / sm.sum(axis=-1, keepdims=True)
    onehot = np.zeros((2, 3))
    onehot[0, 2] = 1
    onehot[1, 0] = 1
    expected = (sm - onehot) / 2
    assert np.allclose(logits.grad.numpy(), expected)


def test_cross_entropy_module():
    crit = CrossEntropyLoss()
    logits = Tensor.from_numpy(np.array([[0.0, 0.0, 0.0]]))
    target = Tensor.from_numpy(np.array([1]))
    loss = crit(logits, target)
    assert np.isclose(loss.item(), np.log(3))
