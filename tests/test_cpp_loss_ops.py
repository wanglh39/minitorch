"""test_cpp_loss_ops：C++ 损失函数测试（Ch9 第二批）。

验证 log_softmax/softmax/nll_loss/cross_entropy/mse_loss 的前向和 backward。
对照 numpy 手算结果。
"""

import numpy as np

from minitorch import _cpp_ext as _C


def _close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


# ── log_softmax ──────────────────────────────────────

def test_log_softmax_forward():
    x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    a = _C.TensorImpl.from_numpy(x)
    r = _C.log_softmax(a, -1)
    # numpy reference
    x_max = np.max(x, axis=-1, keepdims=True)
    expected = (x - x_max) - np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True))
    assert _close(r.numpy(), expected)


def test_log_softmax_backward():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    a = _C.TensorImpl.from_numpy(x)
    a.requires_grad = True
    r = _C.autograd_log_softmax(a, -1)
    grad = _C.TensorImpl.from_numpy(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    r.backward(grad)
    # grad_x = grad - softmax * sum(grad, dim=-1, keepdim=True)
    sm = np.exp(r.numpy())
    g = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    expected = g - sm * np.sum(g, axis=-1, keepdims=True)
    assert _close(a.grad.numpy(), expected)


def test_log_softmax_backward_with_grad():
    x = np.array([[1.0, 2.0, 3.0]])
    a = _C.TensorImpl.from_numpy(x)
    a.requires_grad = True
    r = _C.autograd_log_softmax(a, -1)
    grad = _C.TensorImpl.from_numpy(np.array([[0.1, 0.2, 0.3]]))
    r.backward(grad)
    # Reference: softmax = exp(log_softmax)
    sm = np.exp(r.numpy())
    expected = np.array([[0.1, 0.2, 0.3]]) - sm * np.sum([[0.1, 0.2, 0.3]], axis=-1, keepdims=True)
    assert _close(a.grad.numpy(), expected)


# ── softmax ─────────────────────────────────────────

def test_softmax_forward():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    a = _C.TensorImpl.from_numpy(x)
    r = _C.softmax(a, -1)
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    expected = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    assert _close(r.numpy(), expected)


def test_softmax_sums_to_one():
    a = _C.TensorImpl([1.0, 2.0, 3.0, 4.0], [4])
    r = _C.softmax(a, -1)
    assert _close(sum(r.to_vector()), 1.0)


def test_softmax_backward():
    x = np.array([[1.0, 2.0, 3.0]])
    a = _C.TensorImpl.from_numpy(x)
    a.requires_grad = True
    r = _C.autograd_softmax(a, -1)
    grad = _C.TensorImpl.from_numpy(np.array([[0.1, 0.2, 0.3]]))
    r.backward(grad)
    sm = r.numpy()
    g = np.array([[0.1, 0.2, 0.3]])
    expected = sm * (g - np.sum(g * sm, axis=-1, keepdims=True))
    assert _close(a.grad.numpy(), expected)


# ── nll_loss ────────────────────────────────────────

def test_nll_loss_forward():
    log_probs = np.array([[-0.5, -1.5, -2.0], [-2.0, -0.3, -1.0]])
    target = np.array([0, 1])
    lp = _C.TensorImpl.from_numpy(log_probs)
    t = _C.TensorImpl.from_numpy(target)
    loss = _C.nll_loss(lp, t)
    expected = -np.mean(log_probs[np.arange(2), target])
    assert _close(loss.item(), expected)


def test_nll_loss_backward():
    log_probs = np.array([[-0.5, -1.5, -2.0], [-2.0, -0.3, -1.0]])
    target = np.array([0, 1])
    lp = _C.TensorImpl.from_numpy(log_probs)
    lp.requires_grad = True
    t = _C.TensorImpl.from_numpy(target)
    loss = _C.autograd_nll_loss(lp, t)
    loss.backward(_C.TensorImpl([1.0], []))
    # grad[i, target[i]] = -1/n
    n = 2
    expected = np.zeros((2, 3))
    expected[0, 0] = -1.0 / n
    expected[1, 1] = -1.0 / n
    assert _close(lp.grad.numpy(), expected)


# ── cross_entropy ───────────────────────────────────

def test_cross_entropy_forward():
    logits = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    target = np.array([2, 0])
    lg = _C.TensorImpl.from_numpy(logits)
    t = _C.TensorImpl.from_numpy(target)
    loss = _C.autograd_cross_entropy(lg, t, -1)
    # numpy reference
    x_max = np.max(logits, axis=-1, keepdims=True)
    log_sm = (logits - x_max) - np.log(np.sum(np.exp(logits - x_max), axis=-1, keepdims=True))
    expected = -np.mean(log_sm[np.arange(2), target])
    assert _close(loss.item(), expected)


def test_cross_entropy_backward():
    logits = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    target = np.array([2, 0])
    lg = _C.TensorImpl.from_numpy(logits)
    lg.requires_grad = True
    t = _C.TensorImpl.from_numpy(target)
    loss = _C.autograd_cross_entropy(lg, t, -1)
    loss.backward(_C.TensorImpl([1.0], []))
    # grad = (softmax - one_hot) / n
    x_max = np.max(logits, axis=-1, keepdims=True)
    exp_x = np.exp(logits - x_max)
    sm = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    one_hot = np.zeros((2, 3))
    one_hot[0, 2] = 1
    one_hot[1, 0] = 1
    expected = (sm - one_hot) / 2
    assert _close(lg.grad.numpy(), expected)


# ── mse_loss ────────────────────────────────────────

def test_mse_loss_forward():
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.5, 3.5])
    p = _C.TensorImpl.from_numpy(pred)
    t = _C.TensorImpl.from_numpy(target)
    loss = _C.autograd_mse_loss(p, t)
    expected = np.mean((pred - target) ** 2)
    assert _close(loss.item(), expected)


def test_mse_loss_backward():
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.5, 3.5])
    p = _C.TensorImpl.from_numpy(pred)
    p.requires_grad = True
    t = _C.TensorImpl.from_numpy(target)
    loss = _C.autograd_mse_loss(p, t)
    loss.backward(_C.TensorImpl([1.0], []))
    # d/dpred mean((pred - target)^2) = 2*(pred - target)/n
    n = 3
    expected = 2 * (pred - target) / n
    assert _close(p.grad.numpy(), expected)