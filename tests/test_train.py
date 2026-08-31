"""Ch7 端到端训练测试。"""

import numpy as np

from minitorch import Tensor
from minitorch.nn import Linear, Module, MSELoss, Sequential
from minitorch.nn import functional as F
from minitorch.optim import SGD, Adam


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def test_regression_loss_decreases():
    np.random.seed(42)
    X = np.random.randn(32, 4)
    W_true = np.random.randn(4, 1)
    Y = X @ W_true + 0.1 * np.random.randn(32, 1)

    model = MLP()
    opt = SGD(model.parameters(), lr=0.01)
    crit = MSELoss()

    losses = []
    for _ in range(200):
        pred = model(Tensor.from_numpy(X))
        loss = crit(pred, Tensor.from_numpy(Y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5


def test_overfit_small_batch():
    np.random.seed(0)
    X = np.random.randn(4, 4)
    Y = np.random.randn(4, 1)

    model = MLP()
    opt = Adam(model.parameters(), lr=0.01)
    crit = MSELoss()

    initial_loss = None
    for _ in range(500):
        pred = model(Tensor.from_numpy(X))
        loss = crit(pred, Tensor.from_numpy(Y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if initial_loss is None:
            initial_loss = loss.item()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.01


def test_classification_loss_decreases():
    np.random.seed(42)
    n = 40
    X = np.random.randn(n, 4)
    labels = (X[:, 0] + X[:, 1] > 0).astype(int)
    Y = np.zeros((n, 2))
    Y[np.arange(n), labels] = 1

    model = Sequential(Linear(4, 8), Linear(8, 2))
    opt = Adam(model.parameters(), lr=0.01)
    from minitorch.nn import CrossEntropyLoss

    crit = CrossEntropyLoss()

    losses = []
    for _ in range(200):
        logits = model(Tensor.from_numpy(X))
        loss = crit(logits, Tensor.from_numpy(labels))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
