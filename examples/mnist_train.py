"""MNIST 端到端训练示例（Ch7）。

用合成数据演示完整训练循环（前向 → loss → 反向 → 优化器 step）。
真实 MNIST 数据加载在 Ch10 DataLoader 实现后替换。

运行：
    python examples/mnist_train.py
"""

import numpy as np

from minitorch import Tensor
from minitorch.nn import CrossEntropyLoss, Linear, Module
from minitorch.nn import functional as F
from minitorch.optim import Adam, StepLR


class MLP(Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden)
        self.fc2 = Linear(hidden, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def generate_synthetic_data(n: int = 500, dim: int = 784, num_classes: int = 10):
    np.random.seed(42)
    X = np.random.randn(n, dim) * 0.5
    centers = np.random.randn(num_classes, dim)
    labels = np.random.randint(0, num_classes, size=n)
    X += centers[labels]
    return X.astype(np.float64), labels.astype(np.int64)


def train():
    X, y = generate_synthetic_data(n=500, dim=784, num_classes=10)
    n = X.shape[0]
    batch_size = 32
    num_batches = n // batch_size

    model = MLP(784, 128, 10)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = CrossEntropyLoss()

    num_epochs = 20
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = Tensor.from_numpy(X[start:end])
            y_batch = Tensor.from_numpy(y[start:end])

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.numpy().argmax(axis=1)
            correct += (preds == y[start:end]).sum()

        scheduler.step()
        avg_loss = total_loss / num_batches
        acc = correct / n
        print(f"Epoch {epoch + 1:3d}/{num_epochs}  loss={avg_loss:.4f}  acc={acc:.2%}  lr={optimizer.param_groups[0]['lr']:.6f}")


if __name__ == "__main__":
    train()
