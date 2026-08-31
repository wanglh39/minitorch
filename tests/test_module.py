"""Ch5 nn.Module 体系测试：注册、parameters/state_dict/hooks/train-eval/Linear/Sequential。"""

import numpy as np

from minitorch import Tensor
from minitorch.nn import Linear, Module, ModuleList, Parameter, Sequential


class _MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def test_parameter_requires_grad():
    p = Parameter(Tensor.from_numpy(np.array([1.0, 2.0])))
    assert p.requires_grad


def test_module_setattr_register():
    m = Linear(3, 2)
    assert "weight" in m._parameters
    assert "bias" in m._parameters


def test_nested_parameters_recursive():
    m = _MLP()
    params = list(m.parameters())
    assert len(params) == 4


def test_named_parameters_prefix():
    m = _MLP()
    names = dict(m.named_parameters())
    assert "fc1.weight" in names
    assert "fc2.bias" in names


def test_state_dict_keys():
    m = _MLP()
    sd = m.state_dict()
    assert set(sd.keys()) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}


def test_state_dict_roundtrip():
    m = _MLP()
    x = Tensor.from_numpy(np.random.randn(2, 4))
    out1 = m(x)
    sd = m.state_dict()
    m2 = _MLP()
    m2.load_state_dict(sd)
    out2 = m2(x)
    assert out1.allclose(out2, atol=1e-6)


def test_linear_forward():
    lin = Linear(3, 2)
    x = Tensor.from_numpy(np.random.randn(4, 3))
    out = lin(x)
    assert out.shape == (4, 2)


def test_sequential():
    net = Sequential(Linear(3, 4), Linear(4, 2))
    x = Tensor.from_numpy(np.random.randn(2, 3))
    out = net(x)
    assert out.shape == (2, 2)
    assert len(list(net.parameters())) == 4


def test_module_list():
    ml = ModuleList([Linear(3, 3), Linear(3, 3)])
    assert len(ml) == 2
    assert len(list(ml.parameters())) == 4


def test_train_eval_propagates():
    m = _MLP()
    assert m.training and m.fc1.training and m.fc2.training
    m.eval()
    assert not m.training and not m.fc1.training and not m.fc2.training
    m.train()
    assert m.training and m.fc1.training


def test_forward_hook_order():
    m = Linear(3, 2)
    calls = []
    m.register_forward_pre_hook(lambda mod, inp: calls.append("pre"))
    m.register_forward_hook(lambda mod, inp, out: calls.append("post"))
    m(Tensor.from_numpy(np.random.randn(1, 3)))
    assert calls == ["pre", "post"]


def test_zero_grad():
    m = Linear(3, 2)
    x = Tensor.from_numpy(np.random.randn(2, 3))
    (m(x).sum()).backward()
    m.zero_grad()
    for p in m.parameters():
        assert p.grad is None


def test_linear_backward():
    m = Linear(3, 2)
    x = Tensor.from_numpy(np.random.randn(2, 3))
    (m(x).sum()).backward()
    assert m.weight.grad is not None
    assert m.bias.grad is not None
