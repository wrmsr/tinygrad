import numpy as np
import torch
import unittest
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam, SGD, AdamW

np.random.seed(1337)
x_init = np.random.randn(1, 4).astype(np.float32)
W_init = np.random.randn(4, 4).astype(np.float32)
m_init = np.random.randn(1, 4).astype(np.float32)


class TinyNet:
    def __init__(self, tensor):
        self.x = tensor(x_init.copy(), requires_grad=True)
        self.W = tensor(W_init.copy(), requires_grad=True)
        self.m = tensor(m_init.copy())

    def forward(self):
        out = self.x.matmul(self.W).relu()
        out = out.log_softmax(1)
        out = out.mul(self.m).add(self.m).sum()
        return out


def step(tensor, optim, steps=1, kwargs={}):
    net = TinyNet(tensor)
    optim = optim([net.x, net.W], **kwargs)
    for _ in range(steps):
        out = net.forward()
        optim.zero_grad()
        out.backward()
        optim.step()
    return net.x.detach().numpy(), net.W.detach().numpy()


class TestOptim(unittest.TestCase):
    def _test_optim(self, tinygrad_optim, torch_optim, steps, opts, atol, rtol):
        for x, y in zip(
            step(Tensor, tinygrad_optim, steps, kwargs=opts),
            step(torch.tensor, torch_optim, steps, kwargs=opts),
        ):
            np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

    def _test_sgd(self, steps, opts, atol, rtol):
        self._test_optim(SGD, torch.optim.SGD, steps, opts, atol, rtol)

    def _test_adam(self, steps, opts, atol, rtol):
        self._test_optim(Adam, torch.optim.Adam, steps, opts, atol, rtol)

    def _test_adamw(self, steps, opts, atol, rtol):
        self._test_optim(AdamW, torch.optim.AdamW, steps, opts, atol, rtol)

    def test_sgd(self):
        self._test_sgd(1, {"lr": 0.001}, 1e-6, 0)

    def test_sgd_high_lr(self):
        self._test_sgd(1, {"lr": 10}, 1e-6, 1e-5)

    def test_multistep_sgd(self):
        self._test_sgd(10, {"lr": 0.001}, 1e-6, 0)

    def test_multistep_sgd_high_lr(self):
        self._test_sgd(10, {"lr": 10}, 1e-6, 3e-4)

    def test_multistep_sgd_momentum(self):
        self._test_sgd(10, {"lr": 0.001, "momentum": 0.9}, 1e-6, 0)

    def test_multistep_sgd_high_lr_momentum(self):
        self._test_sgd(10, {"lr": 10, "momentum": 0.9}, 1e-5, 3e-4)

    def test_multistep_sgd_nesterov_momentum(self):
        self._test_sgd(10, {"lr": 0.001, "momentum": 0.9, "nesterov": True}, 1e-5, 0)

    def test_multistep_sgd_high_lr_nesterov_momentum(self):
        self._test_sgd(10, {"lr": 10, "momentum": 0.9, "nesterov": True}, 1e-5, 3e-4)

    def test_adam(self):
        self._test_adam(1, {"lr": 0.001}, 1e-5, 0)

    def test_adam_high_lr(self):
        self._test_adam(1, {"lr": 10}, 1e-5, 1e-5)

    def test_adamw(self):
        self._test_adamw(1, {"lr": 0.001}, 1e-5, 0)

    def test_adamw_high_lr(self):
        self._test_adamw(1, {"lr": 10}, 1e-5, 1e-5)

    def test_multistep_adam(self):
        self._test_adam(10, {"lr": 0.001}, 1e-5, 0)

    def test_multistep_adam_high_lr(self):
        self._test_adam(10, {"lr": 10}, 1e-5, 3e-4)

    def test_multistep_adamw(self):
        self._test_adamw(10, {"lr": 0.001}, 1e-5, 0)

    def test_multistep_adamw_high_lr(self):
        self._test_adamw(10, {"lr": 10}, 1e-5, 3e-4)


if __name__ == "__main__":
    unittest.main()
