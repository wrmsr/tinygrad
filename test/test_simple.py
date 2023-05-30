import unittest

import numpy as np

from tinygrad.tensor import Tensor


class TestSymbolic(unittest.TestCase):
    def test_mul(self):
        print()

        xa = np.asarray([1.0, 2.0], dtype=np.float32)
        ya = np.asarray([3.0, 4.0], dtype=np.float32)
        print(xa)
        print(ya)

        xt = Tensor(xa, requires_grad=True)
        yt = Tensor(ya, requires_grad=True)

        zt = xt * yt
        zt.realize()

        za = zt.numpy()
        print(za)

    def test_mul_backward(self):
        xt = Tensor(np.asarray([1.0, 2.0], dtype=np.float32), requires_grad=True)
        yt = Tensor(np.asarray([3.0, 4.0], dtype=np.float32), requires_grad=True)

        zt = (xt * yt).sum()
        zt.backward()

        print(xt.grad.numpy())
        print(yt.grad.numpy())

    def test_cl(self):
        x = Tensor(list(range(8)), device='GPU')
        x2 = x * 2
        print(x2.numpy())
