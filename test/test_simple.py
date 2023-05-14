import unittest

import numpy as np

from tinygrad.tensor import Tensor


class TestSymbolic(unittest.TestCase):
    def test_mul(self):
        print()

        xa = np.asarray([1., 2.], dtype=np.float32)
        ya = np.asarray([3., 4.], dtype=np.float32)
        print(xa)
        print(ya)

        xt = Tensor(xa, requires_grad=True)
        yt = Tensor(ya, requires_grad=True)

        zt = xt * yt
        zt.realize()

        za = zt.numpy()
        print(za)

    def test_mul_backward(self):
        xt = Tensor(np.asarray([1., 2.], dtype=np.float32), requires_grad=True)
        yt = Tensor(np.asarray([3., 4.], dtype=np.float32), requires_grad=True)

        zt = (xt * yt).sum()
        zt.backward()

        print(xt.grad.numpy())
        print(yt.grad.numpy())
