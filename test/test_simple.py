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

        xt = Tensor(xa)
        yt = Tensor(ya)

        zt = xt * yt
        zt.realize()

        za = zt.numpy()
        print(za)
