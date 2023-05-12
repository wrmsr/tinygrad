import numpy as np
import torch
import unittest
import itertools
from tinygrad.tensor import Tensor, Device
from extra.gradcheck import numerical_jacobian, jacobian, gradcheck


x_init = np.random.randn(1, 3).astype(np.float32)
U_init = np.random.randn(3, 3).astype(np.float32)
V_init = np.random.randn(3, 3).astype(np.float32)
W_init = np.random.randn(3, 3).astype(np.float32)
m_init = np.random.randn(1, 3).astype(np.float32)


class TestTinygrad(unittest.TestCase):
    def test_plus_equals(self):
        a = Tensor.randn(10, 10)
        b = Tensor.randn(10, 10)
        c = a + b
        val1 = c.numpy()
        a += b
        val2 = a.numpy()
        np.testing.assert_allclose(val1, val2)

    def test_slicing(self):
        x = Tensor.randn(10, 10)
        slices = [0, 1, 9, -1, -10, None] + [slice(s, e) for s, e in itertools.combinations([0, 1, -1, None], r=2)] + [
            slice(9, 11), slice(-11, -9)]
        fmt = lambda s: f'{s.start}:{s.stop}' if isinstance(s, slice) else str(s)
        for s in list(itertools.product(slices, slices)) + [(None, 0, None, 0, None),
                                                            (slice(0, 2), None, None, slice(2, 4), None, None)]:
            np.testing.assert_equal(x.numpy()[s], x[s].numpy(),
                                    f'Test failed for slice x[{",".join(fmt(x) for x in s)}]')
        for s in [-11, 10]:
            with self.assertRaises(IndexError):
                x[s]
        with self.assertRaises(AssertionError):
            x[::2]
        with self.assertRaises(AssertionError):
            x[0, 0, 0]

    def test_backward_pass(self):
        def test_tinygrad():
            x = Tensor(x_init, requires_grad=True)
            W = Tensor(W_init, requires_grad=True)
            m = Tensor(m_init)
            out = x.dot(W).relu()
            out = out.log_softmax()
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.cpu().numpy(), x.grad.cpu().numpy(), W.grad.cpu().numpy()

        def test_pytorch():
            x = torch.tensor(x_init, requires_grad=True)
            W = torch.tensor(W_init, requires_grad=True)
            m = torch.tensor(m_init)
            out = x.matmul(W).relu()
            out = torch.nn.functional.log_softmax(out, dim=1)
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.detach().numpy(), x.grad, W.grad

        for x, y in zip(test_tinygrad(), test_pytorch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_backward_pass_diamond_model(self):
        def test_tinygrad():
            u = Tensor(U_init, requires_grad=True)
            v = Tensor(V_init, requires_grad=True)
            w = Tensor(W_init, requires_grad=True)
            x = u.mul(v).relu()
            y = u.mul(w).relu()
            out = x.add(y).mul(y).relu()
            out = out.log_softmax()
            out = out.sum()
            out.backward()
            return out.cpu().numpy(), u.cpu().grad.numpy(), v.cpu().grad.numpy(), w.cpu().grad.numpy()

        def test_pytorch():
            u = torch.tensor(U_init, requires_grad=True)
            v = torch.tensor(V_init, requires_grad=True)
            w = torch.tensor(W_init, requires_grad=True)
            x = u.mul(v).relu()
            y = u.mul(w).relu()
            out = x.add(y).mul(y).relu()
            out = torch.nn.functional.log_softmax(out, dim=1)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), u.grad, v.grad, w.grad

        for x, y in zip(test_tinygrad(), test_pytorch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_nograd(self):
        x = Tensor(x_init, requires_grad=False)
        m = Tensor(m_init, requires_grad=False)
        W = Tensor(W_init, requires_grad=True)
        tmp = x.mul(m)
        mm = tmp.matmul(W)
        out = mm.relu()
        out = out.sum()
        out.backward()
        assert x.grad is None
        assert m.grad is None
        assert tmp.grad is None
        assert mm.grad is not None
        assert W.grad is not None

    def test_dropout(self):
        Tensor.training = True
        n, rate = 1_000_000, 0.1
        w = Tensor.ones(n).dropout(rate)
        non_zeros = np.count_nonzero(w.cpu().numpy())
        expected = n * (1 - rate)
        np.testing.assert_allclose(non_zeros, expected, rtol=2e-3)

    # @unittest.skipUnless(Device.DEFAULT == Device.CPU, "float64 not supported on GPU")
    @unittest.skip("float64 support broken")
    def test_jacobian(self):
        W = np.random.RandomState(1337).random((10, 5))
        x = np.random.RandomState(7331).random((1, 10)) - 0.5

        torch_x = torch.tensor(x, requires_grad=True)
        torch_W = torch.tensor(W, requires_grad=True)
        torch_func = lambda x: torch.nn.functional.log_softmax(x.matmul(torch_W).relu(), dim=1)
        PJ = torch.autograd.functional.jacobian(torch_func, torch_x).squeeze().numpy()

        tiny_x = Tensor(x)
        tiny_W = Tensor(W)
        tiny_func = lambda x: x.dot(tiny_W).relu().log_softmax()
        J = jacobian(tiny_func, tiny_x)
        NJ = numerical_jacobian(tiny_func, tiny_x)

        np.testing.assert_allclose(PJ, J, atol=1e-5)
        np.testing.assert_allclose(PJ, NJ, atol=1e-5)

    # @unittest.skipUnless(Device.DEFAULT == Device.CPU, "float64 not supported on GPU")
    @unittest.skip("float64 support broken")
    def test_gradcheck(self):
        W = np.random.RandomState(1337).random((10, 5))
        x = np.random.RandomState(7331).random((1, 10)) - 0.5

        tiny_x = Tensor(x)
        tiny_W = Tensor(W)
        tiny_func = lambda x: x.dot(tiny_W).relu().log_softmax()

        self.assertTrue(gradcheck(tiny_func, tiny_x))

        # coarse approx. since a "big" eps and the non-linearities of the model
        self.assertFalse(gradcheck(tiny_func, tiny_x, eps=0.1))

    def test_random_fns_are_deterministic_with_seed(self):
        for random_fn in [Tensor.randn, Tensor.uniform, Tensor.scaled_uniform, Tensor.glorot_uniform]:
            with self.subTest(msg=f"Tensor.{random_fn.__name__}"):
                Tensor.manual_seed(1337)
                a = random_fn(10, 10).realize()
                Tensor.manual_seed(1337)
                b = random_fn(10, 10).realize()
                np.testing.assert_allclose(a.numpy(), b.numpy())


if __name__ == '__main__':
    unittest.main()
