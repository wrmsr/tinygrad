import torch
import time
import numpy as np
import unittest
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, IMAGE
from tinygrad.lazy import Device

FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
PRINT_TENSORS = getenv("PRINT_TENSORS", 0)
def helper_test_op(shps, torch_fxn, tinygrad_fxn=None, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3, forward_only=False, vals=None, a=-0.5, b=3):
  if tinygrad_fxn is None: tinygrad_fxn = torch_fxn
  torch.manual_seed(0)
  np.random.seed(0)
  if shps is None:
    ts = [torch.tensor(x, requires_grad=True) for x in vals]
  else:
    ts = [torch.tensor((np.random.random(size=x).astype(np.float32)+a)*b, requires_grad=True) for x in shps]

  tst = [Tensor(x.detach().numpy(), requires_grad=not FORWARD_ONLY) for x in ts]

  st = time.monotonic()
  out = torch_fxn(*ts)
  torch_fp = time.monotonic() - st

  st = time.monotonic()
  ret = tinygrad_fxn(*tst).realize()
  tinygrad_fp = time.monotonic() - st

  def compare(s, x,y,atol,rtol):
    if PRINT_TENSORS: print(s, x, y)
    if y.shape != tuple(): assert x.shape == y.shape, f"shape mismatch (tinygrad){x.shape} != (torch){y.shape}"
    try:
      np.testing.assert_allclose(x,y, atol=atol, rtol=rtol)
    except Exception:
      raise Exception(f"{s} failed shape {x.shape}")

  compare("forward pass", ret.numpy(), out.detach().numpy(), atol=atol, rtol=rtol)

  torch_fbp, tinygrad_fbp = np.nan, np.nan
  if not forward_only and not FORWARD_ONLY:
    st = time.monotonic()
    (out+1).square().mean().backward()
    torch_fbp = time.monotonic() - st

    st = time.monotonic()
    (ret+1).square().mean().backward()
    for tt in tst: tt.grad.realize()
    tinygrad_fbp = time.monotonic() - st

    for i, (t, tt) in enumerate(zip(ts, tst)):
      compare(f"backward pass tensor {i}", tt.grad.numpy(), t.grad.detach().numpy(), atol=grad_atol, rtol=grad_rtol)

  print("\ntesting %40r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms " % (shps, torch_fp*1000, tinygrad_fp*1000, torch_fbp*1000, tinygrad_fbp*1000), end="")

class TestOps(unittest.TestCase):
  def test_zeros(self):
    helper_test_op([], lambda: torch.zeros(45,65), lambda: Tensor.zeros(45,65), forward_only=True)
  def test_ones(self):
    helper_test_op([], lambda: torch.ones(45,65), lambda: Tensor.ones(45,65), forward_only=True)
  def test_eye(self):
    helper_test_op([], lambda: torch.eye(10), lambda: Tensor.eye(10), forward_only=True)
  def test_arange(self):
    helper_test_op([], lambda: torch.arange(10), lambda: Tensor.arange(10), forward_only=True)

  def _test_cmp(self, fxn, reverse=True):
    for shps in [[(3, 4, 5), (3, 4, 5)], [(3, 4, 5), (5,)], [(5,), (3, 4, 5)]]:
      helper_test_op(shps, fxn, fxn, forward_only=True)
    helper_test_op(None, fxn, fxn, forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    helper_test_op(None, lambda x,y: fxn(x,2), lambda x,y: fxn(x,2), forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    if reverse: helper_test_op(None, lambda x,y: fxn(2,y), lambda x,y: fxn(2,y), forward_only=True, vals=[[0.,1,2], [2.,1,0]])

  def test_cmp_eq(self): self._test_cmp(lambda x,y: x==y, reverse=False)
  def test_cmp_gt(self): self._test_cmp(lambda x,y: x>y)
  def test_cmp_ge(self): self._test_cmp(lambda x,y: x>=y)
  def test_cmp_lt(self): self._test_cmp(lambda x,y: x<y)
  def test_cmp_le(self): self._test_cmp(lambda x,y: x<=y)

  def test_cmp_eq_backwards(self):
    t1 = torch.ones(4, requires_grad=True)
    t2 = torch.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (t1 == t2).sum().backward)
    tt1 = Tensor.ones(4, requires_grad=True)
    tt2 = Tensor.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (tt1.eq(tt2)).sum().backward)

  def test_cmp_lt_backwards(self):
    t1 = torch.ones(4, requires_grad=True)
    t2 = torch.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (t1 < t2).sum().backward)
    tt1 = Tensor.ones(4, requires_grad=True)
    tt2 = Tensor.ones(4, requires_grad=True)
    self.assertRaises(RuntimeError, (tt1 < tt2).sum().backward)

  def test_maximum(self):
    helper_test_op([(45,65), (45,65)], torch.maximum, Tensor.maximum)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1., 0., 3., 4.], [1., 2., 3., 0.]])
  def test_minimum(self):
    helper_test_op([(45,65), (45,65)], torch.minimum, Tensor.minimum)
  def test_add(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x+y, Tensor.add)
  def test_add_simple(self):
    helper_test_op([(256), (256)], lambda x,y: x+y, Tensor.add, forward_only=True)
  def test_broadcasted_add(self):
    helper_test_op([(45,65), (45,1)], lambda x,y: x+y, lambda x,y: x+y)
  def test_broadcasted_add_2(self):
    helper_test_op([(45,65), (65,)], lambda x,y: x+y, lambda x,y: x+y)
  def test_sub(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x-y, Tensor.sub)
  def test_neg(self):
    helper_test_op([(45,65)], lambda x: -x)
  def test_mul(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x*y, Tensor.mul)
  def test_div(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x/y, Tensor.div)
  def test_div_const(self):
    helper_test_op([(45,65)], lambda x: x/255, lambda x: x/255)
    helper_test_op([(45,65)], lambda x: x/1, lambda x: x/1)
    helper_test_op([(45,65)], lambda x: 1/x, lambda x: 1/x)
    helper_test_op([(45,65)], lambda x: x/2, lambda x: x/2)
    helper_test_op([(45,65)], lambda x: 2/x, lambda x: 2/x)
  def test_pow(self):
    helper_test_op([(45,65)], lambda x: x**2, lambda x: Tensor.pow(x,2), a=0)
    helper_test_op([(45,65)], lambda x: x**3, lambda x: Tensor.pow(x,3), a=0)
    helper_test_op([(45,65)], lambda x: x**-2, lambda x: Tensor.pow(x,-2), a=0)
    helper_test_op([(45,65), (45,65)], lambda x,y: x**y, Tensor.pow, a=0)
  def test_pow_const(self):
    helper_test_op([(45,65)], lambda x: x**1.0, lambda x: x**1.0)
    helper_test_op([(45,65)], lambda x: 1.0**x, lambda x: 1.0**x)
    helper_test_op([(45,65)], lambda x: x**2.0, lambda x: x**2.0)
    helper_test_op([(45,65)], lambda x: 2.0**x, lambda x: 2.0**x)
  def test_sqrt(self):
    helper_test_op([(45,65)], lambda x: x.sqrt(), Tensor.sqrt, a=0)
  
  def test_sin(self):
    helper_test_op([(45,65)], lambda x: x.sin(), Tensor.sin, a=0)
  def test_cos(self):
    helper_test_op([(45,65)], lambda x: x.cos(), Tensor.cos, a=0)
  def test_tan(self):
    helper_test_op([(45,65)], lambda x: x.tan(), Tensor.tan, a=0)
    
  def test_relu(self):
    helper_test_op([(64,64)], lambda x: x.relu(), Tensor.relu)
  def test_relu_exact(self):
    helper_test_op(None, lambda x: x.relu(), Tensor.relu, vals=[[-1.,0,1]])
  def test_relu_maximum_exact(self):
    helper_test_op(None, lambda x: torch.maximum(x, torch.zeros_like(x, requires_grad=False)), lambda x: Tensor.maximum(x, 0), vals=[[-1.,0,1]])
  def test_leakyrelu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.leaky_relu(x,0.01), Tensor.leakyrelu)
  def test_celu(self):
    for val in range(1, 5):
      helper_test_op([(45,65)], lambda x: torch.nn.functional.celu(x,val), lambda x: x.celu(val))
  def test_abs(self):
    helper_test_op([(45,65)], lambda x: torch.abs(x), Tensor.abs)
  def test_log(self):
    helper_test_op([(45,65)], lambda x: torch.log(x), Tensor.log)
  def test_exp(self):
    helper_test_op([(45,65)], lambda x: torch.exp(x), Tensor.exp)
  def test_sign(self):
    helper_test_op([(45,65)], lambda x: torch.sign(x), Tensor.sign)
  def test_softsign(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.softsign(x), Tensor.softsign)
  def test_sigmoid(self):
    helper_test_op([(45,65)], lambda x: x.sigmoid(), Tensor.sigmoid)
  def test_softplus(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.softplus(x), Tensor.softplus, atol=1e-6, grad_atol=1e-6)
  @unittest.skip("not supported in older pytorch")
  def test_gelu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu)
  def test_quick_gelu(self):
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu)
  def test_elu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.elu(x), Tensor.elu)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.elu(x, alpha=0.1), lambda x: Tensor.elu(x, alpha=0.1))
  def test_relu6(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.relu6(x), Tensor.relu6)
  def test_hardswish(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.hardswish(x), Tensor.hardswish, atol=1e-6, grad_atol=1e-6)
  def test_mish(self):
    def _mish_pytorch(x):
      return x*torch.tanh(torch.nn.functional.softplus(x))
    helper_test_op([(45,65)], _mish_pytorch, Tensor.mish, atol=1e-4)
  def test_dot(self):
    helper_test_op([(45,65), (65,100)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)
  def test_matmul_simple(self):
    helper_test_op([(4), (4,4)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)
  def test_matmul(self):
    helper_test_op([(64), (64,99)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-4)
  def test_gemm(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-3)
  def test_broadcastdot(self):
    helper_test_op([(10,45,65), (65,45)], lambda x,y: x @ y, Tensor.dot, atol=1e-4)
  def test_multidot(self):
    helper_test_op([(10,45,65), (10,65,45)], lambda x,y: x @ y, Tensor.dot, atol=1e-4)
    helper_test_op([(3,3,45,65), (3,3,65,45)], lambda x,y: x @ y, Tensor.dot, atol=1e-4)
  def test_sum_simple(self):
    helper_test_op(None, lambda x: x.sum(), Tensor.sum, vals=[[1.,1.]])
  def test_sum_full(self):
    helper_test_op([(16384)], lambda x: x.sum(), lambda x: x.sum())
  def test_sum_relu(self):
    helper_test_op([(3,4,5)], lambda x: x.relu().sum().relu(), lambda x: x.relu().sum().relu())
  def test_sum(self):
    helper_test_op([(45,3)], lambda x: x.sum(), Tensor.sum)
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=3), lambda x: Tensor.sum(x, axis=3))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(1,3)), lambda x: Tensor.sum(x, axis=(1,3)))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(0,2)), lambda x: Tensor.sum(x, axis=(0,2)))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(1,2)), lambda x: Tensor.sum(x, axis=(1,2)))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=1), lambda x: Tensor.sum(x, axis=1))
  def test_min(self):
    helper_test_op([(3,3)], lambda x: x.min(), Tensor.min)
    helper_test_op([(45,3)], lambda x: x.min(), Tensor.min)
    helper_test_op([(45,3)], lambda x: x.min().mul(0.5), lambda x: Tensor.min(x).mul(0.5))
  def test_max(self):
    helper_test_op([(45,3)], lambda x: x.max(), Tensor.max)
    helper_test_op([(45,3)], lambda x: x.max().mul(0.5), lambda x: Tensor.max(x).mul(0.5))
    helper_test_op(None, lambda x: x.max().mul(0.5), lambda x: Tensor.max(x).mul(0.5),
            vals=[
                [[1.0,1.0,0.0,1.0]],
                ])
    helper_test_op([(3,4,5,6)], lambda x: x.max(axis=1)[0], lambda x: Tensor.max(x, axis=1))
  def test_mean(self):
    helper_test_op([(3,4,5,6)], lambda x: x.mean())
  def test_mean_axis(self):
    helper_test_op([(3,4,5,6)], lambda x: x.mean(axis=(1,2)), lambda x: Tensor.mean(x, axis=(1,2)))
  def test_std(self):
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, unbiased=False), lambda x: Tensor.std(x))
  def test_std_axis(self):
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, unbiased=False, dim=0), lambda x: Tensor.std(x, axis=0))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, unbiased=False, dim=2), lambda x: Tensor.std(x, axis=2))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, unbiased=False, dim=[1, 2]), lambda x: Tensor.std(x, axis=[1, 2]))
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, unbiased=False, dim=None), lambda x: Tensor.std(x, axis=None))
  def test_std_keepdim(self):
    helper_test_op([(45, 65, 85)], lambda x: torch.std(x, keepdim=True), lambda x: Tensor.std(x, keepdim=True))
  def test_log_softmax(self):
    helper_test_op([(45,65)], lambda x: torch.nn.LogSoftmax(dim=1)(x), Tensor.log_softmax, atol=1e-7, grad_atol=1e-7)
  def test_log_softmax_other_axis(self):
    helper_test_op([(10,10,10)], lambda x: x.log_softmax(0), lambda x: x.log_softmax(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(10,10,10)], lambda x: x.log_softmax(1), lambda x: x.log_softmax(1), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(10,10,10)], lambda x: x.log_softmax(2), lambda x: x.log_softmax(2), atol=1e-7, grad_atol=1e-7)
  def test_tanh(self):
    helper_test_op([(45,65)], lambda x: x.tanh(), Tensor.tanh, atol=1e-6, grad_atol=1e-6)
  def test_hardtanh(self):
    for val in range(10, 30, 5):
      helper_test_op([(45,65)], lambda x: torch.nn.functional.hardtanh(x,-val, val), lambda x: x.hardtanh(-val, val), atol=1e-6, grad_atol=1e-6)
  def test_topo_sort(self):
    helper_test_op([(45,65)], lambda x: (x+x)*x, lambda x: x.add(x).mul(x), atol=1e-6, grad_atol=1e-6)

  def test_scalar_mul(self):
    helper_test_op([(45,65)], lambda x: x*2, lambda x: x*2)
  def test_scalar_rmul(self):
    helper_test_op([(45,65)], lambda x: 2*x, lambda x: 2*x)

  def test_scalar_sub(self):
    helper_test_op([(45,65)], lambda x: x-2, lambda x: x-2)
  def test_scalar_rsub(self):
    helper_test_op([(45,65)], lambda x: 2-x, lambda x: 2-x)

  def test_flip_eye_crash(self):
    helper_test_op([], lambda: (torch.eye(10)@torch.eye(10).flip(0)),
                       lambda: (Tensor.eye(10)@Tensor.eye(10).flip(0)), forward_only=True)

  def test_broadcast_full(self):
    for torch_op, tinygrad_op in [(torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul),
                                  (torch.div, Tensor.div), (torch.pow, Tensor.pow)]:
      for shapes in [((5,13,24,16), (5,1,24,1)), ((1,3,1,7,1), (2,1,5,1,8))]:
        with self.subTest(op=torch_op.__name__, shapes=shapes):
          helper_test_op(shapes, torch_op, tinygrad_op, a=-0.5 if tinygrad_op != Tensor.pow else 0.0)

  def test_broadcast_simple(self):
    helper_test_op([(45,65), (45,1)], lambda x,y: x/y, lambda x,y: x/y)

  def test_broadcast_partial(self):
    for torch_op, tinygrad_op in [(torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul),
                                  (torch.div, Tensor.div), (torch.pow, Tensor.pow)]:
      for shapes in [((1,32,32,32), (1,32,1,1)), ((5,13,24,16,2), (1,13,24,1,1)),
                     ((4,1), (4,5)), ((1,4), (5,4))]:
        with self.subTest(op=torch_op.__name__, shapes=shapes):
          # NOTE: ANE backwards?
          helper_test_op(shapes, torch_op, tinygrad_op, a=-0.5 if tinygrad_op != Tensor.pow else 0.0)

  def test_slice_simple(self):
    helper_test_op([(3,3)], lambda x: x[1:2, 1:2], lambda x: x[1:2, 1:2])

  def test_slice(self):
    helper_test_op([(3,3,3,3)], lambda x: x[1:2], lambda x: x[1:2])
    helper_test_op([(3,3,3,3)], lambda x: x[1:2, 1:2], lambda x: x[1:2, 1:2])
    helper_test_op([(3,3,3,3)], lambda x: x[1:2, 1:2, 0:-1], lambda x: x[1:2, 1:2, 0:-1])

  def test_slice_one(self):
    helper_test_op([(3)], lambda x: x[1], lambda x: x[1])

  def test_slice_one_multi(self):
    helper_test_op([(10,10)], lambda x: x[1], lambda x: x[1])

  def test_pad2d(self):
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4)), lambda x: x.pad2d(padding=(1,2,3,4)))

  def test_transpose(self):
    helper_test_op([(3,3,3)], lambda x: x.transpose(1,2), lambda x: x.transpose(1,2))
    helper_test_op([(3,3,3)], lambda x: x.transpose(0,2), lambda x: x.transpose(0,2))
    helper_test_op([(1,2,3,4)], lambda x: x.movedim((3,0,2,1),(0,1,2,3)), lambda x: x.permute(order=(3,0,2,1)))
    helper_test_op([(3,4,5,6)], lambda x: x.movedim((3,2,1,0),(0,1,2,3)), lambda x: x.permute(order=(3,2,1,0)))

  def test_reshape(self):
    helper_test_op([(4,3,6,6)], lambda x: torch.reshape(x, (-1,3,6,6)), lambda x: x.reshape(shape=(-1,3,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: torch.reshape(x, (-1,1,6,6)), lambda x: x.reshape(shape=(-1,1,6,6)))

  def test_flip(self):
    helper_test_op([(4,3,6,6)], lambda x: torch.flip(x, (0,)), lambda x: x.flip(axis=(0,)))
    helper_test_op([(4,3,6,6)], lambda x: torch.flip(x, (0,1)), lambda x: x.flip(axis=(0,1)))
    helper_test_op([(4,3,6,6)], lambda x: torch.flip(x, (0,1,3)), lambda x: x.flip(axis=(0,1,3)))
    helper_test_op([(4,3,6,6)], lambda x: torch.flip(x, (3,)), lambda x: x.flip(axis=(3,)))
    helper_test_op([(4,3,6,6)], lambda x: torch.flip(x, (0,1,3)).flip((0,)), lambda x: x.flip(axis=(0,1,3)).flip(0))
    helper_test_op([(4,3,6,6)], lambda x: torch.flip(x, (3,)), lambda x: x.flip(axis=(-1,)))

  def test_unsqueeze(self):
    helper_test_op([(4,3,6,6)], lambda x: torch.unsqueeze(x, 0), lambda x: x.unsqueeze(dim=0))
    helper_test_op([(4,3,6,6)], lambda x: torch.unsqueeze(x, 4), lambda x: x.unsqueeze(dim=4))
    helper_test_op([(4,3,6,6)], lambda x: torch.unsqueeze(x, -1), lambda x: x.unsqueeze(dim=-1))
    helper_test_op([(4,3,6,6)], lambda x: torch.unsqueeze(x, -3), lambda x: x.unsqueeze(dim=-3))

  def test_flatten(self):
    for axis in range(3):
      helper_test_op([(4,3,6,6)], lambda x: torch.flatten(x, start_dim=axis), lambda x: x.flatten(axis))

  def test_detach(self):
    helper_test_op([(4,3,6,6)], lambda x: x.detach(), lambda x: x.detach(), forward_only=True)

  def test_expand(self):
    arg = (4,3,2,6)
    helper_test_op([(4,3,1,6)], lambda x: x.expand(arg), lambda x: x.expand(shape=arg))

  @unittest.skip("very slow")
  def test_sd_big_conv(self):
    # internal shape (1, 1, 512, 62, 62, 512, 3, 3) overflows a int
    helper_test_op([(1,256,64,64), (512,256,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w), atol=1e-2)

  @unittest.skip("slow")
  def test_large_bs_conv(self):
    # large batch size can cause OpenCL image to exceed max image height on macOS
    # (or cause the conv kernel to overflow short sampling coords)
    helper_test_op([(4096,3,3,3), (1,3,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w), atol=1e-4, rtol=1e-2)

  @unittest.skip("slow")
  def test_large_ic_conv(self):
    # large input channel count can cause OpenCL image to exceed max image width on macOS
    helper_test_op([(1,2048,3,3), (1,2048,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w), atol=1e-4)

  def test_biased_conv2d(self):
    C = 8
    helper_test_op([(1,C,5,5), (C,C,1,1), (C,)],
      lambda x,w,b: torch.nn.functional.conv2d(torch.nn.functional.conv2d(x,w,b).relu(),w,b),
      lambda x,w,b: Tensor.conv2d(x,w,b).relu().conv2d(w,b), atol=1e-4)

  def test_simple_conv2d(self):
    helper_test_op([(1,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_simple_conv3d(self):
    helper_test_op([(1,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv3d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_padded_conv3d(self):
    helper_test_op([(1,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv3d(x,w,padding=1).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=[1,1,1,1,1,1]).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_simple_conv2d_m4(self):
    helper_test_op([(1,16,18,18), (16,16,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_simple_conv2d_1x1(self):
    helper_test_op([(1,4,9,9), (4,4,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_simple_conv2d_1x1_m4(self):
    helper_test_op([(1,16,32,32), (16,16,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_nested_conv2d(self):
    helper_test_op([(1,32,9,9), (32,32,3,3), (32,32,3,3)],
      lambda x,w1,w2: torch.nn.functional.conv2d(torch.nn.functional.conv2d(x,w1).relu(), w2).relu(),
      lambda x,w1,w2: x.conv2d(w1).relu().conv2d(w2).relu(), atol=1e-4, grad_rtol=1e-5)

  # expect reduce nodes == 3
  def test_simple_conv2d_nhwc(self):
    # weights (from tf): filter_height x filter_width x in_channels x out_channels
    helper_test_op([(2,9,9,10), (3,3,10,20)],
      lambda x,w: torch.nn.functional.conv2d(x.permute(0,3,1,2),w.permute(3,2,0,1)).relu(),
      lambda x,w: Tensor.conv2d(x.permute(0,3,1,2),w.permute(3,2,0,1)).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_simple_conv2d_batched(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
      lambda x,w: Tensor.conv2d(x,w).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_conv2d(self):
    for bs in [1,8]:
      for cin in [1,3]:
        for groups in [1,3] if cin == 3 else [1]:
          for H in [1,2,5]:
            for W in [1,2,3,5]:
              with self.subTest(batch_size=bs, channels=cin, groups=groups, height=H, width=W):
                helper_test_op([(bs,cin,11,28), (6,cin//groups,H,W)],
                  lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
                  lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_large_input_conv2d(self):
    bs = 4
    cin = 16
    groups = 1
    H = 5
    W = 2
    helper_test_op([(bs,cin,64,64), (6,cin//groups,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      # needed to relax tolerance on NVIDIA
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-3, grad_rtol=1e-5)

  def test_simple_grouped_conv2d(self):
    bs = 1
    groups = 2
    rcout = 1
    cin = 2
    helper_test_op([(bs,groups*cin,1,1), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_medium_grouped_conv2d(self):
    bs = 1
    groups = 2
    rcout = 2
    cin = 2
    helper_test_op([(bs,groups*cin,1,1), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_depthwise_conv2d(self):
    bs = 1
    groups = 32
    rcout = 1
    cin = 1
    helper_test_op([(bs,groups*cin,32,32), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_grouped_conv2d(self):
    bs = 4
    groups = 5
    rcout = 7
    cin = 3
    helper_test_op([(bs,groups*cin,5,5), (groups*rcout,cin,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_fancy_conv2d(self):
    bs = 2
    cin = 3
    cout = 1
    groups = 3
    H,W = 3,3
    helper_test_op([(bs,cin,11,28), (groups*cout,cin//groups,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
      lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

  def test_strided_conv2d_simple(self):
    bs,H,W = 2,3,1
    helper_test_op([(bs,1,5,1), (1,1,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,stride=2).relu(),
      lambda x,w: Tensor.conv2d(x,w,stride=2).relu(), atol=1e-4)

  def test_strided_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    with self.subTest(stride := 2):
      helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
        lambda x,w: torch.nn.functional.conv2d(x,w,stride=2).relu(),
        lambda x,w: Tensor.conv2d(x,w,stride=stride).relu(), atol=1e-4)
    with self.subTest(stride := (2,1)):
      helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
        lambda x,w: torch.nn.functional.conv2d(x,w,stride=stride).relu(),
        lambda x,w: Tensor.conv2d(x,w,stride=(2,1)).relu(), atol=1e-4)

  def test_negative_padding_conv2d(self):
    n,k = 10, 3
    helper_test_op([(1,1,n,n), (1,1,k,k)],
      lambda x,w: torch.nn.functional.conv2d(x[:, :, 1:-1, 1:-1],w).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=-1).relu(), atol=1e-4)
    helper_test_op([(1,1,n,n), (1,1,k,k)],
      lambda x,w: torch.nn.functional.conv2d(x[:, :, 1:, 1:],w).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=(-1,0,-1,0)).relu(), atol=1e-4)

  def test_simple_padding_conv2d(self):
    p = (1,1,1,1)
    helper_test_op(None,
      lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=p).relu(), atol=1e-4, vals=[[[[[2.,3.]]]], [[[[1.]]]]])

  def test_asymmetric_padding_conv2d(self):
    for p in [(0,1,0,1), (2,1,2,1), (2,0,2,1)]:
      with self.subTest(padding := p):
        for n in [3,4]:
          for k in [2]:
            helper_test_op([(1,1,n,n), (1,1,k,k)],
              lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w).relu(),
              lambda x,w: Tensor.conv2d(x,w,padding=p).relu(), atol=1e-4)
            helper_test_op([(1,1,n,n), (1,1,k,k)],
              lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w).relu(),
              lambda x,w: Tensor.conv2d(x,w,padding=p).relu(), atol=1e-4)

  def test_padded_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    for p in [2, (2,1), (2,2)]:
      with self.subTest(padding := p):
        helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
          lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding).relu(),
          lambda x,w: Tensor.conv2d(x,w,padding=padding).relu(), atol=1e-4)

  def test_padded_conv2d_bs1(self):
    bs = 1
    cin = 3
    H,W = 3,3
    padding = 1
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding).relu(),
      lambda x,w: Tensor.conv2d(x,w,padding=padding).relu(), atol=1e-4)

  def test_dilated_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    for d in [2, (2,1)]:
      with self.subTest(dilation := d):
        helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
          lambda x,w: torch.nn.functional.conv2d(x,w,dilation=dilation).relu(),
          lambda x,w: Tensor.conv2d(x,w,dilation=dilation).relu(), atol=1e-4)

  def test_maxpool2d_simple(self):
    ksz = (2,2)
    helper_test_op([(1,1,2,3)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
      lambda x: Tensor.max_pool2d(x, kernel_size=ksz))

  def test_maxpool2d(self):
    for ksz in [(2,2), (3,3), 2, 3, (3,2), (5,5), (5,1)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([(32,2,110,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
          lambda x: Tensor.max_pool2d(x, kernel_size=ksz))

  def test_maxpool2d_bigger_stride(self):
    for stride in [(2,3), (3,2), 2, 3]:
      with self.subTest(stride=stride):
        helper_test_op([(32,2,110,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2,2), stride=stride),
          lambda x: Tensor.max_pool2d(x, kernel_size=(2,2), stride=stride))

  @unittest.skipIf(Device.DEFAULT == "CUDA", "CUDA fails on this")
  def test_maxpool2d_unit_stride(self):
    helper_test_op([(32,2,110,28)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), stride=1),
      lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), stride=1))

  def test_maxpool2d_smaller_stride(self):
    for stride in [(2,3), (3,2), 2, 3]:
      with self.subTest(stride=stride):
        helper_test_op([(32,2,110,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), stride=stride),
          lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), stride=stride))

  def test_avgpool2d(self):
    shape = (32,2,111,28)
    for ksz in [(2,2), (3,3), (3,2), (5,5), (5,1)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz), rtol=1e-5)

  def test_global_avgpool2d(self):
    helper_test_op([(32,2,111,28)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(111,28)),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(111,28)), rtol=1e-5)

  def test_cat(self):
    for dim in range(-1, 2):
      helper_test_op([(45,65), (45,65)], lambda x,y: torch.cat((x,y), dim), lambda x,y: x.cat(y, dim=dim))

  def test_multicat(self):
    for dim in range(-1, 2):
      helper_test_op([(45,65), (45,65), (45,65)], lambda x,y,z: torch.cat((x,y,z), dim), lambda x,y,z: x.cat(y, z, dim=dim))

  def test_stack(self):
    x = Tensor.randn(45, 65, 3)

    for dim in range(-1, 3):
      helper_test_op([(45, 65, 3), (45, 65, 3), (45, 65, 3)], lambda x, y, z: torch.stack((x, y, z), dim=dim), lambda x, y, z: Tensor.stack([x, y, z], dim=dim))

    with self.assertRaises(IndexError):
      Tensor.stack([x], dim=77)

  def test_repeat(self):
    x = Tensor.randn(45, 65, 3)
    base_repeats = [2, 4, 3]

    for reps in [[], [4], [2, 1], [3, 2, 2]]:
      repeats = base_repeats + reps
      helper_test_op([(45, 65, 3)], lambda x: x.repeat(*repeats), lambda x: x.repeat(repeats))

    with self.assertRaises(AssertionError):
      x.repeat((2, 4))

    with self.assertRaises(AssertionError):
      x.repeat((2, 0, 4))


  def test_clip(self):
    helper_test_op([(45,65)], lambda x: x.clip(-2.3, 1.2), lambda x: x.clip(-2.3, 1.2))

  def test_matvec(self):
    helper_test_op([(1,128), (128,128), (128,128)], lambda x,y,z: (x@y).relu()@z, atol=1e-4)

  # this was the failure in llama early realizing freqs_cis
  def test_double_slice(self):
    helper_test_op([(4,4)], lambda x: x[:, 1:2][1:2])
    helper_test_op([(4,4)], lambda x: x[1:3][1:2])
    helper_test_op([(4,4)], lambda x: x[:, 1:2][0:1])
    helper_test_op([(4,4)], lambda x: x[:, 1:2][:, 0:1])

if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main(verbosity=2)
