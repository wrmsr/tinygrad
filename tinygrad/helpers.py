import dataclasses as dc
import functools
import math
import os
import typing as ta

import numpy as np


ShapeType = ta.Tuple[int, ...]


# NOTE: helpers is not allowed to import from anything else in tinygrad

def dedup(x):
    return list(dict.fromkeys(x))  # retains list order


def prod(x: ta.Union[ta.List[int], ta.Tuple[int, ...]]) -> int:
    return math.prod(x)


def argfix(*x):
    if len(x) == 0:
        return ()
    if isinstance(x[0], (tuple, list)):
        if len(x) > 1:
            raise TypeError('wtf')
        return tuple(x[0])
    return tuple(x)


def argsort(x):
    # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
    return type(x)(sorted(range(len(x)), key=x.__getitem__))


def all_same(items):
    return all(x == items[0] for x in items) if len(items) > 0 else True


def colored(st, color, background=False, bright=False):
    # replace the termcolor library with one line
    cols = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
    if color is None:
        return st
    return f"\u001b[{10 * background + 60 * bright + 30 + cols.index(color)}m{st}\u001b[0m"


def partition(lst, fxn):
    return [x for x in lst if fxn(x)], [x for x in lst if not fxn(x)]


def make_pair(x: ta.Union[int, ta.Tuple[int, ...]], cnt=2) -> ta.Tuple[int, ...]:
    return (x,) * cnt if isinstance(x, int) else x


def flatten(l: ta.Iterator):
    return [item for sublist in l for item in sublist]


def mnum(i) -> str:
    return str(i) if i >= 0 else f"m{-i}"


@functools.lru_cache(maxsize=None)
def getenv(key, default=0):
    return type(default)(os.getenv(key, default))


DEBUG = getenv("DEBUG", 0)
IMAGE = getenv("IMAGE", 0)


# **** tinygrad now supports dtypes! *****

class DType(ta.NamedTuple):
    priority: int  # this determines when things get upcasted
    itemsize: int
    name: str
    np: type  # TODO: someday this will be removed with the "remove numpy" project

    def __repr__(self):
        return f"dtypes.{self.name}"


# dependent typing?
class ImageDType(DType):
    def __new__(cls, priority, itemsize, name, np, shape):
        return super().__new__(cls, priority, itemsize, name, np)

    def __init__(self, priority, itemsize, name, np, shape):
        self.shape: ta.Tuple[int, ...] = shape  # arbitrary arg for the dtype, used in image for the shape
        super().__init__()

    def __repr__(self):
        return f"dtypes.{self.name}({self.shape})"


class LazyNumpyArray:
    def __init__(self, fxn, shape, dtype):
        self.fxn, self.shape, self.dtype = fxn, shape, dtype

    def __call__(self) -> np.ndarray:
        return np.require(
            self.fxn(self) if callable(self.fxn) else self.fxn,
            dtype=self.dtype,
            requirements='C',
        ).reshape(self.shape)

    def reshape(self, new_shape):
        return LazyNumpyArray(self.fxn, new_shape, self.dtype)

    def copy(self):
        return self if callable(self.fxn) else LazyNumpyArray(self.fxn, self.shape, self.dtype)

    def astype(self, typ):
        return LazyNumpyArray(self.fxn, self.shape, typ)


@dc.dataclass()
class dtypes:
    float16: ta.Final[DType] = DType(0, 2, "half", np.float16)
    float32: ta.Final[DType] = DType(1, 4, "float", np.float32)
    int32: ta.Final[DType] = DType(1, 4, "int", np.int32)
    int64: ta.Final[DType] = DType(2, 8, "int64", np.int64)

    @staticmethod
    def from_np(x) -> DType:
        return dc.asdict(dtypes())[np.dtype(x).name]


class GlobalCounters:
    global_ops: ta.ClassVar[int] = 0
    global_mem: ta.ClassVar[int] = 0
    time_sum_s: ta.ClassVar[float] = 0.0
    kernel_count: ta.ClassVar[int] = 0
    mem_used: ta.ClassVar[int] = 0  # NOTE: this is not reset
    cache: ta.ClassVar[ta.Optional[ta.List[ta.Tuple[ta.Callable, ta.Any]]]] = None

    @staticmethod
    def reset():
        GlobalCounters.global_ops = 0
        GlobalCounters.global_mem = 0
        GlobalCounters.time_sum_s = 0.0
        GlobalCounters.kernel_count = 0
        GlobalCounters.cache = None
