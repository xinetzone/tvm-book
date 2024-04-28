from dataclasses import dataclass
import numpy as np


@dataclass(eq=True, order=True)
class TensorType:
    shape: tuple[int]
    dtype: str
    name: str = "data"

    @property
    def empty(self):
        '''空的输出张量'''
        return np.empty(shape=self.shape,
                        dtype=self.dtype)

    @property
    def nbytes(self):
        '''输出张量的存储字节数'''
        return self.empty.nbytes

    @property
    def nelement(self):
        '''输出张量的元素个数'''
        return self.empty.size

    def weak_eq(self, other):
        """排除对名称的依赖"""
        assert_shape = self.shape == other.shape
        assert_dtype = self.dtype == other.dtype
        return assert_shape and assert_dtype
