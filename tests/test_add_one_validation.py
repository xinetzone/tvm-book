import numpy as np
import pytest


def test_add_one_validation_1d():
    import flexloopy
    x = np.ones((2, 2), dtype=np.float32)
    y = np.ones_like(x)
    with pytest.raises(ValueError):
        flexloopy.add_one(x, y)


def test_add_one_validation_dtype_message():
    import flexloopy
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([0, 0, 0], dtype=np.float32)
    # dtype 校验在 C++ 层，Python 层仅维度校验，这里保证能到 C++ 抛错
    with pytest.raises(Exception):
        flexloopy.add_one(x, y)