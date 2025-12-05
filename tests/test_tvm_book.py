import numpy as np
import pytest


def test_raise_error():
    import flexloopy

    with pytest.raises(Exception) as e:
        flexloopy.raise_error("hello")
    assert "hello" in str(e.value)


def test_add_one_cpu():
    import flexloopy

    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    y = np.empty_like(x)
    flexloopy.add_one(x, y)
    assert np.allclose(y, x + 1)


def test_intpair_object():
    import flexloopy

    pair = flexloopy.IntPair(3, 5)
    # reflection binds fields and methods
    assert pair.a == 3
    assert pair.b == 5
    assert pair.get_first() == 3