import numpy as np
import pytest


pytest_plugins = ["pytest_benchmark"]


@pytest.mark.parametrize("n", [1000, 1000000])
def test_add_one_benchmark(benchmark, n):
    import flexloopy
    x = np.arange(n, dtype=np.float32)
    y = np.empty_like(x)
    try:
        benchmark(lambda: flexloopy.add_one(x, y))
    except FileNotFoundError:
        pytest.skip("shared library not built")