"""Public Python API for the flexloopy package."""
from typing import Any

# tvm-ffi-stubgen(begin): export/_ffi_api
# fmt: off
# isort: off
from ._ffi_api import *  # noqa: F403
from ._ffi_api import __all__ as _ffi_api__all__
if "__all__" not in globals():
    __all__ = []
__all__.extend(_ffi_api__all__)
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)

# Version handling
try:
    from ._version import __version__, __version_tuple__  # type: ignore[import-not-found]
except ImportError:
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0", "7d34eb8ab.d20250913")

# Add version to __all__
__all__.extend(["__version__", "__version_tuple__"])

# Custom wrapper for add_one with validation
def add_one(x: Any, y: Any) -> None:
    """Add one to the input tensor.

    Args:
        x: Tensor
            The input tensor.
        y: Tensor
            The output tensor.
    """
    if hasattr(x, "ndim") and getattr(x, "ndim") != 1:
        raise ValueError("x must be 1D tensor")
    if hasattr(y, "ndim") and getattr(y, "ndim") != 1:
        raise ValueError("y must be 1D tensor")
    return _ffi_api.add_one(x, y)

# Add custom add_one to __all__
if "add_one" not in __all__:
    __all__.append("add_one")
