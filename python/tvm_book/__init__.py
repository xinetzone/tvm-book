"""Public Python API for the tvm_book package."""
from typing import Any
import tvm_ffi
from . import _ffi_api
from .base import get_lib

@tvm_ffi.register_object("tvm_book.IntPair")
class IntPair(tvm_ffi.Object):
    """IntPair object."""

    def __init__(self, a: int, b: int) -> None:
        self.__ffi_init__(a, b)

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
    return get_lib().add_one(x, y)


def raise_error(msg: str) -> None:
    """Raise an error with the given message.

    Args:
        msg: The message to raise the error with.

    Raises:
        The error raised by the function.
    """
    return _ffi_api.raise_error(msg)
