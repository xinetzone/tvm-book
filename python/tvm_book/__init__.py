"""Public Python API for the tvm_book package."""

from __future__ import annotations

from typing import Any

from . import _ffi_api
from .base import _LIB, _LOAD_ERROR


def _require_native() -> Any:
    """Return the loaded native module or raise a helpful error."""
    if _LIB is None:
        hint = (
            "The native tvm_book extension is not available. "
            "Install the required C/C++ toolchain (e.g. Visual Studio Build Tools on Windows) "
            "and reinstall tvm-book to build the extension."
        )
        raise RuntimeError(hint) from _LOAD_ERROR
    return _LIB


def add_one(x: Any, y: Any) -> None:
    """Add one to the input tensor.

    Parameters
    ----------
    x : Tensor
      The input tensor.
    y : Tensor
      The output tensor.
    """
    return _require_native().add_one(x, y)


def raise_error(msg: str) -> None:
    """Raise an error with the given message.

    Parameters
    ----------
    msg : str
        The message to raise the error with.

    Raises
    ------
    RuntimeError
        The error raised by the function.
    """
    return _ffi_api.raise_error(msg)
