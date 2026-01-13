"""Utilities to locate and load the flexloopy shared library."""
from typing import Optional
import tvm_ffi
from ._ffi_api import LIB


# Maintain compatibility with existing code that uses _LIB from base
_LIB = LIB


def get_lib() -> tvm_ffi.Module:
    """Get the loaded flexloopy library module.
    
    Returns:
        The loaded library module.
    """
    return _LIB


__all__ = ["get_lib", "_LIB"]
