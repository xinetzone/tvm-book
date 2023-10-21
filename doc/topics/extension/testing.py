from . import _ffi_api

def sym_add(a, b):
    """Transform division by a constant to multiplication by the inverse of the constant"""
    return _ffi_api.sym_add(a, b)