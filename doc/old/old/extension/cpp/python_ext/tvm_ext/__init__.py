import tvm
from . import _ffi_api

def callhello(f):
    return _ffi_api.callhello(f)
