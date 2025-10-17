from typing import TYPE_CHECKING

import tvm_ffi

# make sure lib is loaded first
from .base import _LIB  # noqa: F401

# this is a short cut to register all the global functions
# prefixed by `my_ffi_extension.` to this module
tvm_ffi.init_ffi_api("my_ffi_extension", __name__)


# tvm-ffi-stubgen(begin): global/my_ffi_extension
if TYPE_CHECKING:
    # fmt: off
    def raise_error(_0: str, /) -> None: ...
    # fmt: on
# tvm-ffi-stubgen(end)
