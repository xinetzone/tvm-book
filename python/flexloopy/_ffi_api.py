from typing import TYPE_CHECKING

import tvm_ffi

from .base import get_lib

get_lib()

tvm_ffi.init_ffi_api("flexloopy", __name__)


# tvm-ffi-stubgen(begin): global/flexloopy
if TYPE_CHECKING:
    # fmt: off
    def raise_error(_0: str, /) -> None: ...
    # fmt: on
# tvm-ffi-stubgen(end)
