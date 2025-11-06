from typing import TYPE_CHECKING

import tvm_ffi

# 确保库（lib）先被加载
from .base import _LIB  # noqa: F401

# 这是用于注册所有全局函数的捷径
tvm_ffi.init_ffi_api("tvm_book", __name__)


# tvm-ffi-stubgen(begin): global/tvm_book
if TYPE_CHECKING:
    # fmt: off
    def raise_error(_0: str, /) -> None: ...
    # fmt: on
# tvm-ffi-stubgen(end)
