from __future__ import annotations

from typing import TYPE_CHECKING

import tvm_ffi

from .base import _LIB, _LOAD_ERROR


def _missing_extension() -> None:
    hint = (
        "The native tvm_book extension is unavailable, so FFI bindings cannot be initialised. "
        "Install a compatible compiler toolchain and reinstall tvm-book."
    )
    raise RuntimeError(hint) from _LOAD_ERROR


if _LIB is not None:
    # this is a short cut to register all the global functions
    # prefixed by `tvm_book.` to this module
    tvm_ffi.init_ffi_api("tvm_book", __name__)
else:  # pragma: no cover - triggered on systems without a compiler toolchain
    def raise_error(_0: str, /) -> None:
        _missing_extension()


# tvm-ffi-stubgen(begin): global/tvm_book
if TYPE_CHECKING:
    # fmt: off
    def raise_error(_0: str, /) -> None: ...
    # fmt: on
# tvm-ffi-stubgen(end)
