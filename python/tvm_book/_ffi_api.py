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
    # 使用模块级 __getattr__ 懒处理缺失的 FFI 函数，避免为每个函数显式定义
    # 这样可避免文档生成时重复收集定义，同时在运行时给出清晰错误提示
    def __getattr__(name: str):  # type: ignore[override]
        _missing_extension()


# tvm-ffi-stubgen(begin): global/tvm_book
if TYPE_CHECKING:
    # 使用类型标注声明以避免在文档生成中出现重复的函数定义
    # 仅在类型检查阶段提供函数签名信息，避免 AutoAPI 收集两次
    from typing import Callable
    raise_error: Callable[[str], None]
# tvm-ffi-stubgen(end)
