"""tvm_book 包的公共 Python API"""
from typing import Any
from . import _ffi_api
from .base import _LIB

def add_one(x: Any, y: Any) -> None:
    """将输入张量加一

    Args:
        x : 输入张量
        y : 输出张量
    """
    return  _LIB.add_one(x, y)


def raise_error(msg: str) -> None:
    """使用给定消息抛出错误

    Args:
        msg : 要抛出错误的消息

    Raises:
        RuntimeError: 函数抛出的错误
    """
    return _ffi_api.raise_error(msg)
