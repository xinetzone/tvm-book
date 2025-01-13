"""TVM 插件测试demo"""
import tvm
from tvm import te
from . import _ffi_api

def sym_add(a: tvm.tir.expr.Var, b: tvm.tir.expr.Var):
    """符号加法

    Args:
        a: 符号张量
        b: 符号张量

    Example:
        >>> from tvm import te
        >>> a = te.var("x")
        >>> b = te.var("y")
        >>> c = sym_add(a, b)
        >>> assert c.a == a and c.b == b
        >>> print(c)
        (x: int32 + y: int32)
    """
    return _ffi_api.sym_add(a, b)

def bind_add(f, b):
    """加法偏函数

    Args:
        f: 加法函数
        b: 偏移量

    Returns:
        加法偏函数

    Example:
        >>> def add(a, b):
        ...     return a + b

        >>> f = testing.bind_add(add, 7)
        >>> assert f(2) == 9
    """
    print(type(a), type(b))
    return _ffi_api.bind_add(a, b)

def add_one(A):
    # A = te.placeholder((n,), name="A")
    return te.compute(
        A.shape, lambda *i: tvm.tir.call_extern("float32", "TVMTestAddOne", A(*i)), name="B"
    )
