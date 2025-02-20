# TVM FFI

{mod}`tvm._ffi` 是 TVM 中与 C 代码交互的命名空间。它包含了所有与 C 代码交互的内容。TVM 中的大多数 C 相关对象都是 {mod}`ctypes` 兼容的，这意味着它们包含名为 `handle` 的字段，该字段是 {data}`ctypes.c_void_p`，并且可以通过 {mod}`ctypes` 的函数调用来使用 。

一些性能关键函数由 Cython 实现，并具有 {mod}`ctypes` 回退实现。

```{toctree}
:hidden:

libinfo
datastruct
register-func
funcs
_init_api
register-object
```
