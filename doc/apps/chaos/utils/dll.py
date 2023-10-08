import ctypes


def load_dll(lib_path="lib/libtvm_ext.so"):
    """加载库，函数将被注册到 TVM"""
    # 作为全局加载，这样全局 extern symbol 对其他 dll 是可见的。
    # curr_path = f"{ROOT}/"
    lib = ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)
    return lib
