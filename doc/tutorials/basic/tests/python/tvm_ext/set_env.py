from pathlib import Path
import ctypes

def load_lib():
    """加载库，函数将被注册到 TVM"""
    curr_dir = Path(__file__).resolve().parents[2]
    # 作为全局加载，这样全局 extern symbol 对其他 dll 是可见的。
    curr_path = str(curr_dir/"lib/libtvm_ext.so")
    lib = ctypes.CDLL(curr_path, ctypes.RTLD_GLOBAL)
    return lib


_LIB = load_lib()
