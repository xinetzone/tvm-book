"""libtvm_ext.so 库信息"""
import sys, os
import ctypes
from tvm._ffi.libinfo import find_lib_path

def _load_lib(name: str|list[str]|None=None, search_path: str|list[str]|None=None):
    """通过搜索可能的路径加载库
    
    Arg:
        name: 需要导入的库名称（比如 ``libtvm_ext.so``），可为空（则为 ``libtvm.so``），也可为名称列表（暂未实现）
        search_path: 搜索路径，可为空，也可为名称列表
    """
    lib_path = find_lib_path(name=name, search_path=search_path)
    # 在 Python 3.8 之后，需要在 Windows 中显式添加 dll 搜索路径
    if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    if hasattr(lib, "TVMGetLastError"):
        lib.TVMGetLastError.restype = ctypes.c_char_p
    return lib, os.path.basename(lib_path[0])
