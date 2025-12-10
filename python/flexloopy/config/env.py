"""环境配置模块

该模块用于在本地或CI环境中统一设置与 TVM、VTA 以及第三方工具链相关的 Python 路径与环境变量，确保调用端能够顺利 `import` 对应包并运行示例或训练/编译流程。

使用说明：

- 在项目启动脚本或交互式会话中调用 ``set_tvm`` 指定 TVM 源码根目录；必要时调用 ``set_caffeproto`` 指定 CaffeProto 根目录。
- 本模块仅添加路径和环境变量，不会改变 TVM/VTA 的功能实现。

示例：

    >>> from flexloopy.config.env import set_tvm, set_caffeproto
    >>> set_tvm("/path/to/tvm")
    >>> set_caffeproto("/path/to/CaffeProto")

.. tip::
    路径分隔符在不同操作系统上不同，Windows 为 ``;``，Unix/Linux 为 ``:``。

    本模块对 ``sys.path`` 的处理跨平台，`PYTHONPATH` 的格式请按实际平台调整。
"""
import sys
import os
from pathlib import Path


def set_tvm(tvm_root: str|Path) -> None:
    """配置 TVM/VTA 的 Python 依赖路径与环境变量

    Args:
        tvm_root (str | Path): TVM 源码工程的根目录路径，可以是 `str` 或 `pathlib.Path`。应指向包含 `python/` 与 `vta/python/` 子目录的仓库根。

    Returns:
        None: 函数会原地更新 `sys.path` 和部分 `os.environ` 项以便后续可直接 `import tvm` 与 `import vta`。

    使用示例：
        >>> from flexloopy.config.env import set_tvm
        >>> set_tvm("D:/dev/tvm")
        >>> import tvm, vta  # 成功导入即表示环境设置生效
    """
    tvm_root = Path(tvm_root)
    TVM_PATH = str(tvm_root/'python')  # TVM 的 Python 包所在目录，例如 <tvm_root>/python
    VTA_PATH = str(tvm_root/'vta/python')  # VTA 的 Python 包所在目录，例如 <tvm_root>/vta/python
    # sys.path.extend([TVM_PATH, VTA_PATH])
    for path in [TVM_PATH, VTA_PATH]:
        if path not in sys.path:
            sys.path.extend([path])
    # os.environ['TVM_HOME'] = str(tvm_root)
    os.environ['PYTHONPATH'] = f"{TVM_PATH}:{VTA_PATH}" + ":${PYTHONPATH}"  # 追加到 PYTHONPATH；Windows 建议改用分号 `;` 分隔

def set_caffeproto(caffeproto_root: str|Path="../../../tests/caffeproto") -> None:
    """配置 CaffeProto 的 Python 依赖路径(仅用于测试，暂未在项目中使用)

    Args:
        caffeproto_root (str | Path, optional): CaffeProto 工程根目录路径，默认相对路径 `../../../tests/caffeproto`。应包含 `python/` 子目录。

    Returns:
        None: 函数会在 `sys.path` 中追加 `caffeproto_root/python`，避免重复添加。

    使用示例：
        >>> from flexloopy.config.env import set_caffeproto
        >>> set_caffeproto("D:/dev/CaffeProto")
        >>> import caffe  # 如项目提供对应 Python 包
    """
    caffeproto_root = Path(caffeproto_root)
    CaffeProto_PATH = str(caffeproto_root/'python')  # CaffeProto 的 Python 包目录
    if CaffeProto_PATH not in sys.path:
        sys.path.extend([CaffeProto_PATH])

# 以下为可选的环境示例（默认注释），根据需要开启：
# def set_mxnet():
#     # 关闭 MXNet 的部分自动调优与库检查以规避某些平台兼容性问题
#     os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'  # 取值范围：'0' 或 '1'
#     os.environ['MXNET_CUDNN_LIB_CHECKING'] = '0'     # 取值范围：'0' 或 '1'


# def set_cudnn(cuda_path:str|Path="/usr/local/cuda/bin",
#               LD_LIBRARY_PATH:str|Path="/usr/local/cuda/lib64"):
#     # 为运行时加入 CUDA 的可执行与动态库路径；Unix 系统示例
#     os.environ["PATH"] += f":{cuda_path}"
#     os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH


# def import_tvm(tvm_root):
#     # 便捷导入示例：设置路径后导入 tvm/vta 并返回模块对象
#     from importlib import import_module
#     set_tvm(tvm_root)
#     tvm = import_module('tvm')
#     vta = import_module('vta')
#     return tvm, vta
