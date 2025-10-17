"""环境配置工具

该模块提供了各种环境配置功能，帮助设置 TVM 和其他相关库的运行环境。
"""

import sys
import os
from pathlib import Path
from typing import Union


def set_tvm(tvm_root: Union[str, Path]) -> None:
    """配置 TVM 环境

    Args:
        tvm_root: TVM 项目所在根目录
    """
    tvm_root = Path(tvm_root)
    TVM_PATH = str(tvm_root/'python')
    VTA_PATH = str(tvm_root/'vta/python')
    # sys.path.extend([TVM_PATH, VTA_PATH])
    for path in [TVM_PATH, VTA_PATH]:
        if path not in sys.path:
            sys.path.extend([path])
    # os.environ['TVM_HOME'] = str(tvm_root)
    os.environ['PYTHONPATH'] = f"{TVM_PATH}:{VTA_PATH}" + ":${PYTHONPATH}"


def set_caffeproto(caffeproto_root: Union[str, Path] = "../../../tests/caffeproto") -> None:
    """配置 CaffeProto 环境

    Args:
        caffeproto_root: CaffeProto 项目所在根目录
    """
    caffeproto_root = Path(caffeproto_root)
    CaffeProto_PATH = str(caffeproto_root/'python')
    if CaffeProto_PATH not in sys.path:
        sys.path.extend([CaffeProto_PATH])
