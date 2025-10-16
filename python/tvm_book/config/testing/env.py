"""环境配置"""
import sys
import os
from pathlib import Path


def set_tvm(tvm_root: str|Path):
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

def set_caffeproto(caffeproto_root: str|Path="../../../tests/caffeproto"):
    """配置 CaffeProto 环境

    Args:
        caffeproto_root: CaffeProto 项目所在根目录
    """
    caffeproto_root = Path(caffeproto_root)
    CaffeProto_PATH = str(caffeproto_root/'python')
    if CaffeProto_PATH not in sys.path:
        sys.path.extend([CaffeProto_PATH])
