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
    vta_path = str(tvm_root/"3rdparty/vta-hw")
    os.environ['VTA_HW_PATH'] = os.environ.get('VTA_HW_PATH', vta_path)
    os.environ['TVM_HOME'] = str(tvm_root)
    os.environ['PYTHONPATH'] = f"{TVM_PATH}:{VTA_PATH}" + ":${PYTHONPATH}"


# def set_mxnet():
#     os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
#     os.environ['MXNET_CUDNN_LIB_CHECKING'] = '0'


def set_cudnn(cuda_path:str|Path="/usr/local/cuda/bin",
              LD_LIBRARY_PATH:str|Path="/usr/local/cuda/lib64"):
    os.environ["PATH"] += f":{cuda_path}"
    os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH


# def import_tvm(tvm_root):
#     from importlib import import_module
#     set_tvm(tvm_root)
#     # import 模块
#     tvm = import_module('tvm')
#     vta = import_module('vta')
#     return tvm, vta
