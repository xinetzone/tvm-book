import sys
import os
from pathlib import Path
from importlib import import_module


def set_tvm(tvm_root):
    tvm_root = Path(tvm_root)
    TVM_PATH = str(tvm_root/'python')
    VTA_PATH = str(tvm_root/'vta/python')
    # sys.path.extend([TVM_PATH, VTA_PATH])
    for path in [TVM_PATH, VTA_PATH]:
        if path not in sys.path:
            sys.path.extend([path])
    vta_path = str(tvm_root/"3rdparty/vta-hw")
    os.environ['VTA_HW_PATH'] = os.environ.get('VTA_HW_PATH', vta_path)


def set_mxnet():
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    os.environ['MXNET_CUDNN_LIB_CHECKING'] = '0'


def set_cudnn(cuda_path=":/usr/local/cuda/bin",
              LD_LIBRARY_PATH="/usr/local/cuda/lib64"):
    os.environ["PATH"] += cuda_path
    os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH


def import_tvm(tvm_root):
    set_tvm(tvm_root)
    # import 模块
    tvm = import_module('tvm')
    vta = import_module('vta')
    return tvm, vta
