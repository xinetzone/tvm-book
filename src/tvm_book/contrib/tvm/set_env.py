# 加载自定义库
from platform import python_version
import sys
import os
from pathlib import Path
from importlib import import_module

# ROOT = Path(__file__).absolute().resolve()
# TVM_ROOT = str(ROOT.parents[2])
# MOD_PATH = str(ROOT.parents[1]/'src') #str(TVM_ROOT/'xinetzone/src')
_python_version = python_version()

if _python_version == '3.8.10':
    TVM_ROOT = "/media/pc/data/4tb/lxw/tvm"
else:
    TVM_ROOT = "/media/pc/data/4tb/lxw/books/tvm"
    # TVM_ROOT = "/media/pc/data/4tb/lxw/books/tvm"
MOD_PATH = '/media/pc/data/4tb/lxw/books/tvm/xinetzone/src'
os.environ['VTA_HW_PATH'] = os.environ.get('VTA_HW_PATH', f'{TVM_ROOT}/3rdparty/vta-hw')

if MOD_PATH not in sys.path:
    sys.path.extend([MOD_PATH])

tvmx = import_module('tvmx')
tvmx.set_tvm(TVM_ROOT)
import tvm

print(f'Python: {_python_version}')
print(f'TVM: {tvm.__version__}')
# tvm, vta = tvmx.import_tvm(TVM_ROOT)
# print("TVM 根目录：", TVM_ROOT)