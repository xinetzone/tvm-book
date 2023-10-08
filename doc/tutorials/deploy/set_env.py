from pathlib import Path
from tvm_book.config.env import set_tvm
TVM_ROOT = Path(__file__).absolute().parents[4]
print(TVM_ROOT)
# TVM_ROOT = "/media/pc/data/lxw/ai/tvm/"
# print(TVM_ROOT)
set_tvm(TVM_ROOT)