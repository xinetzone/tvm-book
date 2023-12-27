from pathlib import Path
from tvm_book.config.env import set_tvm
TVM_ROOT = Path(__file__).resolve().parents[3]
# print(TVM_ROOT)
# TVM_ROOT = "/media/pc/data/lxw/ai/tvm/"
set_tvm(TVM_ROOT)