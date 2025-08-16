from pathlib import Path
from tvm_book.config import env
root_dir = Path(env.__file__).resolve().parents[3]
env.set_caffeproto(root_dir/"tests/caffrproto")  # 用于 caffeproto 解析
# TVM_ROOT = Path(__file__).absolute().parents[5]
TVM_ROOT = "/media/pc/data/board/arria10/lxw/tasks/tvm-test"
# print(TVM_ROOT)
env.set_tvm(TVM_ROOT)
# import sys
# sys.path.extend([
#     "/media/pc/data/lxw/caffe_src", 
# ])