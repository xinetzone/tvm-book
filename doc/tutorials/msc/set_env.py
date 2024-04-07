from pathlib import Path
from tvm_book.config.env import set_tvm
# TVM_ROOT = Path(__file__).resolve().parents[3]
# print(TVM_ROOT)
TVM_ROOT = "/media/pc/data/lxw/ai/tvm/"
set_tvm(TVM_ROOT)
# 添加工具链路径
import sys
ROOT = Path("/media/pc/data/board/arria10/lxw/tasks/tools/npu_user_demos") # 工具链根目录
sys.path.extend([str(ROOT), str(ROOT/"tools_python")])
# caffe 环境
sys.path.extend(["/media/pc/data/libs/temp/caffe/python"])
