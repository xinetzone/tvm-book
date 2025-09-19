from pathlib import Path
from . import set_pytorch
from tvm_book.config.env import set_tvm
ROOT = Path(__file__).absolute().parents[2]
TVM_ROOT = "/media/pc/data/lxw/ai/tvm"
# print(ROOT)
set_tvm(TVM_ROOT)
temp_dir = ROOT/"tests/.temp"
temp_dir.mkdir(exist_ok=True)
import sys
sys.path.extend([
    "/media/pc/data/lxw/ai/tasks/mlc-llm/python"
])