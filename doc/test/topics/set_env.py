import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
# print(f"ROOT: {ROOT}")
sys.path.extend([
    f"{ROOT}/tests", f"{ROOT}/src", "/media/pc/data/lxw/ai/ultralytics",
    "/media/pc/data/lxw/caffe_src", # caffe 环境
])
# # # from tools.tag_span import _create_span, _set_span, _verify_structural_equal_with_span
# from tools.torch_utils import verify_model
import tools.set_env