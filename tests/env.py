import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([
    f"{ROOT}/src",
    "/media/pc/data/lxw/ai/ultralytics",
    "/media/pc/data/lxw/caffe_src", # caffe 环境
])
# import tools.set_tensorflow
import tools.set_env
