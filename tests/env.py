import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[1]
sys.path.extend([
    f"{root_dir}/src",
    "/media/pc/data/lxw/ai/ultralytics",
    "/media/pc/data/lxw/caffe_src", # caffe 环境
])
# import tools.set_tensorflow
import tools.set_env
