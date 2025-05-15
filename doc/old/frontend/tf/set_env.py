import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.extend([f"{ROOT}/tests", f"{ROOT}/src"])
import tools.set_env # 设置 TVM 环境
