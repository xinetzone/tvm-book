import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
# print(ROOT)
sys.path.extend([
    f"{ROOT}/tests"
])
import env # 配置 TVM 环境
