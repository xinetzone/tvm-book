import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[4]
sys.path.extend([
    f"{root_dir}/tests"
])
# print(root_dir)
from utils.set_env import temp_dir
