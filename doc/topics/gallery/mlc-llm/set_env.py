import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[4]
sys.path.extend([
    f"{root_dir}/tests"
])
from env import temp_dir
