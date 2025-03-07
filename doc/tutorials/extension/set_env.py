import sys
from pathlib import Path
ROOT = Path("__file__").resolve().parents[3]
sys.path.extend([
    f"{ROOT}/tests"
])
import env
