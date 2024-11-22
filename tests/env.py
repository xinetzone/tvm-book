import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([
    f"{ROOT}/src"
])
import tools.set_env
