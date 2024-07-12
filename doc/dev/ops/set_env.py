import sys
from pathlib import Path
ROOT = Path(".").resolve().parents[2]
sys.path.extend([f"{ROOT}/tests", f"{ROOT}/src"])
print(f"ROOT: {ROOT}")
# # from tools.tag_span import _create_span, _set_span, _verify_structural_equal_with_span
from tools.torch_utils import verify_model
