import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
print(ROOT)
sys.path.extend([f"{ROOT}/tests", f"{ROOT}/src"])
# # from tools.tag_span import _create_span, _set_span, _verify_structural_equal_with_span
from tools.torch_utils import verify_model