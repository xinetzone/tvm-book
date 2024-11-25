# SPDX-FileCopyrightText: 2022-present liuxinwei <liuxinwei@xmsilicon.cn>
#
# SPDX-License-Identifier: MIT
from pathlib import Path
import os

ROOT = Path(__file__).parents[1]
VTA_HW_PATH = os.getenv("VTA_HW_PATH", str(ROOT/"vta-hw"))