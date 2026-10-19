from __future__ import annotations

import sys
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = TESTS_ROOT.parent
SRC_ROOT = WORKSPACE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

