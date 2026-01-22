from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from thesis_pkg.core.sec.suspicious_boundary_diagnostics import main  # noqa: E402


if __name__ == "__main__":
    main(["regress", *sys.argv[1:]])
