from __future__ import annotations

from pathlib import Path


TEMPLATE_DIR = Path(__file__).resolve().parent
TABLE_WRAPPER_TEMPLATE_PATH = TEMPLATE_DIR / "latex_table_wrapper.tex"

__all__ = [
    "TABLE_WRAPPER_TEMPLATE_PATH",
    "TEMPLATE_DIR",
]
