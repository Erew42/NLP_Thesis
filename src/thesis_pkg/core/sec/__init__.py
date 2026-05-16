"""SEC text parsing and extraction core.

The Rust extension is owned by the neutral :mod:`thesis_native` boundary.
This module re-exports it only as a compatibility shim for legacy imports.
"""

from __future__ import annotations

import sys

from thesis_native import _lm2011_rust

sys.modules[f"{__name__}._lm2011_rust"] = _lm2011_rust

__all__ = ["_lm2011_rust"]
