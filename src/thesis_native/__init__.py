from __future__ import annotations

"""Neutral native-extension import boundary."""

try:
    from . import _lm2011_rust as _lm2011_rust
except Exception as exc:  # pragma: no cover - depends on optional native build
    _lm2011_rust = None
    RUST_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    RUST_IMPORT_ERROR = None

__all__ = ["_lm2011_rust", "RUST_IMPORT_ERROR"]
