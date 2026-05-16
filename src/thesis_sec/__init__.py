from __future__ import annotations

"""Repo-internal boundary for SEC filing text and item extraction code.

SEC extraction implementation modules remain under ``thesis_pkg.core.sec`` for
this migration-readiness pass. The boundary is intentionally deferred because
SEC extraction still includes Python/Cython-backed item-boundary logic and
legacy diagnostics that need a separate extraction-focused migration.
"""

__all__ = ["PACKAGE_BOUNDARY"]

PACKAGE_BOUNDARY = "SEC filing text extraction"
