from __future__ import annotations

"""Reusable LSEG client boundary."""

GENERIC_LSEG_CLIENT_MODULES: tuple[str, ...] = (
    "api_common",
    "batching",
    "ledger",
    "provider",
    "stage_audit",
)

__all__ = ["GENERIC_LSEG_CLIENT_MODULES"]
