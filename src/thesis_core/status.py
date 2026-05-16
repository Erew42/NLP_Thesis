from __future__ import annotations

"""Deferred facade for shared pipeline status primitives.

``DataStatus`` remains in the legacy CCM transform module for compatibility
with the existing pipeline import graph. This domain module exists as the
future owner boundary and re-exports the legacy symbols without changing their
identity.
"""

from thesis_pkg.core.ccm.transforms import DataStatus, STATUS_DTYPE, _ensure_data_status, _flag_if

__all__ = ["DataStatus", "STATUS_DTYPE", "_ensure_data_status", "_flag_if"]
