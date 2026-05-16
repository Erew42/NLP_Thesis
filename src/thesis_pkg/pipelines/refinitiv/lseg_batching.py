from __future__ import annotations

"""Compatibility shim for reusable LSEG request batching helpers."""

import sys

from thesis_refinitiv.lseg_client import batching as _impl

_compat_name = __name__
globals().update(_impl.__dict__)
sys.modules[_compat_name] = _impl
