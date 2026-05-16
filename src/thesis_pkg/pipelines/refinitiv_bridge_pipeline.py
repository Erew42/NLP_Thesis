from __future__ import annotations

"""Compatibility shim for the Refinitiv bridge implementation."""

import sys

from thesis_refinitiv import bridge as _impl

_compat_name = __name__
globals().update(_impl.__dict__)
sys.modules[_compat_name] = _impl
