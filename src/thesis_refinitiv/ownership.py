from __future__ import annotations

"""Deferred facade for Refinitiv ownership pipeline entrypoints.

The handoff builders are already owned by ``thesis_refinitiv.bridge``. The
remaining ownership module surface still combines bridge entrypoints with
legacy LSEG ownership API runners, so the facade stays deferred until the
ownership API runner boundary is moved or explicitly split.
"""

from thesis_pkg.pipelines.refinitiv.ownership import *  # noqa: F401,F403
