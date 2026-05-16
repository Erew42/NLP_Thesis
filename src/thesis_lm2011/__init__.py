from __future__ import annotations

"""Repo-internal boundary for LM2011 replication and extension logic.

LM2011 pipeline modules remain under ``thesis_pkg`` for this pass. The boundary
is intentionally deferred until the post-Refinitiv runner and extension
pipeline import graph can be moved as a unit without changing public runner
behavior.
"""

__all__ = ["PACKAGE_BOUNDARY"]

PACKAGE_BOUNDARY = "LM2011 replication and extension"
