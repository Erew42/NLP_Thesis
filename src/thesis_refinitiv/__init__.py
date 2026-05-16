from __future__ import annotations

"""Repo-internal boundary for thesis-specific Refinitiv pipeline logic.

Primary owners in this pass:

- ``thesis_refinitiv.bridge``
- ``thesis_refinitiv.lseg_client``

Deferred facades remain for authority and ownership until the coupled legacy
runner and doc-ownership import graph can be split without changing public
behavior.
"""

__all__ = ["DEFERRED_FACADES", "PACKAGE_BOUNDARY", "PRIMARY_BOUNDARIES"]

PACKAGE_BOUNDARY = "thesis-specific Refinitiv logic"
PRIMARY_BOUNDARIES: tuple[str, ...] = ("bridge", "lseg_client")
DEFERRED_FACADES: tuple[str, ...] = ("authority", "ownership")
