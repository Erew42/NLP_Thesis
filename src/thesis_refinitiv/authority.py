from __future__ import annotations

"""Deferred facade for Refinitiv ownership authority logic.

The implementation still lives at
``thesis_pkg.pipelines.refinitiv.authority``. Moving this module into primary
``thesis_refinitiv`` ownership is intentionally deferred because the authority
surface participates in eager legacy package exports and downstream
doc-ownership assembly; changing its module identity safely requires a broader
lazy-export pass.
"""

from thesis_pkg.pipelines.refinitiv.authority import *  # noqa: F401,F403
