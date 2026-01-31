from __future__ import annotations

from thesis_pkg.core.sec.filing_text import _prefix_looks_like_cross_ref
from thesis_pkg.core.sec.suspicious_boundary_diagnostics import (
    _find_internal_heading_leak,
    _is_midline_heading_prefix,
    _is_10k,
    _prefix_kind,
)


def test_part_prefix_not_cross_ref_or_midline() -> None:
    prefix = "PART I "
    assert not _prefix_looks_like_cross_ref(prefix)
    assert _prefix_kind(prefix) == "part_only"
    assert not _is_midline_heading_prefix(prefix, len(prefix))


def test_cross_ref_prefixes() -> None:
    assert _prefix_looks_like_cross_ref("See ")
    assert _prefix_looks_like_cross_ref("As discussed in ")
    assert _prefix_looks_like_cross_ref("In Part II, ")


def test_midline_punctuated_clause() -> None:
    prefix = "\u2026; "
    assert _is_midline_heading_prefix(prefix, len(prefix))


def test_internal_heading_leak_detects_next_heading() -> None:
    text = ("A" * 210) + "\nITEM 2. Properties\nMore text."
    leak = _find_internal_heading_leak(text)
    assert leak is not None
    assert leak.match_text.upper().startswith("ITEM 2")


def test_internal_heading_leak_ignores_prose_cross_ref() -> None:
    text = ("A" * 210) + "\nWe refer to Item 7 for details.\nMore text."
    leak = _find_internal_heading_leak(text)
    assert leak is None


def test_is_10k_excludes_amendments() -> None:
    assert _is_10k("10-K")
    assert _is_10k("10KSB")
    assert not _is_10k("10-K/A")
    assert not _is_10k("10-K405/A")
    assert not _is_10k("10KSB-A")
