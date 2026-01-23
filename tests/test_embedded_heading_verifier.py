from __future__ import annotations

from pathlib import Path

import pytest

from thesis_pkg.core.sec.suspicious_boundary_diagnostics import (
    _find_embedded_heading_hits,
    _summarize_embedded_hits,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_fixture(name: str) -> str:
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.skip(f"fixture missing: {path}")
    return path.read_text(encoding="utf-8")


def test_embedded_heading_continued_is_ignored() -> None:
    text = "\n".join(
        [
            "ITEM 1. BUSINESS",
            "Some introductory text.",
            "ITEM 1. BUSINESS - continued",
            "More detail follows.",
        ]
    )
    hits = _find_embedded_heading_hits(text, current_item_id="1", current_part="I")
    classifications = [hit.classification for hit in hits]
    assert "same_item_continuation" in classifications
    embedded_warn, embedded_fail, _first_hit, _first_flagged, _first_fail, _counts = (
        _summarize_embedded_hits(hits)
    )
    assert not embedded_warn
    assert not embedded_fail


def test_embedded_heading_toc_row_warns() -> None:
    text = "\n".join(
        [
            "ITEM 10. Directors and Executive Officers.............. 40",
            "ITEM 11. Executive Compensation........................ 50",
            "ITEM 12. Security Ownership............................ 55",
        ]
    )
    hits = _find_embedded_heading_hits(text, current_item_id="7", current_part="II")
    classifications = [hit.classification for hit in hits]
    assert "toc_row" in classifications
    embedded_warn, embedded_fail, _first_hit, _first_flagged, _first_fail, _counts = (
        _summarize_embedded_hits(hits)
    )
    assert embedded_warn
    assert not embedded_fail


def test_embedded_heading_reserved_overlap_fails() -> None:
    text = "\n".join(
        [
            "ITEM 6. [RESERVED]",
            "This line begins the section with prose-like text.",
            "More details are provided in subsequent sentences.",
        ]
    )
    hits = _find_embedded_heading_hits(text, current_item_id="5", current_part="II")
    classifications = [hit.classification for hit in hits]
    assert "true_overlap" in classifications
    _embedded_warn, embedded_fail, _first_hit, _first_flagged, _first_fail, _counts = (
        _summarize_embedded_hits(hits)
    )
    assert embedded_fail


def test_embedded_heading_cross_ref_warns() -> None:
    text = "\n".join(
        [
            "ITEM 7. See Item 7A for further discussion.",
            "Additional prose continues here to resemble body text.",
        ]
    )
    hits = _find_embedded_heading_hits(text, current_item_id="6", current_part="II")
    classifications = [hit.classification for hit in hits]
    assert "cross_ref_line" in classifications
    _embedded_warn, embedded_fail, _first_hit, _first_flagged, _first_fail, _counts = (
        _summarize_embedded_hits(hits)
    )
    assert not embedded_fail


def test_embedded_heading_toc_window_index_style() -> None:
    text = _load_fixture("toc_window_index_style.txt")
    hits = _find_embedded_heading_hits(text, current_item_id="15", current_part="IV")
    classifications = [hit.classification for hit in hits]
    assert "toc_row" in classifications
    assert "true_overlap" not in classifications


def test_embedded_heading_toc_start_misfire() -> None:
    text = _load_fixture("toc_start_misfire_sample.txt")
    hits = _find_embedded_heading_hits(text, current_item_id="15", current_part="IV")
    classifications = [hit.classification for hit in hits]
    assert "toc_start_misfire" in classifications
    _embedded_warn, embedded_fail, _first_hit, _first_flagged, _first_fail, _counts = (
        _summarize_embedded_hits(hits)
    )
    assert embedded_fail
