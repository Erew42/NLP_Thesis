from __future__ import annotations

from thesis_pkg.core.sec.extraction_utils import EmbeddedHeadingHit
from thesis_pkg.core.sec.suspicious_boundary_diagnostics import (
    _build_item_breakdown_rows,
    _update_item_breakdown,
)


def test_item_breakdown_sorting_and_drivers() -> None:
    item_breakdown: dict[tuple[str, str], object] = {}

    fail_hit = EmbeddedHeadingHit(
        kind="item",
        classification="true_overlap",
        item_id="15",
        part=None,
        line_idx=10,
        char_pos=50,
        full_text_len=200,
        snippet="overlap",
    )

    _update_item_breakdown(
        item_breakdown,
        form_type="10-K",
        item_part=None,
        item_id="15",
        flags=["embedded_heading_fail"],
        item_warn=False,
        item_fail=True,
        internal_heading_leak=False,
        embedded_hits=[fail_hit],
        embedded_first_flagged=fail_hit,
        truncated_successor=False,
        truncated_part=False,
        item_missing_part=False,
        item_full_text="X" * 200,
    )

    _update_item_breakdown(
        item_breakdown,
        form_type="10-K",
        item_part=None,
        item_id="15",
        flags=["toc_like_heading"],
        item_warn=True,
        item_fail=False,
        internal_heading_leak=False,
        embedded_hits=[],
        embedded_first_flagged=None,
        truncated_successor=False,
        truncated_part=False,
        item_missing_part=False,
        item_full_text="X" * 200,
    )

    _update_item_breakdown(
        item_breakdown,
        form_type="10-K",
        item_part=None,
        item_id="7",
        flags=["midline_heading"],
        item_warn=True,
        item_fail=False,
        internal_heading_leak=False,
        embedded_hits=[],
        embedded_first_flagged=None,
        truncated_successor=False,
        truncated_part=False,
        item_missing_part=False,
        item_full_text="X" * 200,
    )

    _update_item_breakdown(
        item_breakdown,
        form_type="10-K",
        item_part=None,
        item_id="7",
        flags=["midline_heading"],
        item_warn=True,
        item_fail=False,
        internal_heading_leak=False,
        embedded_hits=[],
        embedded_first_flagged=None,
        truncated_successor=False,
        truncated_part=False,
        item_missing_part=False,
        item_full_text="X" * 200,
    )

    rows_by_form = _build_item_breakdown_rows(item_breakdown, focus_items=None)
    rows = rows_by_form["10-K"]

    assert any(row["item_key"] == "15" for row in rows)
    assert rows[0]["item_key"] == "15"
    assert "true_overlap" in rows[0]["top_fail_drivers"]
