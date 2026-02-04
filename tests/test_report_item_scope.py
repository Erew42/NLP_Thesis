from __future__ import annotations

from pathlib import Path

from thesis_pkg.core.sec.suspicious_boundary_diagnostics import (
    _build_diagnostics_report,
    _update_item_breakdown,
)


def test_report_scope_target_excludes_non_target_item() -> None:
    item_breakdown: dict[tuple[str, str], object] = {}

    _update_item_breakdown(
        item_breakdown,
        form_type="10-K",
        item_part="I",
        item_id="1A",
        flags=[],
        item_warn=False,
        item_fail=False,
        internal_heading_leak=False,
        embedded_hits=[],
        embedded_first_flagged=None,
        truncated_successor=False,
        truncated_part=False,
        item_missing_part=False,
        item_full_text="X",
    )
    _update_item_breakdown(
        item_breakdown,
        form_type="10-K",
        item_part="IV",
        item_id="15",
        flags=[],
        item_warn=False,
        item_fail=False,
        internal_heading_leak=False,
        embedded_hits=[],
        embedded_first_flagged=None,
        truncated_successor=False,
        truncated_part=False,
        item_missing_part=False,
        item_full_text="X",
    )

    report = _build_diagnostics_report(
        rows=[],
        total_filings=0,
        total_items=0,
        total_part_only_prefix=0,
        start_candidates_total=0,
        start_candidates_toc_rejected_total=0,
        start_selection_unverified_total=0,
        truncated_successor_total=0,
        truncated_part_total=0,
        parquet_dir=Path("."),
        max_examples=1,
        provenance={},
        item_breakdown=item_breakdown,
        focus_items=None,
        report_item_scope="target",
        target_set="cohen2020_all_items",
    )

    assert "Per-item breakdown (Target set: cohen2020_all_items)" in report
    assert "I:1A" in report
    assert "IV:15" not in report
