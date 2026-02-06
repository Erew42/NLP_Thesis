from __future__ import annotations

from thesis_pkg.core.sec.extraction import extract_filing_items
from thesis_pkg.core.sec.suspicious_boundary_diagnostics import DiagnosticsRow


def _parts_by_item_id(items: list[dict[str, object]]) -> dict[str, list[str | None]]:
    parts: dict[str, list[str | None]] = {}
    for item in items:
        item_id = str(item.get("item_id") or "")
        if not item_id:
            continue
        parts.setdefault(item_id, []).append(item.get("item_part"))
    return parts


def test_v2_10q_midline_form_header_part_i() -> None:
    text = "\n".join(
        [
            "FORM 10-Q QUARTERLY REPORT PART I FINANCIAL INFORMATION",
            "ITEM 1. FINANCIAL STATEMENTS",
            "Some text here.",
        ]
    )
    items = extract_filing_items(
        text,
        form_type="10-Q",
        regime=False,
        repair_boundaries=False,
        extraction_regime="v2",
    )
    parts = _parts_by_item_id(items)
    assert parts["1"] == ["I"]


def test_v2_10q_part_marker_monotonic_after_part_ii() -> None:
    text = "\n".join(
        [
            "PART I FINANCIAL INFORMATION",
            "ITEM 1. FINANCIAL STATEMENTS",
            "Some text here.",
            "PART II OTHER INFORMATION",
            "ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
            "More text here.",
            "FORM 10-Q QUARTERLY REPORT PART I FINANCIAL INFORMATION",
            "ITEM 4. CONTROLS AND PROCEDURES",
            "ITEM 5. OTHER INFORMATION",
            "ITEM 6. EXHIBITS",
        ]
    )
    items = extract_filing_items(
        text,
        form_type="10-Q",
        regime=False,
        repair_boundaries=False,
        extraction_regime="v2",
    )
    parts = _parts_by_item_id(items)
    for item_id in ("3", "4", "5", "6"):
        assert parts[item_id] == ["II"]


def test_v2_10q_title_inference_without_part_markers() -> None:
    text = "\n".join(
        [
            "ITEM 1. LEGAL PROCEEDINGS",
            "Some text here.",
            "ITEM 2. UNREGISTERED SALES OF EQUITY SECURITIES AND USE OF PROCEEDS",
            "More text here.",
            "ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS OF RESULTS OF OPERATIONS",
            "Even more text here.",
        ]
    )
    items = extract_filing_items(
        text,
        form_type="10-Q",
        regime=False,
        repair_boundaries=False,
        extraction_regime="v2",
    )
    parts = _parts_by_item_id(items)
    assert parts["1"] == ["II"]
    assert set(parts["2"]) == {"I", "II"}


def _make_diag_row(item_missing_part: bool) -> DiagnosticsRow:
    return DiagnosticsRow(
        doc_id="doc-1",
        cik="0000000000",
        accession="0000000000-00-000000",
        form_type="10-Q",
        filing_date="2020-01-01",
        period_end="2019-12-31",
        item_part="",
        item_id="1",
        item="?:1",
        item_missing_part=item_missing_part,
        canonical_item="",
        exists_by_regime=None,
        item_status="",
        counts_for_target=False,
        filing_exclusion_reason="",
        gij_omitted_items="",
        heading_line="ITEM 1. FINANCIAL STATEMENTS",
        heading_index=None,
        heading_offset=None,
        prefix_text="",
        prefix_kind="blank",
        is_part_only_prefix=False,
        is_crossref_prefix=False,
        prev_line="",
        next_line="",
        flags=(),
        embedded_heading_warn=False,
        embedded_heading_fail=False,
        first_hit_kind="",
        first_hit_classification="",
        first_hit_item_id="",
        first_hit_part="",
        first_hit_line_idx=None,
        first_hit_char_pos=None,
        first_hit_snippet="",
        first_fail_kind="",
        first_fail_classification="",
        first_fail_item_id="",
        first_fail_part="",
        first_fail_line_idx=None,
        first_fail_char_pos=None,
        first_fail_snippet="",
        first_embedded_kind="",
        first_embedded_classification="",
        first_embedded_class="",
        first_embedded_item_id="",
        first_embedded_part="",
        first_embedded_line_idx=None,
        first_embedded_char_pos=None,
        first_embedded_snippet="",
        heading_line_raw="ITEM 1. FINANCIAL STATEMENTS",
        internal_heading_leak=False,
        leak_pos="",
        leak_match="",
        leak_context="",
        leak_next_item_id="",
        leak_next_heading="",
        embedded_hits=(),
        item_full_text="",
    )


def test_diagnostics_row_csv_fields_include_form_and_missing_part() -> None:
    row = _make_diag_row(True)
    row_dict = row.to_dict({})
    assert "form_type" in row_dict
    assert row_dict["form_type"] == "10-Q"
    assert row_dict["item_missing_part"] is True

    row_false = _make_diag_row(False).to_dict({})
    assert row_false["item_missing_part"] is False
