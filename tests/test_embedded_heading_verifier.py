from __future__ import annotations

from pathlib import Path

import pytest

from thesis_pkg.core.sec.suspicious_boundary_diagnostics import (
    _build_diagnostics_report,
    _build_flagged_row,
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
    assert "toc_start_misfire_early" in classifications
    embedded_warn, embedded_fail, _first_hit, _first_flagged, _first_fail, _counts = (
        _summarize_embedded_hits(hits)
    )
    assert embedded_warn
    assert not embedded_fail


def test_embedded_heading_glued_title_marker_warns() -> None:
    text = "\n".join(
        [
            "ITEM 3. LEGAL PROCEEDINGS",
            "ITEM 4M.ine Safety Disclosures",
            "Short body line.",
        ]
    )
    hits = _find_embedded_heading_hits(text, current_item_id="3", current_part="II")
    glued_hits = [hit for hit in hits if hit.classification == "glued_title_marker"]
    assert glued_hits
    assert all(hit.item_id == "4" for hit in glued_hits)
    embedded_warn, embedded_fail, _first_hit, _first_flagged, _first_fail, _counts = (
        _summarize_embedded_hits(hits)
    )
    assert embedded_warn
    assert not embedded_fail


def test_embedded_heading_toc_cluster_no_dot_leaders_is_not_overlap() -> None:
    text = _load_fixture("toc_cluster_no_dot_leaders.txt")
    hits = _find_embedded_heading_hits(text, current_item_id="7", current_part="II")
    classifications = [hit.classification for hit in hits]
    assert "true_overlap" not in classifications
    assert "true_overlap_next_item" not in classifications
    assert any(
        cls in {"toc_start_misfire_early", "toc_row"} for cls in classifications
    )


def test_embedded_heading_early_toc_start_misfire_warns() -> None:
    text = "\n".join(
        [
            "ITEM 1. BUSINESS",
            "ITEM 2. PROPERTIES...................... 12",
            "Introductory text begins here.",
        ]
    )
    hits = _find_embedded_heading_hits(text, current_item_id="1", current_part="I")
    classifications = [hit.classification for hit in hits]
    assert "toc_start_misfire_early" in classifications
    embedded_warn, embedded_fail, _first_hit, _first_flagged, _first_fail, _counts = (
        _summarize_embedded_hits(hits)
    )
    assert embedded_warn
    assert not embedded_fail


def test_embedded_heading_successor_consistent_overlap_fails() -> None:
    filler = [
        "This is filler narrative to move the next item heading deeper into the text."
        for _ in range(10)
    ]
    text = "\n".join(
        [
            "ITEM 3. LEGAL PROCEEDINGS",
            "Some introductory text.",
            *filler,
            "ITEM 4. Mine safety disclosures and related matters",
            "This section includes several detailed sentences about mine safety disclosures.",
            "Additional narrative follows to keep the prose check satisfied.",
            "More explanation appears later in the section as part of the narrative.",
        ]
    )
    hits = _find_embedded_heading_hits(
        text,
        current_item_id="3",
        current_part="II",
        next_item_id="4",
        nearby_item_ids={"4"},
    )
    classifications = [hit.classification for hit in hits]
    assert "true_overlap_next_item" in classifications
    _embedded_warn, embedded_fail, _first_hit, _first_flagged, _first_fail, _counts = (
        _summarize_embedded_hits(hits)
    )
    assert embedded_fail


def test_embedded_heading_toc_cluster_successor_not_promoted() -> None:
    filler = [
        "This filler line pads the content so the successor heading appears later in the text."
        for _ in range(8)
    ]
    text = "\n".join(
        [
            "ITEM 1. BUSINESS",
            *filler,
            "ITEM 2.",
            "PROPERTIES",
            "ITEM 3.",
            "LEGAL PROCEEDINGS",
            "ITEM 4.",
            "MINE SAFETY DISCLOSURES",
            "This line contains enough words to count as a sentence for the prose check.",
        ]
    )
    hits = _find_embedded_heading_hits(
        text,
        current_item_id="1",
        current_part="I",
        next_item_id="2",
        nearby_item_ids={"2"},
    )
    classifications = [hit.classification for hit in hits]
    assert "true_overlap_next_item" not in classifications


def test_flagged_row_includes_embedded_fields() -> None:
    text = "\n".join(
        [
            "ITEM 6. [RESERVED]",
            "This line begins the section with prose-like text.",
            "More details are provided in subsequent sentences.",
        ]
    )
    hits = _find_embedded_heading_hits(text, current_item_id="5", current_part="II")
    embedded_warn, embedded_fail, first_hit, first_flagged, first_fail, _counts = (
        _summarize_embedded_hits(hits)
    )
    assert embedded_fail
    row = _build_flagged_row(
        doc_id="0000000000:000000000000000000",
        cik="0000000000",
        accession="000000000000000000",
        form_type="10-K",
        filing_date=None,
        period_end=None,
        item={
            "item": "II:5",
            "canonical_item": "II:5",
            "exists_by_regime": True,
            "item_status": "ok",
        },
        item_id="5",
        item_part="II",
        heading_line_clean="ITEM 5. OTHER INFORMATION",
        heading_line_raw="ITEM 5. OTHER INFORMATION",
        heading_idx=0,
        heading_offset=0,
        prefix_text="",
        prefix_kind="textual",
        is_part_only_prefix=False,
        is_crossref_prefix=False,
        prev_line="",
        next_line="",
        flags=["embedded_heading_fail"],
        embedded_hits=hits,
        embedded_warn=embedded_warn,
        embedded_fail=embedded_fail,
        embedded_first_hit=first_hit,
        embedded_first_flagged=first_flagged,
        embedded_first_fail=first_fail,
        leak_info=None,
        leak_pos="",
        leak_match="",
        leak_context="",
        leak_next_item_id="",
        leak_next_heading="",
        item_full_text=text,
    )
    provenance = {
        "prov_python": "py",
        "prov_module": "module",
        "prov_cwd": "cwd",
        "prov_sys_path_hash": "abc12345",
        "prov_enable_embedded_verifier": "True",
    }
    row_dict = row.to_dict(provenance)
    assert row_dict["embedded_heading_fail"]
    for key in (
        "embedded_heading_warn",
        "embedded_heading_fail",
        "first_embedded_kind",
        "first_embedded_class",
        "first_embedded_item_id",
        "first_embedded_part",
        "first_embedded_char_pos",
        "first_embedded_line_idx",
        "first_embedded_snippet",
    ):
        assert key in row_dict


def test_report_includes_embedded_summary_and_not_v3() -> None:
    text = "\n".join(
        [
            "ITEM 6. [RESERVED]",
            "This line begins the section with prose-like text.",
            "More details are provided in subsequent sentences.",
        ]
    )
    hits = _find_embedded_heading_hits(text, current_item_id="5", current_part="II")
    embedded_warn, embedded_fail, first_hit, first_flagged, first_fail, _counts = (
        _summarize_embedded_hits(hits)
    )
    row = _build_flagged_row(
        doc_id="0000000000:000000000000000000",
        cik="0000000000",
        accession="000000000000000000",
        form_type="10-K",
        filing_date=None,
        period_end=None,
        item={
            "item": "II:5",
            "canonical_item": "II:5",
            "exists_by_regime": True,
            "item_status": "ok",
        },
        item_id="5",
        item_part="II",
        heading_line_clean="ITEM 5. OTHER INFORMATION",
        heading_line_raw="ITEM 5. OTHER INFORMATION",
        heading_idx=0,
        heading_offset=0,
        prefix_text="",
        prefix_kind="textual",
        is_part_only_prefix=False,
        is_crossref_prefix=False,
        prev_line="",
        next_line="",
        flags=["embedded_heading_fail"],
        embedded_hits=hits,
        embedded_warn=embedded_warn,
        embedded_fail=embedded_fail,
        embedded_first_hit=first_hit,
        embedded_first_flagged=first_flagged,
        embedded_first_fail=first_fail,
        leak_info=None,
        leak_pos="",
        leak_match="",
        leak_context="",
        leak_next_item_id="",
        leak_next_heading="",
        item_full_text=text,
    )
    report = _build_diagnostics_report(
        rows=[row],
        total_filings=1,
        total_items=1,
        total_part_only_prefix=0,
        parquet_dir=Path("example_dir"),
        max_examples=5,
        provenance={
            "prov_python": "py",
            "prov_module": "module",
            "prov_cwd": "cwd",
            "prov_sys_path_hash": "abc12345",
            "prov_enable_embedded_verifier": "True",
        },
    )
    assert "Embedded heading summary:" in report
    assert "(v3)" not in report
