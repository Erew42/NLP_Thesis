from __future__ import annotations

import datetime as dt

import polars as pl

from thesis_pkg.benchmarking.contracts import ItemTextCleaningConfig
from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.item_text_cleaning import build_segment_policy_id
from thesis_pkg.benchmarking.item_text_cleaning import clean_item_scopes_with_audit
from thesis_pkg.benchmarking.item_text_cleaning import clean_item_text
from thesis_pkg.benchmarking.item_text_cleaning import cleaned_scopes_for_sentence_materialization


def test_clean_item_text_removes_layout_artifacts_but_preserves_inline_references() -> None:
    text = "\n".join(
        [
            "Management discussion starts here. See page 29 of this Form 10-K for context.",
            "1",
            "- 17 -",
            "(23)",
            "[31]",
            "Page 4",
            "4 of 87",
            "ANNUAL REPORT ON FORM 10-K",
            "<EX-21.1>",
            "cto-20221231_pre.xml",
            "",
            "Operations continue with normal prose.",
        ]
    )

    result = clean_item_text(text, "item_7_mda")

    assert "- 17 -" not in result.cleaned_text
    assert "ANNUAL REPORT ON FORM 10-K" not in result.cleaned_text
    assert "<EX-21.1>" not in result.cleaned_text
    assert "cto-20221231_pre.xml" not in result.cleaned_text
    assert "See page 29 of this Form 10-K" in result.cleaned_text
    assert "context.\n\nOperations continue" in result.cleaned_text
    assert result.page_marker_lines_removed == 6
    assert result.report_header_footer_lines_removed == 1
    assert result.structural_tag_lines_removed == 2


def test_clean_item_text_trims_clustered_toc_prefix_and_preserves_weak_heading_reference() -> None:
    clustered_toc = "\n".join(
        [
            "Item 1. Business 4",
            "Item 1A. Risk Factors 9",
            "Item 7. Management Discussion and Analysis 31",
            "",
            "Item 7. Management's Discussion and Analysis",
            "Management discusses results of operations and liquidity in detail.",
        ]
    )
    single_reference = "Item 7 is discussed elsewhere. Management discusses operations here."

    result = clean_item_text(clustered_toc, "item_7_mda")
    weak = clean_item_text(single_reference, "item_7_mda")

    assert result.toc_prefix_trimmed is True
    assert result.cleaned_text.startswith("Item 7. Management's Discussion")
    assert weak.toc_prefix_trimmed is False
    assert weak.cleaned_text == single_reference


def test_clean_item_text_truncates_item_aware_tail_without_exhibit_prose_false_positive() -> None:
    body = ("Management explains liquidity and operations. " * 20).strip()
    exhibit_reference = "The company filed an exhibit reference inside ordinary prose."
    tail = "Item 8. Financial Statements and Supplementary Data\nThis should not survive."

    result = clean_item_text(f"{body}\n{exhibit_reference}\n{tail}", "item_7_mda")

    assert result.tail_truncated is True
    assert exhibit_reference in result.cleaned_text
    assert "This should not survive" not in result.cleaned_text


def test_clean_item_scopes_enforces_item7_token_floor_and_keeps_char_warning_diagnostic_only() -> None:
    sections = pl.DataFrame(
        {
            "doc_id": ["doc1", "doc1", "doc1"],
            "cik_10": ["0000000001"] * 3,
            "accession_nodash": ["000000000100000001"] * 3,
            "filing_date": [dt.date(2020, 3, 1)] * 3,
            "filing_year": [2020] * 3,
            "benchmark_row_id": ["doc1:item_7", "doc1:item_1a", "doc1:item_1"],
            "benchmark_item_code": ["item_7", "item_1a", "item_1"],
            "benchmark_item_label": ["10-K Item 7", "10-K Item 1A", "10-K Item 1"],
            "item_id": ["7", "1A", "1"],
            "canonical_item": ["II:7_MDA", "I:1A_RISK_FACTORS", "I:1_BUSINESS"],
            "document_type": ["10-K"] * 3,
            "document_type_raw": ["10-K"] * 3,
            "document_type_normalized": ["10-K"] * 3,
            "source_year_file": [2020] * 3,
            "full_text": [
                "short mda text",
                "Risk factors are short but retained.",
                "Business is short but retained.",
            ],
        }
    )
    segment_policy_id = build_segment_policy_id(SentenceDatasetConfig(), ItemTextCleaningConfig())

    result = clean_item_scopes_with_audit(sections, segment_policy_id=segment_policy_id)

    assert result.cleaned_scope_df["text_scope"].to_list() == [
        "item_1a_risk_factors",
        "item_1_business",
    ]
    item7_audit = result.row_audit_df.filter(pl.col("text_scope") == "item_7_mda").row(0, named=True)
    assert item7_audit["drop_reason"] == "item7_below_lm_token_floor"
    assert item7_audit["item7_lm_token_floor_failed"] is True
    item1a_audit = result.row_audit_df.filter(pl.col("text_scope") == "item_1a_risk_factors").row(0, named=True)
    assert item1a_audit["warning_below_clean_char_count"] is True
    assert item1a_audit["dropped_after_cleaning"] is False
    diagnostics = result.scope_diagnostics_df.filter(pl.col("text_scope") == "item_7_mda").row(0, named=True)
    assert diagnostics["activation_status"] == "blocked_pending_manual_audit"
    assert result.manual_audit_sample_df.height > 0


def test_clean_item_scopes_fail_closed_for_unknown_or_missing_boundary_authority() -> None:
    long_text = ("Management discussion remains extensive and specific. " * 80).strip()
    sections = pl.DataFrame(
        {
            "doc_id": ["doc_unknown", "doc_missing", "doc_review", "doc_approved"],
            "cik_10": ["0000000001"] * 4,
            "accession_nodash": [
                "000000000100000001",
                "000000000100000002",
                "000000000100000003",
                "000000000100000004",
            ],
            "filing_date": [dt.date(2020, 3, 1)] * 4,
            "filing_year": [2020] * 4,
            "benchmark_row_id": [
                "doc_unknown:item_7",
                "doc_missing:item_7",
                "doc_review:item_7",
                "doc_approved:item_7",
            ],
            "benchmark_item_code": ["item_7"] * 4,
            "benchmark_item_label": ["10-K Item 7"] * 4,
            "item_id": ["7"] * 4,
            "canonical_item": ["II:7_MDA"] * 4,
            "document_type": ["10-K"] * 4,
            "document_type_raw": ["10-K"] * 4,
            "document_type_normalized": ["10-K"] * 4,
            "source_year_file": [2020] * 4,
            "full_text": [long_text] * 4,
            "boundary_authority_status": ["unknown", None, "review_needed", "unknown"],
            "review_status": [None, None, None, "approved"],
        }
    )
    segment_policy_id = build_segment_policy_id(SentenceDatasetConfig(), ItemTextCleaningConfig())

    result = clean_item_scopes_with_audit(sections, segment_policy_id=segment_policy_id)
    audit_by_doc = {
        row["doc_id"]: row
        for row in result.row_audit_df.select(
            "doc_id",
            "boundary_authority_status",
            "review_status",
            "production_eligible",
        ).to_dicts()
    }

    assert audit_by_doc["doc_unknown"]["review_status"] == "required_unreviewed"
    assert audit_by_doc["doc_unknown"]["production_eligible"] is False
    assert audit_by_doc["doc_missing"]["review_status"] == "required_unreviewed"
    assert audit_by_doc["doc_missing"]["production_eligible"] is False
    assert audit_by_doc["doc_review"]["review_status"] == "required_unreviewed"
    assert audit_by_doc["doc_review"]["production_eligible"] is False
    assert audit_by_doc["doc_approved"]["review_status"] == "approved"
    assert audit_by_doc["doc_approved"]["production_eligible"] is True

    materialized = cleaned_scopes_for_sentence_materialization(result.cleaned_scope_df)
    assert materialized["doc_id"].to_list() == ["doc_approved"]


def test_clean_item_scopes_fail_closed_when_authority_column_is_missing() -> None:
    long_text = ("Risk factors remain detailed and recurring. " * 80).strip()
    sections = pl.DataFrame(
        {
            "doc_id": ["doc_missing_authority"],
            "cik_10": ["0000000001"],
            "accession_nodash": ["000000000100000005"],
            "filing_date": [dt.date(2020, 3, 1)],
            "filing_year": [2020],
            "benchmark_row_id": ["doc_missing_authority:item_1a"],
            "benchmark_item_code": ["item_1a"],
            "benchmark_item_label": ["10-K Item 1A"],
            "item_id": ["1A"],
            "canonical_item": ["I:1A_RISK_FACTORS"],
            "document_type": ["10-K"],
            "document_type_raw": ["10-K"],
            "document_type_normalized": ["10-K"],
            "source_year_file": [2020],
            "full_text": [long_text],
        }
    )

    result = clean_item_scopes_with_audit(
        sections,
        segment_policy_id=build_segment_policy_id(SentenceDatasetConfig(), ItemTextCleaningConfig()),
    )

    audit_row = result.row_audit_df.row(0, named=True)
    assert audit_row["boundary_authority_status"] is None
    assert audit_row["review_status"] == "required_unreviewed"
    assert audit_row["production_eligible"] is False
    assert cleaned_scopes_for_sentence_materialization(result.cleaned_scope_df).is_empty()
