from __future__ import annotations

import datetime as dt

import polars as pl

from thesis_pkg.benchmarking.contracts import ItemTextCleaningConfig
from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.item_text_cleaning import build_segment_policy_id
from thesis_pkg.benchmarking.item_text_cleaning import clean_item_scopes_with_audit
from thesis_pkg.benchmarking.item_text_cleaning import clean_item_text
from thesis_pkg.benchmarking.item_text_cleaning import cleaned_scopes_for_sentence_materialization


def test_build_segment_policy_id_includes_sentence_postprocess_policy() -> None:
    segment_policy_id = build_segment_policy_id(
        SentenceDatasetConfig(postprocess_policy="item7_reference_stitch_protect_v1"),
        ItemTextCleaningConfig(),
    )

    assert "__post_item7_reference_stitch_protect_v1__" in segment_policy_id
    assert "__bucket128_256__" in segment_policy_id


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


def test_clean_item_text_default_table_drop_removes_single_numeric_line() -> None:
    text = "\n".join(
        [
            "Management overview remains ordinary prose.",
            "Loss expense reserves 100 200 300 400",
            "Closing prose resumes.",
        ]
    )

    result = clean_item_text(
        text,
        "item_7_mda",
        ItemTextCleaningConfig(drop_table_like_lines=True),
    )

    assert "Loss expense reserves 100 200 300 400" not in result.cleaned_text
    assert result.table_like_lines_removed == 1


def test_clean_item_text_block_table_drop_preserves_isolated_numeric_prose_and_removes_table_block() -> None:
    text = "\n".join(
        [
            "Management overview remains ordinary prose.",
            "from January 1, 2007 to December 31, 2007",
            "",
            "CONSOLIDATED STATEMENTS OF CASH FLOWS",
            "Loss expense reserves 100 200 300 400",
            "Policy reserves 50 60 70 80 90",
            "",
            "Closing prose resumes.",
        ]
    )
    cfg = ItemTextCleaningConfig(
        drop_table_like_lines=True,
        table_like_min_consecutive_lines=2,
        table_like_drop_header_context=True,
    )

    result = clean_item_text(text, "item_7_mda", cfg)

    assert "from January 1, 2007 to December 31, 2007" in result.cleaned_text
    assert "CONSOLIDATED STATEMENTS OF CASH FLOWS" not in result.cleaned_text
    assert "Loss expense reserves 100 200 300 400" not in result.cleaned_text
    assert "Policy reserves 50 60 70 80 90" not in result.cleaned_text
    assert result.table_like_lines_removed == 3


def test_clean_item_text_block_table_drop_scope_allowlist_skips_item_1a() -> None:
    text = "\n".join(
        [
            "Risk factors remain extensive.",
            "CONSOLIDATED STATEMENTS OF CASH FLOWS",
            "Loss expense reserves 100 200 300 400",
            "Policy reserves 50 60 70 80 90",
            "Risk prose continues.",
        ]
    )
    cfg = ItemTextCleaningConfig(
        drop_table_like_lines=True,
        table_like_min_consecutive_lines=2,
        table_like_drop_header_context=True,
        table_like_target_text_scopes=("item_7_mda", "item_1_business"),
    )

    result = clean_item_text(text, "item_1a_risk_factors", cfg)

    assert "CONSOLIDATED STATEMENTS OF CASH FLOWS" in result.cleaned_text
    assert "Loss expense reserves 100 200 300 400" in result.cleaned_text
    assert "Policy reserves 50 60 70 80 90" in result.cleaned_text
    assert result.table_like_lines_removed == 0


def test_clean_item_text_block_table_drop_allows_single_numeric_line_only_with_header() -> None:
    text = "\n".join(
        [
            "Management overview remains ordinary prose.",
            "CONSOLIDATED STATEMENTS OF CASH FLOWS",
            "Loss expense reserves 100 200 300 400",
            "",
            "from January 1, 2007 to December 31, 2007",
            "Loss expense reserves 500 600 700 800",
            "Closing prose resumes.",
        ]
    )
    cfg = ItemTextCleaningConfig(
        drop_table_like_lines=True,
        table_like_min_consecutive_lines=2,
        table_like_drop_header_context=True,
        table_like_allow_single_line_with_header=True,
    )

    result = clean_item_text(text, "item_7_mda", cfg)

    assert "CONSOLIDATED STATEMENTS OF CASH FLOWS" not in result.cleaned_text
    assert "Loss expense reserves 100 200 300 400" not in result.cleaned_text
    assert "from January 1, 2007 to December 31, 2007" in result.cleaned_text
    assert "Loss expense reserves 500 600 700 800" in result.cleaned_text
    assert result.table_like_lines_removed == 2


def test_clean_item_text_block_table_drop_removes_title_and_unit_header_block() -> None:
    text = "\n".join(
        [
            "Management overview remains ordinary prose.",
            "CONSOLIDATED RESULTS OF OPERATIONS",
            "Year ended December 31 - dollars in millions",
            "",
            "Closing prose resumes.",
        ]
    )
    cfg = ItemTextCleaningConfig(
        drop_table_like_lines=True,
        table_like_min_consecutive_lines=2,
        table_like_drop_header_context=True,
        table_like_target_text_scopes=("item_7_mda", "item_1_business"),
    )

    result = clean_item_text(text, "item_7_mda", cfg)

    assert "CONSOLIDATED RESULTS OF OPERATIONS" not in result.cleaned_text
    assert "Year ended December 31 - dollars in millions" not in result.cleaned_text
    assert "Closing prose resumes." in result.cleaned_text
    assert result.table_like_lines_removed == 2


def test_clean_item_text_block_table_drop_removes_intro_line_only_with_table_support() -> None:
    text = "\n".join(
        [
            "Management overview remains ordinary prose.",
            "The following table summarizes the components of our development revenues for the",
            "years ended December 31, 2006, 2005, and 2004:",
            "Dollars in thousands",
            "Development revenues 10 20 30 40",
            "",
            "Closing prose resumes.",
        ]
    )
    cfg = ItemTextCleaningConfig(
        drop_table_like_lines=True,
        table_like_min_consecutive_lines=2,
        table_like_drop_header_context=True,
        table_like_allow_single_line_with_header=False,
        table_like_target_text_scopes=("item_7_mda",),
    )

    result = clean_item_text(text, "item_7_mda", cfg)

    assert "The following table summarizes" not in result.cleaned_text
    assert "years ended December 31, 2006, 2005, and 2004:" not in result.cleaned_text
    assert "Dollars in thousands" not in result.cleaned_text
    assert "Development revenues 10 20 30 40" not in result.cleaned_text
    assert "Closing prose resumes." in result.cleaned_text
    assert result.table_like_lines_removed == 4


def test_clean_item_text_block_table_drop_preserves_split_narrative_table_intro_sentence() -> None:
    text = "\n".join(
        [
            "Management overview remains ordinary prose.",
            "Results of Operations The following table presents certain amounts included in our consolidated statements of income, the relative percentage that those amounts represent to revenues, and the percentage change in those amounts from year to",
            "year. This information should be read along with the consolidated financial statements and accompanying notes.",
            "",
            "Closing prose resumes.",
        ]
    )
    cfg = ItemTextCleaningConfig(
        drop_table_like_lines=True,
        table_like_min_consecutive_lines=2,
        table_like_drop_header_context=True,
        table_like_target_text_scopes=("item_7_mda",),
    )

    result = clean_item_text(text, "item_7_mda", cfg)

    assert "Results of Operations The following table presents" in result.cleaned_text
    assert "year. This information should be read" in result.cleaned_text
    assert result.table_like_lines_removed == 0


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


def test_clean_item_scopes_handles_late_non_null_drop_reason_past_inference_window() -> None:
    leading_row_count = 105
    long_text = ("Business overview remains detailed and specific. " * 20).strip()
    sections = pl.DataFrame(
        {
            "doc_id": [f"doc_{idx:03d}" for idx in range(leading_row_count)] + ["doc_drop"],
            "cik_10": ["0000000001"] * (leading_row_count + 1),
            "accession_nodash": [f"000000000100000{idx:03d}" for idx in range(leading_row_count)] + ["000000000100099999"],
            "filing_date": [dt.date(2020, 3, 1)] * (leading_row_count + 1),
            "filing_year": [2020] * (leading_row_count + 1),
            "benchmark_row_id": [f"doc_{idx:03d}:item_1" for idx in range(leading_row_count)] + ["doc_drop:item_7"],
            "benchmark_item_code": ["item_1"] * leading_row_count + ["item_7"],
            "benchmark_item_label": ["10-K Item 1"] * leading_row_count + ["10-K Item 7"],
            "item_id": ["1"] * leading_row_count + ["7"],
            "canonical_item": ["I:1_BUSINESS"] * leading_row_count + ["II:7_MDA"],
            "document_type": ["10-K"] * (leading_row_count + 1),
            "document_type_raw": ["10-K"] * (leading_row_count + 1),
            "document_type_normalized": ["10-K"] * (leading_row_count + 1),
            "source_year_file": [2020] * (leading_row_count + 1),
            "full_text": [long_text] * leading_row_count + ["short mda text"],
            "boundary_authority_status": ["auto_accepted"] * (leading_row_count + 1),
        }
    )

    result = clean_item_scopes_with_audit(
        sections,
        segment_policy_id=build_segment_policy_id(SentenceDatasetConfig(), ItemTextCleaningConfig()),
    )

    dropped_row = result.row_audit_df.filter(pl.col("doc_id") == "doc_drop").row(0, named=True)

    assert result.row_audit_df.height == leading_row_count + 1
    assert result.row_audit_df.schema["drop_reason"] == pl.Utf8
    assert result.row_audit_df.schema["manual_audit_reason"] == pl.Utf8
    assert dropped_row["drop_reason"] == "item7_below_lm_token_floor"
    assert dropped_row["item7_lm_token_floor_failed"] is True
