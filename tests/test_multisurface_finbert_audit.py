from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from thesis_pkg.benchmarking.item_text_cleaning import CLEANING_ROW_AUDIT_SCHEMA
from thesis_pkg.benchmarking.item_text_cleaning import MANUAL_AUDIT_SAMPLE_SCHEMA
from thesis_pkg.benchmarking.item_text_cleaning import SCOPE_DIAGNOSTICS_SCHEMA
from thesis_pkg.benchmarking.multisurface_audit import MultiSurfaceAuditPackConfig
from thesis_pkg.benchmarking.multisurface_audit import build_multisurface_audit_pack
from thesis_pkg.benchmarking.multisurface_audit import resolve_multisurface_audit_sources
from thesis_pkg.benchmarking.multisurface_audit import summarize_reviewed_audit_pack


def _row_audit_defaults() -> dict[str, object]:
    return {
        "doc_id": None,
        "cik_10": "0000000000",
        "accession_nodash": "000000000000000000",
        "filing_date": None,
        "filing_year": 2006,
        "calendar_year": 2006,
        "benchmark_row_id": None,
        "benchmark_item_code": None,
        "benchmark_item_label": None,
        "text_scope": None,
        "item_id": None,
        "canonical_item": None,
        "document_type": "10-K",
        "document_type_raw": "10-K",
        "document_type_normalized": "10-K",
        "source_year_file": 2006,
        "source_record_id": None,
        "source_file_row_nr": 0,
        "cleaning_policy_id": "item_text_clean_v2",
        "original_char_count": 1000,
        "cleaned_char_count": 900,
        "removed_char_count": 100,
        "removal_ratio": 0.1,
        "cleaned_lm_total_token_count": 120,
        "dropped_after_cleaning": False,
        "drop_reason": None,
        "segment_policy_id": "segment_policy_v1",
        "boundary_authority_status": "trusted",
        "review_status": "not_required",
        "production_eligible": True,
        "page_marker_lines_removed": 0,
        "report_header_footer_lines_removed": 0,
        "structural_tag_lines_removed": 0,
        "table_like_lines_removed": 0,
        "toc_prefix_trimmed": False,
        "toc_prefix_trimmed_char_count": 0,
        "tail_truncated": False,
        "tail_truncated_char_count": 0,
        "reference_only_stub": False,
        "effectively_non_body_text": False,
        "warning_large_removal": False,
        "warning_below_clean_char_count": False,
        "item7_lm_token_floor_failed": False,
        "manual_audit_candidate": False,
        "manual_audit_reason": None,
        "original_start_snippet": "",
        "cleaned_start_snippet": "",
        "original_end_snippet": "",
        "cleaned_end_snippet": "",
    }


def _manual_boundary_defaults() -> dict[str, object]:
    return {
        "doc_id": None,
        "filing_date": None,
        "calendar_year": 2006,
        "audit_period": "pre_2009",
        "text_scope": None,
        "benchmark_row_id": None,
        "cleaning_policy_id": "item_text_clean_v2",
        "boundary_authority_status": "review_needed",
        "review_status": "required_unreviewed",
        "production_eligible": False,
        "original_start_snippet": "",
        "cleaned_start_snippet": "",
        "original_end_snippet": "",
        "cleaned_end_snippet": "",
        "sample_reason": "boundary_risk",
        "dropped_after_cleaning": False,
        "warning_large_removal": False,
        "toc_prefix_trimmed": False,
        "tail_truncated": False,
        "reference_only_stub": False,
        "item7_lm_token_floor_failed": False,
        "start_boundary_correct": None,
        "end_boundary_correct": None,
        "wrong_item_capture_absent": None,
        "toc_capture_absent": None,
        "body_text_nonempty": None,
    }


def _scope_diag_defaults(text_scope: str) -> dict[str, object]:
    return {
        "calendar_year": 2006,
        "text_scope": text_scope,
        "n_filings_candidate": 2,
        "n_filings_extracted": 2,
        "extraction_rate": 1.0,
        "n_rows_after_cleaning": 2,
        "token_count_mean": 100.0,
        "token_count_median": 100.0,
        "token_count_p05": 50.0,
        "toc_trimmed_rows": 0,
        "toc_leakage_rate_proxy": 0.0,
        "tail_truncated_rows": 0,
        "reference_stub_rows": 0,
        "empty_after_cleaning_rows": 0,
        "large_removal_warning_rows": 0,
        "manual_audit_queue_n": 0,
        "manual_audit_pass_rate": 1.0,
        "activation_status": "active",
    }


def _write_fixture_run(tmp_path: Path) -> Path:
    sample_root = tmp_path / "sample"
    run_dir = sample_root / "results" / "finbert_item_analysis_runner" / "_staged_intermediates" / "test_run"
    sentence_dir = run_dir / "sentence_dataset" / "by_year"
    cleaned_scope_dir = run_dir / "cleaned_item_scopes" / "by_year"
    year_merged_dir = sample_root / "year_merged"
    sentence_dir.mkdir(parents=True, exist_ok=True)
    cleaned_scope_dir.mkdir(parents=True, exist_ok=True)
    year_merged_dir.mkdir(parents=True, exist_ok=True)

    doc1 = "0000000001:0000000001-06-000001"
    doc2 = "0000000002:0000000002-06-000002"
    doc3 = "0000000003:0000000003-06-000003"

    row_ids = {
        "doc1_item7": f"{doc1}:item_7",
        "doc1_item1a": f"{doc1}:item_1a",
        "doc2_item1": f"{doc2}:item_1",
        "doc2_item7": f"{doc2}:item_7",
        "doc3_item1a": f"{doc3}:item_1a",
        "doc3_item1": f"{doc3}:item_1",
    }

    sentence_rows = [
        {
            "benchmark_sentence_id": f"{row_ids['doc1_item7']}:0",
            "benchmark_row_id": row_ids["doc1_item7"],
            "doc_id": doc1,
            "filing_year": 2006,
            "benchmark_item_code": "item_7",
            "text_scope": "item_7_mda",
            "sentence_index": 0,
            "sentence_text": "SFAS No.",
            "sentence_char_count": 8,
            "finbert_token_count_512": 5,
            "finbert_token_bucket_512": "short",
        },
        {
            "benchmark_sentence_id": f"{row_ids['doc1_item7']}:1",
            "benchmark_row_id": row_ids["doc1_item7"],
            "doc_id": doc1,
            "filing_year": 2006,
            "benchmark_item_code": "item_7",
            "text_scope": "item_7_mda",
            "sentence_index": 1,
            "sentence_text": "CONSOLIDATED STATEMENTS OF CASH FLOWS\n2005 2004 2003\n------ ------ ------\n120 130 140\n150 160 170\nThis table continues with several numeric columns and separators that make it look like a statement block rather than ordinary prose.",
            "sentence_char_count": 248,
            "finbert_token_count_512": 210,
            "finbert_token_bucket_512": "long",
        },
        {
            "benchmark_sentence_id": f"{row_ids['doc1_item7']}:2",
            "benchmark_row_id": row_ids["doc1_item7"],
            "doc_id": doc1,
            "filing_year": 2006,
            "benchmark_item_code": "item_7",
            "text_scope": "item_7_mda",
            "sentence_index": 2,
            "sentence_text": "We monitor credit quality and adjust pricing as needed.",
            "sentence_char_count": 53,
            "finbert_token_count_512": 11,
            "finbert_token_bucket_512": "short",
        },
        {
            "benchmark_sentence_id": f"{row_ids['doc2_item7']}:0",
            "benchmark_row_id": row_ids["doc2_item7"],
            "doc_id": doc2,
            "filing_year": 2006,
            "benchmark_item_code": "item_7",
            "text_scope": "item_7_mda",
            "sentence_index": 0,
            "sentence_text": "ITEM 7.",
            "sentence_char_count": 7,
            "finbert_token_count_512": 5,
            "finbert_token_bucket_512": "short",
        },
        {
            "benchmark_sentence_id": f"{row_ids['doc2_item7']}:1",
            "benchmark_row_id": row_ids["doc2_item7"],
            "doc_id": doc2,
            "filing_year": 2006,
            "benchmark_item_code": "item_7",
            "text_scope": "item_7_mda",
            "sentence_index": 1,
            "sentence_text": "The following table sets forth contractual obligations.\nYear Ended December 31\n2005 2006 2007 2008\n------ ------ ------ ------\n50 60 70 80\n90 100 110 120\nThis merged block is intentionally long so it lands in the suspicious table-like pool.",
            "sentence_char_count": 260,
            "finbert_token_count_512": 185,
            "finbert_token_bucket_512": "long",
        },
        {
            "benchmark_sentence_id": f"{row_ids['doc2_item7']}:2",
            "benchmark_row_id": row_ids["doc2_item7"],
            "doc_id": doc2,
            "filing_year": 2006,
            "benchmark_item_code": "item_7",
            "text_scope": "item_7_mda",
            "sentence_index": 2,
            "sentence_text": "Demand remained strong across our core end markets.",
            "sentence_char_count": 52,
            "finbert_token_count_512": 10,
            "finbert_token_bucket_512": "short",
        },
    ]
    pl.DataFrame(sentence_rows).write_parquet(sentence_dir / "2006.parquet")

    cleaned_scopes = [
        {"benchmark_row_id": row_ids["doc1_item7"], "cleaned_text": "MD&A start\nSFAS No.\nCONSOLIDATED STATEMENTS OF CASH FLOWS", "cleaned_char_count": 120},
        {"benchmark_row_id": row_ids["doc1_item1a"], "cleaned_text": "Risk factor start text.", "cleaned_char_count": 120},
        {"benchmark_row_id": row_ids["doc2_item1"], "cleaned_text": "Business start text.", "cleaned_char_count": 140},
        {"benchmark_row_id": row_ids["doc2_item7"], "cleaned_text": "MD&A table intro text.", "cleaned_char_count": 160},
        {"benchmark_row_id": row_ids["doc3_item1a"], "cleaned_text": "Risk factor boundary text.", "cleaned_char_count": 130},
        {"benchmark_row_id": row_ids["doc3_item1"], "cleaned_text": "Business hotspot text.", "cleaned_char_count": 125},
    ]
    pl.DataFrame(cleaned_scopes).write_parquet(cleaned_scope_dir / "2006.parquet")

    row_audit_rows = []
    row_audit_rows.append(
        {
            **_row_audit_defaults(),
            "doc_id": doc1,
            "filing_date": "2006-03-01",
            "benchmark_row_id": row_ids["doc1_item7"],
            "benchmark_item_code": "item_7",
            "benchmark_item_label": "Item 7",
            "text_scope": "item_7_mda",
            "item_id": "7",
            "canonical_item": "item_7",
            "source_record_id": "r1",
            "removed_char_count": 900,
            "removal_ratio": 0.35,
            "manual_audit_candidate": True,
            "manual_audit_reason": "large_removed_block",
            "boundary_authority_status": "review_needed",
            "production_eligible": False,
            "original_start_snippet": "Item 7 start. SFAS No. appears early in the section.",
            "cleaned_start_snippet": "Item 7 start. SFAS No. appears early in the section.",
            "original_end_snippet": "Item 7 end with note about balances.",
            "cleaned_end_snippet": "Item 7 end with note about balances.",
            "cleaned_char_count": 1200,
        }
    )
    row_audit_rows.append(
        {
            **_row_audit_defaults(),
            "doc_id": doc1,
            "filing_date": "2006-03-01",
            "benchmark_row_id": row_ids["doc1_item1a"],
            "benchmark_item_code": "item_1a",
            "benchmark_item_label": "Item 1A",
            "text_scope": "item_1a_risk_factors",
            "item_id": "1A",
            "canonical_item": "item_1a",
            "source_record_id": "r2",
            "removed_char_count": 150,
            "removal_ratio": 0.03,
            "original_start_snippet": "Risk Factors\nItem 2. Properties\nThis looks like boundary leakage.",
            "cleaned_start_snippet": "Risk Factors\nThis looks like boundary leakage.",
            "original_end_snippet": "The risk section ends before Item 2.",
            "cleaned_end_snippet": "The risk section ends before Item 2.",
            "cleaned_char_count": 900,
        }
    )
    row_audit_rows.append(
        {
            **_row_audit_defaults(),
            "doc_id": doc2,
            "filing_date": "2006-03-02",
            "benchmark_row_id": row_ids["doc2_item1"],
            "benchmark_item_code": "item_1",
            "benchmark_item_label": "Item 1",
            "text_scope": "item_1_business",
            "item_id": "1",
            "canonical_item": "item_1",
            "source_record_id": "r3",
            "removed_char_count": 700,
            "removal_ratio": 0.22,
            "original_start_snippet": "Business start with a heading and table intro.",
            "cleaned_start_snippet": "Business start with a heading and table intro.",
            "original_end_snippet": "Business end text and website boilerplate.",
            "cleaned_end_snippet": "Business end text and website boilerplate.",
            "cleaned_char_count": 1000,
        }
    )
    row_audit_rows.append(
        {
            **_row_audit_defaults(),
            "doc_id": doc2,
            "filing_date": "2006-03-02",
            "benchmark_row_id": row_ids["doc2_item7"],
            "benchmark_item_code": "item_7",
            "benchmark_item_label": "Item 7",
            "text_scope": "item_7_mda",
            "item_id": "7",
            "canonical_item": "item_7",
            "source_record_id": "r4",
            "removed_char_count": 180,
            "removal_ratio": 0.04,
            "boundary_authority_status": "review_needed",
            "production_eligible": False,
            "original_start_snippet": "Item 7. The following table sets forth contractual obligations.",
            "cleaned_start_snippet": "Item 7. The following table sets forth contractual obligations.",
            "original_end_snippet": "Item 7 end. Signatures appear later.",
            "cleaned_end_snippet": "Item 7 end. Signatures appear later.",
            "cleaned_char_count": 1100,
        }
    )
    row_audit_rows.append(
        {
            **_row_audit_defaults(),
            "doc_id": doc3,
            "filing_date": "2006-03-03",
            "benchmark_row_id": row_ids["doc3_item1a"],
            "benchmark_item_code": "item_1a",
            "benchmark_item_label": "Item 1A",
            "text_scope": "item_1a_risk_factors",
            "item_id": "1A",
            "canonical_item": "item_1a",
            "source_record_id": "r5",
            "removed_char_count": 120,
            "removal_ratio": 0.02,
            "tail_truncated": True,
            "tail_truncated_char_count": 85,
            "original_start_snippet": "Risk factors start. Item 1B and Item 2 are listed nearby.",
            "cleaned_start_snippet": "Risk factors start.",
            "original_end_snippet": "Risk section ends before Item 1B.",
            "cleaned_end_snippet": "Risk section ends before Item 1B.",
            "cleaned_char_count": 980,
        }
    )
    row_audit_rows.append(
        {
            **_row_audit_defaults(),
            "doc_id": doc3,
            "filing_date": "2006-03-03",
            "benchmark_row_id": row_ids["doc3_item1"],
            "benchmark_item_code": "item_1",
            "benchmark_item_label": "Item 1",
            "text_scope": "item_1_business",
            "item_id": "1",
            "canonical_item": "item_1",
            "source_record_id": "r6",
            "removed_char_count": 110,
            "removal_ratio": 0.02,
            "original_start_snippet": "Business start text with clean narrative.",
            "cleaned_start_snippet": "Business start text with clean narrative.",
            "original_end_snippet": "Business end text with clean narrative. Item 1A Risk Factors appears immediately after this.",
            "cleaned_end_snippet": "Business end text with clean narrative.",
            "cleaned_char_count": 960,
        }
    )
    row_audit_df = pl.DataFrame(row_audit_rows).select(list(CLEANING_ROW_AUDIT_SCHEMA))
    row_audit_df.write_parquet(run_dir / "cleaning_row_audit.parquet")

    manual_boundary_rows = [
        {
            **_manual_boundary_defaults(),
            "doc_id": doc1,
            "filing_date": "2006-03-01",
            "text_scope": "item_1a_risk_factors",
            "benchmark_row_id": row_ids["doc1_item1a"],
            "original_start_snippet": "Risk Factors\nItem 2. Properties\nThis looks like boundary leakage.",
            "cleaned_start_snippet": "Risk Factors\nThis looks like boundary leakage.",
            "original_end_snippet": "The risk section ends before Item 2.",
            "cleaned_end_snippet": "The risk section ends before Item 2.",
        },
        {
            **_manual_boundary_defaults(),
            "doc_id": doc3,
            "filing_date": "2006-03-03",
            "text_scope": "item_1a_risk_factors",
            "benchmark_row_id": row_ids["doc3_item1a"],
            "original_start_snippet": "Risk factors start. Item 1B and Item 2 are listed nearby.",
            "cleaned_start_snippet": "Risk factors start.",
            "original_end_snippet": "Risk section ends before Item 1B.",
            "cleaned_end_snippet": "Risk section ends before Item 1B.",
        },
    ]
    pl.DataFrame(manual_boundary_rows).select(list(MANUAL_AUDIT_SAMPLE_SCHEMA)).write_parquet(
        run_dir / "manual_boundary_audit_sample.parquet"
    )

    scope_diag_rows = [
        _scope_diag_defaults("item_1_business"),
        _scope_diag_defaults("item_1a_risk_factors"),
        _scope_diag_defaults("item_7_mda"),
    ]
    pl.DataFrame(scope_diag_rows).select(list(SCOPE_DIAGNOSTICS_SCHEMA)).write_parquet(
        run_dir / "item_scope_cleaning_diagnostics.parquet"
    )

    year_merged_rows = [
        {
            "doc_id": doc1,
            "accession_number": "0000000001-06-000001",
            "full_text": "Header\nItem 7 start. SFAS No. appears early in the section. We monitor credit quality and adjust pricing as needed. CONSOLIDATED STATEMENTS OF CASH FLOWS 2005 2004 2003 120 130 140. Risk Factors This looks like boundary leakage. Item 2. Properties",
        },
        {
            "doc_id": doc2,
            "accession_number": "0000000002-06-000002",
            "full_text": "Header\nBusiness start with a heading and table intro. Item 7. The following table sets forth contractual obligations. Demand remained strong across our core end markets. Signatures appear later.",
        },
        {
            "doc_id": doc3,
            "accession_number": "0000000003-06-000003",
            "full_text": "Header\nRisk factors start. Item 1B and Item 2 are listed nearby. Business start text with clean narrative. Business end text with clean narrative.",
        },
    ]
    pl.DataFrame(year_merged_rows).write_parquet(year_merged_dir / "2006.parquet")

    manifest = {
        "path_semantics": "manifest_relative_v1",
        "artifacts": {
            "sentence_dataset_dir": "sentence_dataset",
            "cleaned_item_scopes_dir": "cleaned_item_scopes",
            "cleaning_row_audit_path": "cleaning_row_audit.parquet",
            "item_scope_cleaning_diagnostics_path": "item_scope_cleaning_diagnostics.parquet",
            "manual_boundary_audit_sample_path": "manual_boundary_audit_sample.parquet",
        },
    }
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_manifest_path


def _test_cfg() -> MultiSurfaceAuditPackConfig:
    return MultiSurfaceAuditPackConfig(
        sentence_short_case_target=2,
        sentence_long_case_target=2,
        sentence_control_case_target=2,
        item_cleaning_case_target=2,
        item_hotspot_case_target=2,
        item_boundary_case_target=2,
        escalation_cap=4,
        chunk_count=3,
        random_seed=17,
        full_report_window_chars=120,
    )


def test_build_multisurface_audit_pack_creates_cases_and_chunks(tmp_path: Path) -> None:
    run_manifest_path = _write_fixture_run(tmp_path)
    sources = resolve_multisurface_audit_sources(run_manifest_path)
    output_dir = tmp_path / "audit_pack"

    artifacts = build_multisurface_audit_pack(sources, output_dir=output_dir, cfg=_test_cfg())

    assert artifacts.primary_case_count == 12
    assert artifacts.escalated_case_count == 4
    assert artifacts.requested_full_report_doc_count >= 1
    assert artifacts.fetched_full_report_doc_count == artifacts.requested_full_report_doc_count

    cases_df = pl.read_parquet(artifacts.audit_cases_path)
    assert cases_df.height == 12
    assert cases_df["case_id"].n_unique() == 12
    assert cases_df.filter(pl.col("doc_id").is_null() | (pl.col("primary_text") == "")).is_empty()

    escalated_df = cases_df.filter(pl.col("full_report_needed"))
    assert escalated_df.height == 4
    assert escalated_df.filter(pl.col("full_report_match_status").is_null()).is_empty()

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["chunks"]) == 3
    for chunk in manifest["chunks"]:
        chunk_path = artifacts.manifest_path.parent / chunk["path"]
        chunk_df = pl.read_csv(chunk_path)
        assert chunk_df.height == chunk["expected_row_count"]
        assert set(chunk_df["case_id"].to_list()) == set(chunk["case_ids"])


def test_summarize_multisurface_audit_pack_validates_and_aggregates(tmp_path: Path) -> None:
    run_manifest_path = _write_fixture_run(tmp_path)
    sources = resolve_multisurface_audit_sources(run_manifest_path)
    output_dir = tmp_path / "audit_pack"
    artifacts = build_multisurface_audit_pack(sources, output_dir=output_dir, cfg=_test_cfg())

    for chunk_path in sorted((artifacts.chunk_dir).glob("chunk_*.csv")):
        chunk_df = pl.read_csv(chunk_path).with_columns(
            [
                pl.when(pl.col("case_source") == "sentence_short_fragment")
                .then(pl.lit("artifact_reference_stub"))
                .when(pl.col("case_source") == "sentence_long_table_like")
                .then(pl.lit("artifact_table_row"))
                .when(pl.col("case_source") == "sentence_control")
                .then(pl.lit("valid_prose"))
                .when(pl.col("case_source") == "item_boundary_risk")
                .then(pl.lit("boundary_error_item"))
                .when(pl.col("case_source") == "item_cleaning_flagged")
                .then(pl.lit("split_prose_fragment"))
                .otherwise(pl.lit("uncertain"))
                .alias("review_label"),
                pl.when(pl.col("case_source") == "sentence_short_fragment")
                .then(pl.lit("sentence_segmentation"))
                .when(pl.col("case_source") == "sentence_long_table_like")
                .then(pl.lit("report_layout_loss"))
                .when(pl.col("case_source") == "sentence_control")
                .then(pl.lit("mixed_or_unknown"))
                .when(pl.col("case_source") == "item_boundary_risk")
                .then(pl.lit("item_boundary_extraction"))
                .when(pl.col("case_source") == "item_cleaning_flagged")
                .then(pl.lit("item_cleaning"))
                .otherwise(pl.lit("mixed_or_unknown"))
                .alias("root_cause"),
                pl.when(pl.col("case_source") == "sentence_short_fragment")
                .then(pl.lit("sentence_protect_rule"))
                .when(pl.col("case_source") == "sentence_long_table_like")
                .then(pl.lit("cleaner_rule"))
                .when(pl.col("case_source") == "sentence_control")
                .then(pl.lit("no_change"))
                .when(pl.col("case_source") == "item_boundary_risk")
                .then(pl.lit("boundary_extraction_followup"))
                .when(pl.col("case_source") == "item_cleaning_flagged")
                .then(pl.lit("cleaner_rule"))
                .otherwise(pl.lit("no_change"))
                .alias("recommended_action"),
                pl.when(pl.col("full_report_needed"))
                .then(pl.lit(True))
                .otherwise(pl.lit(None, dtype=pl.Boolean))
                .alias("full_report_changed_decision"),
                pl.concat_str([pl.lit("reviewed "), pl.col("case_id")]).alias("notes"),
            ]
        )
        chunk_df.write_csv(chunk_path)

    summary = summarize_reviewed_audit_pack(output_dir)
    review_summary = json.loads(summary.review_summary_path.read_text(encoding="utf-8"))
    assert review_summary["counts"]["reviewed_case_count"] == 12
    assert review_summary["counts"]["escalated_case_count"] == 4
    assert summary.pattern_summary_path.exists()
    assert summary.rule_candidates_path.exists()
    assert summary.do_not_touch_patterns_path.exists()


def test_summarizer_rejects_row_count_mismatch(tmp_path: Path) -> None:
    run_manifest_path = _write_fixture_run(tmp_path)
    sources = resolve_multisurface_audit_sources(run_manifest_path)
    output_dir = tmp_path / "audit_pack"
    artifacts = build_multisurface_audit_pack(sources, output_dir=output_dir, cfg=_test_cfg())

    first_chunk = sorted(artifacts.chunk_dir.glob("chunk_*.csv"))[0]
    truncated = pl.read_csv(first_chunk).head(1)
    truncated.write_csv(first_chunk)

    with pytest.raises(ValueError, match="row count"):
        summarize_reviewed_audit_pack(output_dir)


def test_summarizer_accepts_boolean_strings_in_reviewed_csv(tmp_path: Path) -> None:
    run_manifest_path = _write_fixture_run(tmp_path)
    sources = resolve_multisurface_audit_sources(run_manifest_path)
    output_dir = tmp_path / "audit_pack"
    artifacts = build_multisurface_audit_pack(sources, output_dir=output_dir, cfg=_test_cfg())

    for chunk_path in sorted((artifacts.chunk_dir).glob("chunk_*.csv")):
        chunk_df = pl.read_csv(chunk_path).with_columns(
            [
                pl.lit("uncertain").alias("review_label"),
                pl.lit("mixed_or_unknown").alias("root_cause"),
                pl.lit("no_change").alias("recommended_action"),
                pl.when(pl.col("full_report_needed"))
                .then(pl.lit("false"))
                .otherwise(pl.lit(None))
                .alias("full_report_changed_decision"),
                pl.lit("ok").alias("notes"),
            ]
        )
        chunk_df.write_csv(chunk_path)

    summary = summarize_reviewed_audit_pack(output_dir)
    assert summary.review_summary_path.exists()
