from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from thesis_pkg.notebooks_and_scripts import lm2011_phase0_validation_audit as audit_cli
from thesis_pkg.pipelines import lm2011_validation_audit as audit


@dataclass(frozen=True)
class _AuditLayout:
    sample_root: Path
    upstream_run_root: Path
    lm2011_output_dir: Path
    finbert_output_root: Path
    output_root: Path
    sample_backbone_path: Path
    event_panel_path: Path
    finbert_run_dir: Path


def _write_parquet(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _repeat(token: str, count: int) -> str:
    return " ".join([token] * count)


def _structured_filing_text(label: str) -> str:
    return "\n".join(
        [
            "ITEM 1. Business",
            _repeat(f"{label}business", 80),
            f"{label}-linked a",
            "ITEM 1A. Risk Factors",
            _repeat(f"{label}risk", 80),
            "ITEM 7. Management's Discussion and Analysis of Financial Condition and Results of Operations",
            _repeat(f"{label}analysis", 120),
            "ITEM 8. Financial Statements and Supplementary Data",
            _repeat(f"{label}financial", 40),
        ]
    )


def _appendix_style_tokenize(text: str | None) -> list[str]:
    if text is None:
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"([A-Za-z])-\s*(?:\r?\n)\s*([A-Za-z])", r"\1-\2", normalized)
    return [token.casefold() for token in re.findall(r"[A-Za-z]{2,}(?:[-'][A-Za-z]{2,})*", normalized)]


def _token_delta_metrics(rows: list[dict[str, object]], scope: str) -> dict[str, object]:
    scoped_rows = [row for row in rows if row["text_scope"] == scope]
    doc_count = len(scoped_rows)
    current_vs_appendix_doc_count = sum(row["current_token_count"] != row["appendix_token_count"] for row in scoped_rows)
    appendix_vs_recognized_doc_count = sum(row["appendix_token_count"] != row["recognized_word_count"] for row in scoped_rows)
    mean_current_minus_appendix = (
        sum(row["current_token_count"] - row["appendix_token_count"] for row in scoped_rows) / doc_count if doc_count else 0.0
    )
    mean_appendix_minus_recognized = (
        sum(row["appendix_token_count"] - row["recognized_word_count"] for row in scoped_rows) / doc_count if doc_count else 0.0
    )
    return {
        "doc_count": doc_count,
        "current_vs_appendix_doc_count": current_vs_appendix_doc_count,
        "appendix_vs_recognized_doc_count": appendix_vs_recognized_doc_count,
        "mean_current_minus_appendix": mean_current_minus_appendix,
        "mean_appendix_minus_recognized": mean_appendix_minus_recognized,
    }


def _build_sample_audit_layout(tmp_path: Path) -> _AuditLayout:
    sample_root = tmp_path / "full_data_run" / "sample_5pct_seed42"
    upstream_run_root = sample_root / "results" / "sec_ccm_unified_runner" / "local_sample"
    lm2011_output_dir = sample_root / "results" / "lm2011_sample_post_refinitiv_runner"
    finbert_output_root = sample_root / "results" / "finbert_item_analysis_runner"
    finbert_run_dir = finbert_output_root / "sample_smoke_2006"
    output_root = tmp_path / "reports" / "audit_output"

    dirty_full_text = "\n".join(
        [
            "<SEC-HEADER>metadata</SEC-HEADER>",
            _repeat("a", 2100),
            "EXHIBIT INDEX",
            "EXHIBIT 21",
            _repeat("loss", 50),
        ]
    )
    clean_full_text = _repeat("loss", 2050)
    broad_extra_text = _repeat("lawsuit", 120)

    filing_date = dt.date(2006, 3, 15)
    second_filing_date = dt.date(2006, 3, 20)
    period_end = dt.date(2005, 12, 31)

    sample_backbone_path = lm2011_output_dir / "lm2011_sample_backbone.parquet"
    _write_parquet(
        sample_backbone_path,
        pl.DataFrame(
            {
                "doc_id": ["d1", "d2"],
                "filing_date": [filing_date, second_filing_date],
                "normalized_form": ["10-K", "10-K405"],
                "document_type_filename": ["10-K", "10-K405"],
                "period_end": [period_end, period_end],
                "gvkey_int": [1, 2],
                "KYPERMNO": [10001, 10002],
                "full_text": [_structured_filing_text("alpha"), _structured_filing_text("beta")],
            }
        ),
    )

    _write_parquet(
        sample_root / "year_merged" / "2006.parquet",
        pl.DataFrame(
            {
                "doc_id": ["d1", "d2", "d3"],
                "full_text": [dirty_full_text, clean_full_text, broad_extra_text],
                "document_type_filename": ["10-K", "10-K405", "10-Q"],
            }
        ),
    )

    _write_parquet(
        lm2011_output_dir / "lm2011_text_features_full_10k.parquet",
        pl.DataFrame(
            {
                "doc_id": ["d1", "d2", "d3"],
                "filing_date": [filing_date, second_filing_date, dt.date(2006, 4, 1)],
                "normalized_form": ["10-K", "10-K405", "10-Q"],
                "token_count_full_10k": [2100, 2050, 120],
                "total_token_count_full_10k": [2100, 2050, 120],
            }
        ),
    )

    _write_parquet(
        lm2011_output_dir / "lm2011_text_features_mda.parquet",
        pl.DataFrame(
            {
                "doc_id": ["d1", "d2"],
                "filing_date": [filing_date, second_filing_date],
                "normalized_form": ["10-K", "10-K405"],
                "item_id": ["7", "7"],
                "token_count_mda": [260, 320],
                "total_token_count_mda": [260, 320],
            }
        ),
    )

    event_panel_path = lm2011_output_dir / "lm2011_event_panel.parquet"
    _write_parquet(
        event_panel_path,
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "filing_date": [filing_date],
                "gvkey_int": [1],
                "KYPERMNO": [10001],
                "pre_filing_trade_date": [dt.date(2006, 3, 14)],
                "filing_period_excess_return": [0.02],
            }
        ),
    )

    _write_parquet(
        lm2011_output_dir / "lm2011_sue_panel.parquet",
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "sue": [0.015],
                "analyst_dispersion": [0.005],
                "analyst_revisions": [0.01],
            }
        ),
    )

    _write_parquet(
        lm2011_output_dir / "lm2011_quarterly_accounting_panel.parquet",
        pl.DataFrame(
            {
                "gvkey_int": [1],
                "quarter_report_date": [dt.date(2006, 3, 31)],
                "APDEDATEQ": [period_end],
                "KEYSET": ["STD"],
                "FYYYYQ": [2006],
                "fyrq": [1],
            }
        ),
    )

    empty_results = pl.DataFrame(schema={"table_id": pl.Utf8, "estimate": pl.Float64})
    _write_parquet(lm2011_output_dir / "lm2011_table_iv_results.parquet", empty_results)
    _write_parquet(lm2011_output_dir / "lm2011_table_viii_results.parquet", empty_results)

    _write_parquet(
        sample_root / "derived_data" / "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet",
        pl.DataFrame(
            {
                "doc_id": ["d1", "d1", "d1", "d2", "d3", "d4", "d5"],
                "KYPERMNO": [10001, 10001, 10001, 10002, 10003, 10004, 10005],
                "CALDT": [
                    dt.date(2006, 2, 28),
                    dt.date(2006, 3, 14),
                    dt.date(2006, 3, 31),
                    dt.date(2006, 3, 14),
                    dt.date(2006, 3, 14),
                    dt.date(2006, 3, 14),
                    dt.date(2006, 3, 14),
                ],
                "FINAL_PRC": [8.0, 10.0, 20.0, 11.0, 12.0, 5.0, 6.0],
            }
        ),
    )

    _write_parquet(
        upstream_run_root / "refinitiv_doc_analyst_lm2011" / "refinitiv_doc_analyst_selected.parquet",
        pl.DataFrame(
            {
                "gvkey_int": [1],
                "matched_announcement_date": [dt.date(2006, 3, 31)],
                "matched_fiscal_period_end": [period_end],
                "actual_eps": [1.0],
                "forecast_consensus_mean": [0.8],
                "forecast_dispersion": [0.1],
                "forecast_revision_4m": [0.2],
                "forecast_revision_1m": [0.1],
                "analyst_match_status": ["MATCHED"],
            }
        ),
    )

    _write_parquet(
        upstream_run_root / "items_analysis" / "2006.parquet",
        pl.DataFrame(
            {
                "cik_10": ["0000000001", "0000000001", "0000000001", "0000000001", "0000000002", "0000000002", "0000000003", "0000000004"],
                "accession_nodash": ["1", "1", "1", "1", "2", "2", "3", "4"],
                "filing_date": [filing_date] * 8,
                "doc_id": ["d1", "d1", "d1", "d1", "d2", "d2", "d3", "d4"],
                "item_id": ["1", "1A", "1A", "7", "7", "1", "1", "1"],
                "full_text": [
                    _repeat("business", 60),
                    _repeat("risk", 60),
                    _repeat("risk", 55),
                    _repeat("a", 260),
                    _repeat("management", 60),
                    _repeat("business", 60),
                    _repeat("business", 60),
                    _repeat("business", 60),
                ],
                "document_type_filename": ["10-K", "10-K", "10-K", "10-K", "10-K405", "10-K405", "10-K", "8-K"],
                "item_status": ["active", "active", "active", "active", "active", "inactive", "active", "active"],
                "exists_by_regime": [True, True, True, True, True, True, True, True],
                "canonical_item": ["item1", "item1a", "item1a_dup", "item7", "item7", "item1", "item1", "item1"],
                "filename": ["d1.txt", "d1.txt", "d1_dup.txt", "d1.txt", "d2.txt", "d2.txt", "d3.txt", "d4.txt"],
            }
        ),
    )

    _write_text(
        sample_root.parent / "LM2011_additional_data" / "LM2011_MasterDictionary.txt",
        "Word\nloss\ngain\nuncertain\nlawsuit\nmust\nmay\nbusiness\nrisk\nmanagement\nanalysis\nfinancial\n",
    )

    _write_parquet(
        finbert_run_dir / "item_features_long.parquet",
        pl.DataFrame(
            {
                "doc_id": ["d1", "d1", "d2", "d3"],
                "benchmark_item_code": ["item_1", "item_7", "item_7", "item_1"],
            }
        ),
    )
    _write_parquet(
        finbert_run_dir / "coverage_report.parquet",
        pl.DataFrame(
            {
                "metric": ["backbone_doc_count", "covered_doc_count"],
                "value": [5, 3],
            }
        ),
    )
    _write_json(
        finbert_run_dir / "run_manifest.json",
        {
            "run_name": "sample_smoke_2006",
            "backbone_path": str(sample_backbone_path),
            "year_filter": [2006],
            "coverage_summary": {
                "backbone_doc_count": 5,
                "covered_doc_count": 3,
            },
            "section_universe": {
                "form_types": ["10-K", "10-K405"],
                "require_active_items": True,
                "require_exists_by_regime": True,
                "min_char_count": 250,
                "target_items": [
                    {"benchmark_item_code": "item_1", "item_id": "1", "benchmark_item_label": "Item 1"},
                    {"benchmark_item_code": "item_1a", "item_id": "1A", "benchmark_item_label": "Item 1A"},
                    {"benchmark_item_code": "item_7", "item_id": "7", "benchmark_item_label": "Item 7"},
                ],
            },
        },
    )

    return _AuditLayout(
        sample_root=sample_root,
        upstream_run_root=upstream_run_root,
        lm2011_output_dir=lm2011_output_dir,
        finbert_output_root=finbert_output_root,
        output_root=output_root,
        sample_backbone_path=sample_backbone_path,
        event_panel_path=event_panel_path,
        finbert_run_dir=finbert_run_dir,
    )


def _run_audit(layout: _AuditLayout, packets: tuple[str, ...] = audit.PACKET_CHOICES) -> audit.Phase0ValidationAuditArtifacts:
    cfg = audit.Phase0ValidationAuditConfig(
        sample_root=layout.sample_root,
        upstream_run_root=layout.upstream_run_root,
        lm2011_output_dir=layout.lm2011_output_dir,
        finbert_output_root=layout.finbert_output_root,
        output_root=layout.output_root,
        packets=packets,
    )
    return audit.run_phase0_validation_audit(cfg)


def _write_finbert_candidate_run(run_dir: Path, run_manifest: dict[str, object], *, mtime: int) -> None:
    _write_json(run_dir / "run_manifest.json", run_manifest)
    _write_parquet(
        run_dir / "item_features_long.parquet",
        pl.DataFrame({"doc_id": ["d1"], "benchmark_item_code": ["item_1"]}),
    )
    _write_parquet(
        run_dir / "coverage_report.parquet",
        pl.DataFrame({"metric": ["backbone_doc_count", "covered_doc_count"], "value": [2, 1]}),
    )
    timestamp = float(mtime)
    for path in (run_dir, run_dir / "run_manifest.json", run_dir / "item_features_long.parquet", run_dir / "coverage_report.parquet"):
        os.utime(path, (timestamp, timestamp))


def test_packet_a_detects_marker_and_threshold_flips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = _build_sample_audit_layout(tmp_path)
    monkeypatch.setattr(audit, "tokenize_lm2011_text", _appendix_style_tokenize)

    artifacts = _run_audit(layout, packets=("A",))

    assert artifacts.packet_statuses["A"] == audit.STATUS_COMPLETED_WITH_WARNINGS
    threshold_df = pl.read_parquet(layout.output_root / "packet_a_threshold_diff.parquet")
    summary_df = pl.read_parquet(layout.output_root / "packet_a_summary.parquet")
    strip_df = pl.read_parquet(layout.output_root / "packet_a_strip_comparison.parquet")
    examples_df = pl.read_parquet(layout.output_root / "packet_a_examples.parquet")
    report_text = (layout.output_root / "phase0_validation_report.md").read_text(encoding="utf-8")

    full_flip = threshold_df.filter((pl.col("doc_id") == "d1") & (pl.col("text_scope") == "full_10k")).row(0, named=True)
    mda_flip = threshold_df.filter((pl.col("doc_id") == "d1") & (pl.col("text_scope") == "mda")).row(0, named=True)
    full_metrics = _token_delta_metrics(threshold_df.to_dicts(), "full_10k")
    mda_metrics = _token_delta_metrics(threshold_df.to_dicts(), "mda")

    assert full_flip["threshold_flip"] is True
    assert full_flip["appendix_threshold_pass"] is False
    assert full_flip["any_marker"] is True
    assert full_flip["edgar_stripped_has_html_marker"] is False
    assert full_flip["paper_cleaned_has_exhibit_marker"] is False
    assert full_flip["paper_cleaned_cut_reason"] == "strong_anchor_exhibit_index"
    assert full_flip["paper_cleaned_current_token_count"] < full_flip["edgar_stripped_current_token_count"]
    assert mda_flip["threshold_flip"] is True
    assert mda_flip["appendix_threshold_pass"] is False
    assert int(summary_df.filter((pl.col("text_scope") == "full_10k") & pl.col("filing_year").is_null()).item(0, "threshold_flip_count")) == 1
    assert int(strip_df.filter((pl.col("text_scope") == "full_10k") & pl.col("filing_year").is_null()).item(0, "docs_with_exhibit_marker_after_paper_cleaning")) == 0
    assert int(strip_df.filter((pl.col("text_scope") == "full_10k") & pl.col("filing_year").is_null()).item(0, "truncated_doc_count")) == 1
    assert examples_df.filter(pl.col("paper_cleaned_cut_reason") == "strong_anchor_exhibit_index").height == 1
    assert audit._legacy_tokenize_lm2011_text("alpha-linked a") == ["alpha", "linked", "a"]
    assert audit.tokenize_lm2011_text("alpha-linked a") == ["alpha-linked"]

    full_current_line = (
        f"- Accepted sampled full 10-K current-vs-Appendix token delta: "
        f"{full_metrics['current_vs_appendix_doc_count']}/{full_metrics['doc_count']} docs differ; "
        f"mean current-minus-Appendix = {full_metrics['mean_current_minus_appendix']:.2f}"
    )
    full_recognized_line = (
        f"- Accepted sampled full 10-K Appendix-vs-recognized token delta: "
        f"{full_metrics['appendix_vs_recognized_doc_count']}/{full_metrics['doc_count']} docs differ; "
        f"mean Appendix-minus-recognized = {full_metrics['mean_appendix_minus_recognized']:.2f}"
    )
    mda_current_line = (
        f"- Accepted sampled MD&A current-vs-Appendix token delta: "
        f"{mda_metrics['current_vs_appendix_doc_count']}/{mda_metrics['doc_count']} docs differ; "
        f"mean current-minus-Appendix = {mda_metrics['mean_current_minus_appendix']:.2f}"
    )
    mda_recognized_line = (
        f"- Accepted sampled MD&A Appendix-vs-recognized token delta: "
        f"{mda_metrics['appendix_vs_recognized_doc_count']}/{mda_metrics['doc_count']} docs differ; "
        f"mean Appendix-minus-recognized = {mda_metrics['mean_appendix_minus_recognized']:.2f}"
    )
    assert full_current_line in report_text
    assert full_recognized_line in report_text
    assert mda_current_line in report_text
    assert mda_recognized_line in report_text


def test_packet_b_reports_corpus_scope_and_event_attrition(tmp_path: Path) -> None:
    layout = _build_sample_audit_layout(tmp_path)

    artifacts = _run_audit(layout, packets=("B",))

    assert artifacts.packet_statuses["B"] == audit.STATUS_COMPLETED
    corpus_df = pl.read_parquet(layout.output_root / "packet_b_corpus_summary.parquet")
    attrition_df = pl.read_parquet(layout.output_root / "packet_b_event_attrition.parquet")

    assert int(corpus_df.filter(pl.col("corpus_label") == "year_merged_broad").item(0, "doc_count")) == 3
    assert int(corpus_df.filter(pl.col("corpus_label") == "accepted_backbone").item(0, "doc_count")) == 2
    assert int(corpus_df.filter(pl.col("corpus_label") == "text_features_full_10k").item(0, "doc_count")) == 3
    assert int(attrition_df.filter(pl.col("filing_year").is_null()).item(0, "lost_doc_count")) == 1


def test_packet_c_units_and_denominator_audit(tmp_path: Path) -> None:
    layout = _build_sample_audit_layout(tmp_path)

    artifacts = _run_audit(layout, packets=("C",))

    assert artifacts.packet_statuses["C"] == audit.STATUS_COMPLETED
    units_df = pl.read_parquet(layout.output_root / "packet_c_units_audit.parquet")
    denominator_df = pl.read_parquet(layout.output_root / "packet_c_denominator_candidates.parquet")
    classification_df = pl.read_parquet(layout.output_root / "packet_c_output_classification.parquet")

    assert int(units_df.filter(pl.col("field_name") == "filing_period_excess_return").item(0, "nonnull_row_count")) == 1
    assert float(denominator_df.item(0, "filing_anchor_sue")) == pytest.approx(0.02)
    assert float(denominator_df.item(0, "announcement_anchor_sue")) == pytest.approx(0.01)
    assert float(denominator_df.item(0, "abs_diff_sue")) == pytest.approx(0.01)
    assert "clearly_internal_unit" in classification_df.get_column("classification").to_list()


def test_packet_d_reconciles_dictionary_and_finbert_universe(tmp_path: Path) -> None:
    layout = _build_sample_audit_layout(tmp_path)

    artifacts = _run_audit(layout, packets=("D",))

    assert artifacts.packet_statuses["D"] == audit.STATUS_COMPLETED_WITH_WARNINGS
    reconciliation_df = pl.read_parquet(layout.output_root / "packet_d_reconciliation.parquet")
    coverage_df = pl.read_parquet(layout.output_root / "packet_d_coverage_reconciliation.parquet")
    removal_df = pl.read_parquet(layout.output_root / "packet_d_removal_waterfall.parquet")
    regime_df = pl.read_parquet(layout.output_root / "packet_d_regime_comparison.parquet")

    assert int(reconciliation_df.filter((pl.col("item_id") == "1A") & (pl.col("classification") == "dictionary_only")).item(0, "row_count")) == 1
    assert int(reconciliation_df.filter((pl.col("item_id") == "1") & (pl.col("classification") == "finbert_only")).item(0, "row_count")) == 1
    assert int(coverage_df.item(0, "reported_backbone_doc_count")) == 5
    assert int(coverage_df.item(0, "actual_filtered_doc_count")) == 2
    assert int(coverage_df.item(0, "sample_backbone_filtered_doc_count")) == 2
    assert int(coverage_df.item(0, "sample_backbone_denominator_gap")) == 3
    assert coverage_df.item(0, "finbert_backbone_matches_sample_backbone") is True
    assert int(removal_df.filter(pl.col("stage_name") == "raw_target_items").item(0, "row_count")) > int(
        removal_df.filter(pl.col("stage_name") == "deduped_final").item(0, "row_count")
    )
    assert regime_df.height > 0


def test_resolve_latest_valid_finbert_run_prefers_newest_compatible_backbone_contract(tmp_path: Path) -> None:
    layout = _build_sample_audit_layout(tmp_path)
    shutil.rmtree(layout.finbert_run_dir)
    sampled_items_analysis_dir = layout.upstream_run_root / "items_analysis"
    base_manifest = {
        "year_filter": [2006],
        "section_universe": {
            "form_types": ["10-K", "10-K405"],
            "require_active_items": True,
            "require_exists_by_regime": True,
            "min_char_count": 250,
            "target_items": [
                {"benchmark_item_code": "item_1", "item_id": "1", "benchmark_item_label": "Item 1"},
                {"benchmark_item_code": "item_1a", "item_id": "1A", "benchmark_item_label": "Item 1A"},
                {"benchmark_item_code": "item_7", "item_id": "7", "benchmark_item_label": "Item 7"},
            ],
        },
    }
    section_cfg = audit._section_universe_from_manifest(sampled_items_analysis_dir, base_manifest)
    expected_contract = audit._expected_packet_d_contract(
        section_cfg=section_cfg,
        sample_backbone_path=layout.sample_backbone_path,
        effective_year_filter=(2006,),
    )

    newer_incompatible_dir = layout.finbert_output_root / "newer_incompatible"
    older_compatible_dir = layout.finbert_output_root / "older_compatible"
    _write_finbert_candidate_run(
        newer_incompatible_dir,
        {
            **base_manifest,
            "run_name": "newer_incompatible",
            "backbone_path": str(layout.sample_backbone_path),
            "backbone_contract": {
                **expected_contract,
                "contract_version": "backbone_contract_v2",
                "filtered_backbone_doc_universe_fingerprint": "mismatched_doc_universe",
            },
        },
        mtime=2_000,
    )
    _write_finbert_candidate_run(
        older_compatible_dir,
        {
            **base_manifest,
            "run_name": "older_compatible",
            "backbone_path": str(layout.sample_backbone_path),
            "backbone_contract": {
                **expected_contract,
                "contract_version": "backbone_contract_v2",
            },
        },
        mtime=1_000,
    )

    selected = audit._resolve_latest_valid_finbert_run(
        layout.finbert_output_root,
        sampled_items_analysis_dir=sampled_items_analysis_dir,
        sample_backbone_path=layout.sample_backbone_path,
        requested_year_filter=(2006,),
    )

    assert selected == older_compatible_dir


def test_resolve_latest_valid_finbert_run_accepts_legacy_path_fallback_only_without_filtered_fingerprint(
    tmp_path: Path,
) -> None:
    layout = _build_sample_audit_layout(tmp_path)
    shutil.rmtree(layout.finbert_run_dir)
    sampled_items_analysis_dir = layout.upstream_run_root / "items_analysis"
    base_manifest = {
        "run_name": "legacy_path_fallback",
        "backbone_path": str(layout.sample_backbone_path),
        "year_filter": [2006],
        "section_universe": {
            "form_types": ["10-K", "10-K405"],
            "require_active_items": True,
            "require_exists_by_regime": True,
            "min_char_count": 250,
            "target_items": [
                {"benchmark_item_code": "item_1", "item_id": "1", "benchmark_item_label": "Item 1"},
                {"benchmark_item_code": "item_1a", "item_id": "1A", "benchmark_item_label": "Item 1A"},
                {"benchmark_item_code": "item_7", "item_id": "7", "benchmark_item_label": "Item 7"},
            ],
        },
    }
    section_cfg = audit._section_universe_from_manifest(sampled_items_analysis_dir, base_manifest)
    accepted_universe_contract = audit.section_universe_contract_payload(
        section_cfg,
        target_doc_universe_path=layout.sample_backbone_path,
    )
    legacy_dir = layout.finbert_output_root / "legacy_path_fallback"
    _write_finbert_candidate_run(
        legacy_dir,
        {
            **base_manifest,
            "source_sentence_dataset_manifest": {
                "accepted_universe_contract": accepted_universe_contract,
            },
        },
        mtime=1_000,
    )

    selected = audit._resolve_latest_valid_finbert_run(
        layout.finbert_output_root,
        sampled_items_analysis_dir=sampled_items_analysis_dir,
        sample_backbone_path=layout.sample_backbone_path,
        requested_year_filter=(2006,),
    )

    assert selected == legacy_dir


def test_resolve_latest_valid_finbert_run_rejects_incompatible_accepted_universe_contract(tmp_path: Path) -> None:
    layout = _build_sample_audit_layout(tmp_path)
    shutil.rmtree(layout.finbert_run_dir)
    sampled_items_analysis_dir = layout.upstream_run_root / "items_analysis"
    base_manifest = {
        "year_filter": [2006],
        "section_universe": {
            "form_types": ["10-K", "10-K405"],
            "require_active_items": True,
            "require_exists_by_regime": True,
            "min_char_count": 250,
            "target_items": [
                {"benchmark_item_code": "item_1", "item_id": "1", "benchmark_item_label": "Item 1"},
                {"benchmark_item_code": "item_1a", "item_id": "1A", "benchmark_item_label": "Item 1A"},
                {"benchmark_item_code": "item_7", "item_id": "7", "benchmark_item_label": "Item 7"},
            ],
        },
    }
    section_cfg = audit._section_universe_from_manifest(sampled_items_analysis_dir, base_manifest)
    expected_contract = audit._expected_packet_d_contract(
        section_cfg=section_cfg,
        sample_backbone_path=layout.sample_backbone_path,
        effective_year_filter=(2006,),
    )
    incompatible_dir = layout.finbert_output_root / "incompatible_accepted_universe"
    _write_finbert_candidate_run(
        incompatible_dir,
        {
            **base_manifest,
            "run_name": "incompatible_accepted_universe",
            "backbone_path": str(layout.sample_backbone_path),
            "backbone_contract": {
                **expected_contract,
                "contract_version": "backbone_contract_v2",
                "accepted_universe_contract_fingerprint": "different_contract",
            },
        },
        mtime=1_000,
    )

    selected = audit._resolve_latest_valid_finbert_run(
        layout.finbert_output_root,
        sampled_items_analysis_dir=sampled_items_analysis_dir,
        sample_backbone_path=layout.sample_backbone_path,
        requested_year_filter=(2006,),
    )

    assert selected is None


def test_cli_main_writes_report_and_manifest_for_sample_layout(tmp_path: Path) -> None:
    layout = _build_sample_audit_layout(tmp_path)

    exit_code = audit_cli.main(
        [
            "--sample-root",
            str(layout.sample_root),
            "--upstream-run-root",
            str(layout.upstream_run_root),
            "--lm2011-output-dir",
            str(layout.lm2011_output_dir),
            "--finbert-output-root",
            str(layout.finbert_output_root),
            "--output-dir",
            str(layout.output_root),
        ]
    )

    assert exit_code == 0
    assert (layout.output_root / "phase0_validation_report.md").exists()
    manifest = json.loads((layout.output_root / "audit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["packet_statuses"]["A"] == audit.STATUS_COMPLETED_WITH_WARNINGS
    assert manifest["packet_statuses"]["D"] == audit.STATUS_COMPLETED_WITH_WARNINGS


def test_missing_artifacts_block_optional_packets_and_fail_packet_a(tmp_path: Path) -> None:
    layout = _build_sample_audit_layout(tmp_path)
    layout.sample_backbone_path.unlink()
    layout.event_panel_path.unlink()
    for path in layout.finbert_run_dir.iterdir():
        if path.name != "run_manifest.json":
            path.unlink()

    artifacts = _run_audit(layout)

    assert artifacts.packet_statuses["A"] == audit.STATUS_FAILED_VALIDATION
    assert artifacts.packet_statuses["B"] == audit.STATUS_BLOCKED_MISSING_INPUT
    assert artifacts.packet_statuses["C"] == audit.STATUS_BLOCKED_MISSING_INPUT
    assert artifacts.packet_statuses["D"] == audit.STATUS_BLOCKED_MISSING_INPUT
