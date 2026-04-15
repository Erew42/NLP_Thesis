from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import BenchmarkItemSpec
from thesis_pkg.benchmarking.contracts import FinbertSectionUniverseConfig
from thesis_pkg.benchmarking.finbert_dataset import load_eligible_section_universe
from thesis_pkg.benchmarking.finbert_dataset import section_universe_contract_payload
from thesis_pkg.benchmarking.manifest_contracts import json_sha256
from thesis_pkg.benchmarking.manifest_contracts import normalize_contract_path
from thesis_pkg.benchmarking.manifest_contracts import resolve_manifest_path
from thesis_pkg.benchmarking.manifest_contracts import stable_string_fingerprint
from thesis_pkg.core.ccm.lm2011 import attach_eligible_quarterly_accounting
from thesis_pkg.core.ccm.lm2011 import normalize_lm2011_form_value
from thesis_pkg.core.sec.extraction import extract_filing_items
from thesis_pkg.core.sec.extraction import _strip_edgar_metadata
from thesis_pkg.core.sec.lm2011_dictionary import load_lm2011_master_dictionary_words
from thesis_pkg.core.sec.lm2011_cleaning import _apply_lm2011_paper_cleaning
from thesis_pkg.core.sec.lm2011_text import tokenize_lm2011_text
from thesis_pkg.pipelines.lm2011_pipeline import _attach_pre_filing_price_and_prior_month_price
from thesis_pkg.pipelines.refinitiv.analyst import select_refinitiv_lm2011_doc_analyst_inputs


PACKET_CHOICES: tuple[str, ...] = ("A", "B", "C", "D")
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STATUS_BLOCKED_MISSING_INPUT = "blocked_missing_input"
STATUS_FAILED_VALIDATION = "failed_validation"

FULL_10K_THRESHOLD = 2_000
MDA_THRESHOLD = 250
PARQUET_COMPRESSION = "zstd"

_LEGACY_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_TARGET_ITEM_IDS: tuple[str, ...] = ("1", "1A", "7")
_RAW_FORM_10K_ONLY: tuple[str, ...] = ("10-K", "10-K405")
_REPRESENTATIVE_TERMS: tuple[tuple[str, str], ...] = (
    ("negative", "loss"),
    ("positive", "gain"),
    ("uncertainty", "uncertain"),
    ("litigious", "lawsuit"),
    ("modal_strong", "must"),
    ("modal_weak", "may"),
)
_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "sec_header",
        re.compile(
            r"(?is)<sec-header|conformed submission type|public document count|accession number:",
        ),
    ),
    ("html", re.compile(r"(?is)<html|<div\b|<body\b|<p\b|</html>|</div>|</body>|</p>")),
    ("table", re.compile(r"(?is)<table\b|</table>|<tr\b|</tr>|<td\b|</td>")),
    (
        "exhibit",
        re.compile(r"(?is)</?ex-[a-z0-9][a-z0-9.-]*>|exhibit index|index to exhibits|exhibit\s+\d{1,3}[a-z]?(?:\.\d+)?"),
    ),
)
_EXTRACTION_ITEM_LABELS: dict[str, str] = {
    "1": "item_1",
    "1A": "item_1a",
    "7": "item_7",
}


@dataclass(frozen=True)
class Phase0ValidationAuditConfig:
    sample_root: Path
    upstream_run_root: Path | None = None
    lm2011_output_dir: Path | None = None
    finbert_output_root: Path | None = None
    output_root: Path | None = None
    packets: tuple[str, ...] = PACKET_CHOICES
    year_filter: tuple[int, ...] | None = None
    finbert_run_dir: Path | None = None
    max_example_rows: int = 25
    snippet_char_limit: int = 400
    regime_compare_doc_limit: int = 100

    def __post_init__(self) -> None:
        normalized_packets = tuple(dict.fromkeys(packet.upper() for packet in self.packets))
        invalid_packets = sorted(set(normalized_packets) - set(PACKET_CHOICES))
        if invalid_packets:
            raise ValueError(f"Unsupported packets: {invalid_packets}")
        object.__setattr__(self, "packets", normalized_packets)
        object.__setattr__(self, "year_filter", _normalize_year_filter(self.year_filter))
        if self.max_example_rows <= 0:
            raise ValueError("max_example_rows must be positive.")
        if self.snippet_char_limit <= 0:
            raise ValueError("snippet_char_limit must be positive.")
        if self.regime_compare_doc_limit <= 0:
            raise ValueError("regime_compare_doc_limit must be positive.")


@dataclass(frozen=True)
class Phase0ValidationAuditArtifacts:
    report_path: Path
    manifest_path: Path
    packet_output_paths: dict[str, dict[str, dict[str, str]]]
    packet_statuses: dict[str, str]
    blocked_reasons: dict[str, str]


@dataclass(frozen=True)
class _ResolvedAuditPaths:
    sample_root: Path
    upstream_run_root: Path
    lm2011_output_dir: Path
    finbert_output_root: Path
    output_root: Path
    year_merged_dir: Path
    items_analysis_dir: Path
    additional_data_dir: Path
    sample_backbone_path: Path
    text_features_full_10k_path: Path
    text_features_mda_path: Path
    event_panel_path: Path
    sue_panel_path: Path
    quarterly_accounting_panel_path: Path
    table_iv_results_path: Path
    table_viii_results_path: Path
    daily_panel_path: Path
    doc_analyst_selected_path: Path
    finbert_run_dir: Path | None


@dataclass
class _PacketResult:
    status: str
    output_paths: dict[str, dict[str, str]]
    summary: dict[str, Any]
    blocked_reason: str | None = None
    warnings: list[str] | None = None
    top_findings: list[str] | None = None


def _resolve_optional_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def run_phase0_validation_audit(
    cfg: Phase0ValidationAuditConfig,
) -> Phase0ValidationAuditArtifacts:
    paths = _resolve_paths(cfg)
    paths.output_root.mkdir(parents=True, exist_ok=True)

    packet_results: dict[str, _PacketResult] = {}
    for packet in cfg.packets:
        if packet == "A":
            packet_results[packet] = _run_packet_a(cfg, paths)
        elif packet == "B":
            packet_results[packet] = _run_packet_b(cfg, paths)
        elif packet == "C":
            packet_results[packet] = _run_packet_c(cfg, paths)
        elif packet == "D":
            packet_results[packet] = _run_packet_d(cfg, paths)

    report_path = _write_report(cfg, paths, packet_results)
    manifest_path = paths.output_root / "audit_manifest.json"
    manifest_payload = {
        "runner_name": "lm2011_phase0_validation_audit",
        "generated_at_utc": _utc_timestamp(),
        "config": {
            "sample_root": str(paths.sample_root),
            "upstream_run_root": str(paths.upstream_run_root),
            "lm2011_output_dir": str(paths.lm2011_output_dir),
            "finbert_output_root": str(paths.finbert_output_root),
            "output_root": str(paths.output_root),
            "packets": list(cfg.packets),
            "year_filter": list(cfg.year_filter) if cfg.year_filter is not None else None,
            "finbert_run_dir": str(paths.finbert_run_dir) if paths.finbert_run_dir is not None else None,
            "max_example_rows": cfg.max_example_rows,
            "snippet_char_limit": cfg.snippet_char_limit,
            "regime_compare_doc_limit": cfg.regime_compare_doc_limit,
        },
        "packet_statuses": {packet: result.status for packet, result in packet_results.items()},
        "blocked_reasons": {
            packet: result.blocked_reason
            for packet, result in packet_results.items()
            if result.blocked_reason is not None
        },
        "packet_output_paths": {
            packet: result.output_paths
            for packet, result in packet_results.items()
        },
        "packet_summaries": {
            packet: result.summary
            for packet, result in packet_results.items()
        },
        "report_path": str(report_path),
    }
    _write_json(manifest_path, manifest_payload)

    return Phase0ValidationAuditArtifacts(
        report_path=report_path,
        manifest_path=manifest_path,
        packet_output_paths={packet: result.output_paths for packet, result in packet_results.items()},
        packet_statuses={packet: result.status for packet, result in packet_results.items()},
        blocked_reasons={
            packet: result.blocked_reason
            for packet, result in packet_results.items()
            if result.blocked_reason is not None
        },
    )


def _run_packet_a(
    cfg: Phase0ValidationAuditConfig,
    paths: _ResolvedAuditPaths,
) -> _PacketResult:
    if not paths.sample_backbone_path.exists():
        return _PacketResult(
            status=STATUS_FAILED_VALIDATION,
            output_paths={},
            summary={},
            blocked_reason=f"Missing sampled backbone artifact: {paths.sample_backbone_path}",
            top_findings=["Packet A failed because the sampled accepted-doc universe could not be resolved."],
        )
    if not paths.year_merged_dir.exists():
        return _PacketResult(
            status=STATUS_FAILED_VALIDATION,
            output_paths={},
            summary={},
            blocked_reason=f"Missing sampled year_merged directory: {paths.year_merged_dir}",
            top_findings=["Packet A failed because the sampled full-text shards are unavailable."],
        )

    master_dictionary = _load_master_dictionary_tokens(paths.additional_data_dir)
    accepted_backbone = _load_backbone_doc_years(paths.sample_backbone_path, cfg.year_filter)
    if accepted_backbone.is_empty():
        return _PacketResult(
            status=STATUS_FAILED_VALIDATION,
            output_paths={},
            summary={},
            blocked_reason="Accepted sampled backbone resolved to zero documents.",
            top_findings=["Packet A failed because the accepted benchmark universe is empty."],
        )

    full_rows = _packet_a_full_text_rows(cfg, paths, accepted_backbone, master_dictionary)
    mda_rows = _packet_a_mda_rows(cfg, paths, accepted_backbone, master_dictionary)
    combined_rows = [*full_rows, *mda_rows]
    delta_summary = _packet_a_delta_summary(combined_rows)
    summary_df = _packet_a_summary_frame(combined_rows)
    threshold_df = _packet_a_threshold_frame(combined_rows)
    examples_df = _packet_a_examples_frame(combined_rows, cfg.max_example_rows)
    output_paths = {
        "summary": _write_table_pair(paths.output_root, "packet_a_summary", summary_df),
        "threshold_diff": _write_table_pair(paths.output_root, "packet_a_threshold_diff", threshold_df),
        "strip_comparison": _write_table_pair(
            paths.output_root,
            "packet_a_strip_comparison",
            _packet_a_strip_comparison_frame(combined_rows),
        ),
        "examples": _write_table_pair(paths.output_root, "packet_a_examples", examples_df),
    }

    full_dirty_count = _count_true(full_rows, "any_marker")
    full_post_strip_dirty_count = _count_true(full_rows, "edgar_stripped_any_marker")
    full_post_strip_html_count = _count_true(full_rows, "edgar_stripped_has_html_marker")
    full_post_paper_dirty_count = _count_true(full_rows, "paper_cleaned_any_marker")
    full_post_paper_exhibit_count = _count_true(full_rows, "paper_cleaned_has_exhibit_marker")
    full_truncated_count = sum(1 for row in full_rows if row.get("paper_cleaned_cut_reason") != "no_tail_anchor")
    full_flip_count = _count_true(
        [row for row in combined_rows if row["text_scope"] == "full_10k"],
        "threshold_flip",
    )
    full_paper_flip_count = _count_true(
        [row for row in combined_rows if row["text_scope"] == "full_10k"],
        "paper_cleaned_threshold_flip",
    )
    mda_flip_count = _count_true(
        [row for row in combined_rows if row["text_scope"] == "mda"],
        "threshold_flip",
    )
    findings = [
        f"Accepted sampled full 10-K docs with raw-marker hits: {full_dirty_count}",
        (
            "Accepted sampled full 10-K docs with remaining markers after "
            f"_strip_edgar_metadata(): {full_post_strip_dirty_count}"
        ),
        (
            "Accepted sampled full 10-K docs with remaining HTML markers after "
            f"_strip_edgar_metadata(): {full_post_strip_html_count}"
        ),
        (
            "Accepted sampled full 10-K docs with remaining markers after "
            f"LM2011 paper cleaning: {full_post_paper_dirty_count}"
        ),
        (
            "Accepted sampled full 10-K docs with remaining exhibit markers after "
            f"LM2011 paper cleaning: {full_post_paper_exhibit_count}"
        ),
        f"Accepted sampled full 10-K docs truncated by exhibit stripping: {full_truncated_count}",
        f"Accepted sampled full 10-K threshold flips at 2,000 words: {full_flip_count}",
        f"Accepted sampled full 10-K threshold flips after LM2011 paper cleaning: {full_paper_flip_count}",
    ]
    if delta_summary.get("full_10k_doc_count"):
        findings.extend(
            [
                _format_delta_finding(
                    "full 10-K",
                    "current-vs-Appendix",
                    int(delta_summary["full_10k_current_vs_appendix_doc_count"]),
                    int(delta_summary["full_10k_doc_count"]),
                    float(delta_summary["full_10k_mean_current_minus_appendix_token_delta"]),
                ),
                _format_delta_finding(
                    "full 10-K",
                    "Appendix-vs-recognized",
                    int(delta_summary["full_10k_appendix_vs_recognized_doc_count"]),
                    int(delta_summary["full_10k_doc_count"]),
                    float(delta_summary["full_10k_mean_appendix_minus_recognized_token_delta"]),
                ),
            ]
        )
    if mda_rows:
        findings.append(f"Accepted sampled MD&A threshold flips at 250 words: {mda_flip_count}")
        findings.extend(
            [
                _format_delta_finding(
                    "MD&A",
                    "current-vs-Appendix",
                    int(delta_summary["mda_current_vs_appendix_doc_count"]),
                    int(delta_summary["mda_doc_count"]),
                    float(delta_summary["mda_mean_current_minus_appendix_token_delta"]),
                ),
                _format_delta_finding(
                    "MD&A",
                    "Appendix-vs-recognized",
                    int(delta_summary["mda_appendix_vs_recognized_doc_count"]),
                    int(delta_summary["mda_doc_count"]),
                    float(delta_summary["mda_mean_appendix_minus_recognized_token_delta"]),
                ),
            ]
        )

    status = (
        STATUS_COMPLETED
        if not full_dirty_count and not full_flip_count and not mda_flip_count
        else STATUS_COMPLETED_WITH_WARNINGS
    )
    return _PacketResult(
        status=status,
        output_paths=output_paths,
        summary={
            "accepted_doc_count": int(accepted_backbone.height),
            "dirty_full_10k_doc_count": int(full_dirty_count),
            "post_strip_dirty_full_10k_doc_count": int(full_post_strip_dirty_count),
            "post_strip_html_full_10k_doc_count": int(full_post_strip_html_count),
            "post_paper_cleaning_dirty_full_10k_doc_count": int(full_post_paper_dirty_count),
            "post_paper_cleaning_exhibit_full_10k_doc_count": int(full_post_paper_exhibit_count),
            "paper_cleaned_truncated_full_10k_doc_count": int(full_truncated_count),
            "full_10k_threshold_flip_count": int(full_flip_count),
            "paper_cleaned_full_10k_threshold_flip_count": int(full_paper_flip_count),
            "mda_threshold_flip_count": int(mda_flip_count),
            **delta_summary,
        },
        warnings=[
            "Packet A uses the sampled accepted-doc universe and does not claim full-root coverage.",
        ],
        top_findings=findings,
    )


def _run_packet_b(
    cfg: Phase0ValidationAuditConfig,
    paths: _ResolvedAuditPaths,
) -> _PacketResult:
    warnings: list[str] = []
    output_paths: dict[str, dict[str, str]] = {}
    summary: dict[str, Any] = {}
    findings: list[str] = []
    statuses: list[str] = []

    if not paths.sample_backbone_path.exists() or not paths.year_merged_dir.exists():
        return _PacketResult(
            status=STATUS_BLOCKED_MISSING_INPUT,
            output_paths={},
            summary={},
            blocked_reason="Packet B requires sampled backbone and year_merged artifacts.",
            top_findings=["Packet B was blocked because the sampled corpus inputs are incomplete."],
        )

    year_paths = _resolve_year_paths(paths.year_merged_dir, cfg.year_filter)
    accepted_backbone = _load_backbone_doc_years(paths.sample_backbone_path, cfg.year_filter)
    feature_doc_df = (
        _load_feature_doc_years(paths.text_features_full_10k_path, cfg.year_filter)
        if paths.text_features_full_10k_path.exists()
        else pl.DataFrame(schema={"doc_id": pl.Utf8, "filing_year": pl.Int32})
    )

    corpus_summary_df = _packet_b_corpus_summary(paths, accepted_backbone, cfg.year_filter)
    output_paths["corpus_summary"] = _write_table_pair(paths.output_root, "packet_b_corpus_summary", corpus_summary_df)
    summary["corpus_summary_rows"] = int(corpus_summary_df.height)
    statuses.append(STATUS_COMPLETED)

    form_mix_df = _packet_b_form_mix(paths, cfg.year_filter)
    output_paths["form_mix"] = _write_table_pair(paths.output_root, "packet_b_form_mix", form_mix_df)

    if paths.text_features_full_10k_path.exists():
        term_df = _packet_b_representative_terms(year_paths, accepted_backbone, feature_doc_df)
        output_paths["representative_terms"] = _write_table_pair(
            paths.output_root,
            "packet_b_representative_terms",
            term_df,
        )
        statuses.append(STATUS_COMPLETED)
        broad_docs = corpus_summary_df.filter(pl.col("corpus_label") == "year_merged_broad").item(0, "doc_count")
        feature_docs = corpus_summary_df.filter(pl.col("corpus_label") == "text_features_full_10k").item(0, "doc_count")
        findings.append(
            f"Current full-text feature artifact doc count = {feature_docs}, broad sampled text corpus doc count = {broad_docs}."
        )
    else:
        warnings.append("Current sampled lm2011_text_features_full_10k.parquet is missing; B1 representative-term audit was blocked.")
        statuses.append(STATUS_BLOCKED_MISSING_INPUT)

    if paths.event_panel_path.exists():
        attrition_df = _packet_b_event_attrition(paths, accepted_backbone, cfg.year_filter)
        output_paths["event_attrition"] = _write_table_pair(
            paths.output_root,
            "packet_b_event_attrition",
            attrition_df,
        )
        lost_docs = int(attrition_df.filter(pl.col("filing_year").is_null()).item(0, "lost_doc_count"))
        findings.append(f"Sampled backbone docs lost before event_panel: {lost_docs}")
        summary["event_panel_attrition_count"] = lost_docs
        statuses.append(STATUS_COMPLETED)
    else:
        warnings.append("Current sampled lm2011_event_panel.parquet is missing; B2 strategy-inheritance audit was blocked.")
        statuses.append(STATUS_BLOCKED_MISSING_INPUT)

    status = STATUS_COMPLETED if all(item == STATUS_COMPLETED for item in statuses) else STATUS_COMPLETED_WITH_WARNINGS
    return _PacketResult(
        status=status,
        output_paths=output_paths,
        summary=summary,
        warnings=warnings or None,
        top_findings=findings or None,
    )


def _run_packet_c(
    cfg: Phase0ValidationAuditConfig,
    paths: _ResolvedAuditPaths,
) -> _PacketResult:
    required_paths = (
        paths.event_panel_path,
        paths.sue_panel_path,
        paths.quarterly_accounting_panel_path,
        paths.daily_panel_path,
        paths.doc_analyst_selected_path,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return _PacketResult(
            status=STATUS_BLOCKED_MISSING_INPUT,
            output_paths={},
            summary={},
            blocked_reason=f"Packet C missing required sampled artifacts: {missing}",
            top_findings=["Packet C was blocked because sampled LM2011 inputs for denominator auditing are incomplete."],
        )

    units_df = _packet_c_units_audit(paths)
    denominator_df = _packet_c_denominator_audit(cfg, paths)
    classification_df = _packet_c_output_classification(paths)

    output_paths = {
        "units_audit": _write_table_pair(paths.output_root, "packet_c_units_audit", units_df),
        "denominator_candidates": _write_table_pair(
            paths.output_root,
            "packet_c_denominator_candidates",
            denominator_df,
        ),
        "output_classification": _write_table_pair(
            paths.output_root,
            "packet_c_output_classification",
            classification_df,
        ),
    }
    nonnull_abs_diff = denominator_df.get_column("abs_diff_sue").drop_nulls() if denominator_df.height else pl.Series([], dtype=pl.Float64)
    mean_abs_sue_gap = float(nonnull_abs_diff.mean()) if nonnull_abs_diff.len() > 0 else None
    findings = [
        f"Clearly internal-unit sampled columns identified: {int(units_df.filter(pl.col('classification') == 'clearly_internal_unit').height)}",
    ]
    if mean_abs_sue_gap is not None:
        findings.append(f"Mean absolute SUE difference between filing and announcement anchors: {mean_abs_sue_gap:.6f}")
    return _PacketResult(
        status=STATUS_COMPLETED,
        output_paths=output_paths,
        summary={
            "units_audit_rows": int(units_df.height),
            "denominator_rows": int(denominator_df.height),
            "mean_abs_sue_gap": mean_abs_sue_gap,
        },
        top_findings=findings,
    )


def _run_packet_d(
    cfg: Phase0ValidationAuditConfig,
    paths: _ResolvedAuditPaths,
) -> _PacketResult:
    finbert_run_dir = paths.finbert_run_dir or _resolve_latest_valid_finbert_run(
        paths.finbert_output_root,
        sampled_items_analysis_dir=paths.items_analysis_dir,
        sample_backbone_path=paths.sample_backbone_path,
        requested_year_filter=cfg.year_filter,
    )
    if finbert_run_dir is None:
        return _PacketResult(
            status=STATUS_BLOCKED_MISSING_INPUT,
            output_paths={},
            summary={},
            blocked_reason="No valid sampled FinBERT item-analysis run was found.",
            top_findings=["Packet D was blocked because no sampled FinBERT run with item features and coverage was available."],
        )
    if not paths.items_analysis_dir.exists() or not paths.sample_backbone_path.exists():
        return _PacketResult(
            status=STATUS_BLOCKED_MISSING_INPUT,
            output_paths={},
            summary={},
            blocked_reason="Packet D requires sampled items_analysis shards and the sampled backbone artifact.",
            top_findings=["Packet D was blocked because the sampled item universe could not be resolved."],
        )

    run_manifest_path = finbert_run_dir / "run_manifest.json"
    item_features_long_path = finbert_run_dir / "item_features_long.parquet"
    coverage_report_path = finbert_run_dir / "coverage_report.parquet"
    if not (run_manifest_path.exists() and item_features_long_path.exists() and coverage_report_path.exists()):
        return _PacketResult(
            status=STATUS_BLOCKED_MISSING_INPUT,
            output_paths={},
            summary={},
            blocked_reason=f"FinBERT run is incomplete: {finbert_run_dir}",
            top_findings=["Packet D was blocked because the selected sampled FinBERT run is incomplete."],
        )

    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    section_cfg = _section_universe_from_manifest(paths.items_analysis_dir, run_manifest)
    effective_year_filter = cfg.year_filter or _manifest_year_filter(run_manifest)
    selected_year_paths = _resolve_year_paths(paths.items_analysis_dir, effective_year_filter)

    dictionary_universe_df = load_eligible_section_universe(
        section_cfg,
        year_paths=selected_year_paths,
        target_doc_universe_path=paths.sample_backbone_path,
    ).collect()
    reconciliation_df = _packet_d_reconciliation_table(dictionary_universe_df, item_features_long_path)
    removal_df = _packet_d_removal_waterfall(section_cfg, selected_year_paths, paths.sample_backbone_path)
    coverage_df = _packet_d_coverage_reconciliation(
        section_cfg,
        selected_year_paths,
        run_manifest,
        run_manifest_path,
        item_features_long_path,
        sample_backbone_path=paths.sample_backbone_path,
    )
    regime_df = _packet_d_extraction_regime_comparison(
        cfg,
        paths,
        dictionary_universe_df,
        effective_year_filter,
    )

    output_paths = {
        "reconciliation": _write_table_pair(paths.output_root, "packet_d_reconciliation", reconciliation_df),
        "removal_waterfall": _write_table_pair(paths.output_root, "packet_d_removal_waterfall", removal_df),
        "coverage_reconciliation": _write_table_pair(
            paths.output_root,
            "packet_d_coverage_reconciliation",
            coverage_df,
        ),
        "regime_comparison": _write_table_pair(paths.output_root, "packet_d_regime_comparison", regime_df),
    }
    both_rows = reconciliation_df.filter(pl.col("classification") == "both")
    overlap = int(both_rows["row_count"].sum()) if both_rows.height else 0
    findings = [
        f"Harmonized dictionary/FinBERT overlap rows: {overlap}",
        (
            f"Reported FinBERT denominator = {int(coverage_df.item(0, 'reported_backbone_doc_count'))}, "
            f"actual filtered denominator = {int(coverage_df.item(0, 'actual_filtered_doc_count')) if coverage_df.item(0, 'actual_filtered_doc_count') is not None else 'n/a'}"
        ),
        (
            "FinBERT backbone matches sampled LM backbone: "
            f"{bool(coverage_df.item(0, 'finbert_backbone_matches_sample_backbone'))}"
        ),
    ]
    warnings: list[str] = []
    backbone_matches = bool(coverage_df.item(0, "finbert_backbone_matches_sample_backbone"))
    denominator_gap = coverage_df.item(0, "denominator_gap")
    sample_backbone_denominator_gap = coverage_df.item(0, "sample_backbone_denominator_gap")
    if not backbone_matches:
        warnings.append("FinBERT run backbone path does not match the sampled LM2011 backbone path.")
    if denominator_gap is not None and int(denominator_gap) != 0:
        warnings.append("FinBERT reported denominator differs from actual filtered item-universe docs.")
    if sample_backbone_denominator_gap is not None and int(sample_backbone_denominator_gap) != 0:
        warnings.append("FinBERT reported denominator differs from sampled LM2011 backbone docs.")
    return _PacketResult(
        status=STATUS_COMPLETED if not warnings else STATUS_COMPLETED_WITH_WARNINGS,
        output_paths=output_paths,
        summary={
            "dictionary_universe_rows": int(dictionary_universe_df.height),
            "reconciliation_rows": int(reconciliation_df.height),
        },
        warnings=warnings or None,
        top_findings=findings,
    )


def _resolve_paths(cfg: Phase0ValidationAuditConfig) -> _ResolvedAuditPaths:
    sample_root = Path(cfg.sample_root).resolve()
    repo_root = sample_root.parent.parent
    upstream_run_root = (
        Path(cfg.upstream_run_root).resolve()
        if cfg.upstream_run_root is not None
        else sample_root / "results" / "sec_ccm_unified_runner" / "local_sample"
    )
    lm2011_output_dir = (
        Path(cfg.lm2011_output_dir).resolve()
        if cfg.lm2011_output_dir is not None
        else sample_root / "results" / "lm2011_sample_post_refinitiv_runner"
    )
    finbert_output_root = (
        Path(cfg.finbert_output_root).resolve()
        if cfg.finbert_output_root is not None
        else sample_root / "results" / "finbert_item_analysis_runner"
    )
    output_root = (
        Path(cfg.output_root).resolve()
        if cfg.output_root is not None
        else repo_root / "reports" / f"lm2011_phase0_validation_sample_{dt.date.today():%Y%m%d}"
    )
    finbert_run_dir = Path(cfg.finbert_run_dir).resolve() if cfg.finbert_run_dir is not None else None
    additional_data_dir = (
        _resolve_optional_existing_path(
            sample_root / "LM2011_additional_data",
            sample_root.parent / "LM2011_additional_data",
        )
        or sample_root.parent / "LM2011_additional_data"
    )
    daily_panel_path = (
        _resolve_optional_existing_path(
            sample_root / "derived_data" / "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet",
            sample_root / "derived_data" / "final_flagged_data_compdesc_added.parquet",
        )
        or sample_root / "derived_data" / "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet"
    )
    return _ResolvedAuditPaths(
        sample_root=sample_root,
        upstream_run_root=upstream_run_root,
        lm2011_output_dir=lm2011_output_dir,
        finbert_output_root=finbert_output_root,
        output_root=output_root,
        year_merged_dir=sample_root / "year_merged",
        items_analysis_dir=upstream_run_root / "items_analysis",
        additional_data_dir=additional_data_dir,
        sample_backbone_path=lm2011_output_dir / "lm2011_sample_backbone.parquet",
        text_features_full_10k_path=lm2011_output_dir / "lm2011_text_features_full_10k.parquet",
        text_features_mda_path=lm2011_output_dir / "lm2011_text_features_mda.parquet",
        event_panel_path=lm2011_output_dir / "lm2011_event_panel.parquet",
        sue_panel_path=lm2011_output_dir / "lm2011_sue_panel.parquet",
        quarterly_accounting_panel_path=lm2011_output_dir / "lm2011_quarterly_accounting_panel.parquet",
        table_iv_results_path=lm2011_output_dir / "lm2011_table_iv_results.parquet",
        table_viii_results_path=lm2011_output_dir / "lm2011_table_viii_results.parquet",
        daily_panel_path=daily_panel_path,
        doc_analyst_selected_path=(
            upstream_run_root / "refinitiv_doc_analyst_lm2011" / "refinitiv_doc_analyst_selected.parquet"
        ),
        finbert_run_dir=finbert_run_dir,
    )


def _normalize_year_filter(year_filter: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if year_filter is None:
        return None
    normalized = tuple(sorted({int(year) for year in year_filter}))
    for year in normalized:
        if year < 0:
            raise ValueError(f"year_filter values must be positive integers, got {year!r}")
    return normalized


def _resolve_year_paths(directory: Path, year_filter: tuple[int, ...] | None) -> list[Path]:
    year_paths = sorted(path for path in directory.glob("*.parquet") if path.stem.isdigit() and len(path.stem) == 4)
    if year_filter is None:
        return year_paths
    requested = set(year_filter)
    selected = [path for path in year_paths if int(path.stem) in requested]
    missing_years = sorted(requested - {int(path.stem) for path in selected})
    if missing_years:
        raise FileNotFoundError(f"Requested filing years were not found in {directory}: {missing_years}")
    return selected


def _year_filter_expr(column_name: str, year_filter: tuple[int, ...] | None, *, use_date_year: bool = False) -> pl.Expr:
    if year_filter is None:
        return pl.lit(True)
    column = pl.col(column_name).dt.year() if use_date_year else pl.col(column_name)
    return column.cast(pl.Int32, strict=False).is_in(year_filter)


def _previous_month_end(value: dt.date) -> dt.date:
    first_of_month = value.replace(day=1)
    return first_of_month - dt.timedelta(days=1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_table_pair(
    output_root: Path,
    stem: str,
    df: pl.DataFrame,
) -> dict[str, str]:
    parquet_path = output_root / f"{stem}.parquet"
    csv_path = output_root / f"{stem}.csv"
    df.write_parquet(parquet_path, compression=PARQUET_COMPRESSION)
    df.write_csv(csv_path)
    return {
        "parquet": str(parquet_path),
        "csv": str(csv_path),
    }


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _write_report(
    cfg: Phase0ValidationAuditConfig,
    paths: _ResolvedAuditPaths,
    packet_results: dict[str, _PacketResult],
) -> Path:
    report_path = paths.output_root / "phase0_validation_report.md"
    lines: list[str] = []
    lines.append("# Phase 0 LM2011 Validation Audit Report")
    lines.append("")
    lines.append(f"- Generated at: `{_utc_timestamp()}`")
    lines.append(f"- Sample root: `{paths.sample_root}`")
    lines.append(f"- LM2011 output dir: `{paths.lm2011_output_dir}`")
    lines.append(f"- FinBERT output root: `{paths.finbert_output_root}`")
    if paths.finbert_run_dir is not None:
        lines.append(f"- Selected FinBERT run: `{paths.finbert_run_dir}`")
    lines.append("")
    lines.append("## Packet Statuses")
    lines.append("")
    status_rows = [
        {"packet": packet, "status": result.status, "blocked_reason": result.blocked_reason or ""}
        for packet, result in packet_results.items()
    ]
    lines.append(_markdown_table(pl.DataFrame(status_rows)))
    lines.append("")
    lines.append("## Top Findings")
    lines.append("")
    for packet in PACKET_CHOICES:
        if packet not in packet_results:
            continue
        result = packet_results[packet]
        lines.append(f"### Packet {packet}")
        lines.append("")
        if result.top_findings:
            for finding in result.top_findings:
                lines.append(f"- {finding}")
        else:
            lines.append("- No additional findings recorded.")
        if result.warnings:
            for warning in result.warnings:
                lines.append(f"- Warning: {warning}")
        lines.append("")
    lines.append("## Packet Details")
    lines.append("")
    for packet in PACKET_CHOICES:
        if packet not in packet_results:
            continue
        result = packet_results[packet]
        lines.append(f"### Packet {packet}")
        lines.append("")
        lines.append(f"- Status: `{result.status}`")
        if result.blocked_reason:
            lines.append(f"- Blocked reason: {result.blocked_reason}")
        if result.summary:
            lines.append("- Summary:")
            for key, value in sorted(result.summary.items()):
                lines.append(f"  - `{key}`: `{value}`")
        if result.output_paths:
            lines.append("- Machine-readable outputs:")
            for label, paths_dict in sorted(result.output_paths.items()):
                lines.append(
                    f"  - `{label}`: `{paths_dict.get('parquet', '')}` / `{paths_dict.get('csv', '')}`"
                )
        example_table_path = result.output_paths.get("examples", {}).get("parquet") if result.output_paths else None
        if example_table_path:
            example_df = pl.read_parquet(example_table_path)
            if example_df.height:
                lines.append("")
                lines.append("Example rows:")
                lines.append("")
                lines.append(_markdown_table(example_df.head(cfg.max_example_rows)))
        lines.append("")
    lines.append("## Suggested Next Remediation Target")
    lines.append("")
    lines.append(f"- {_suggested_next_target(packet_results)}")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _load_master_dictionary_tokens(additional_data_dir: Path) -> frozenset[str]:
    words, _, _ = load_lm2011_master_dictionary_words(additional_data_dir)
    return frozenset(
        token
        for word in words
        for token in tokenize_lm2011_text(word)
    )


def _legacy_tokenize_lm2011_text(text: str | None) -> list[str]:
    if text is None:
        return []
    return [token.casefold() for token in _LEGACY_TOKEN_RE.findall(text)]


def _load_backbone_doc_years(
    sample_backbone_path: Path,
    year_filter: tuple[int, ...] | None,
) -> pl.DataFrame:
    lf = pl.scan_parquet(sample_backbone_path)
    schema = lf.collect_schema()
    if "filing_date" in schema:
        year_expr = pl.col("filing_date").cast(pl.Date, strict=False).dt.year().cast(pl.Int32).alias("filing_year")
    elif "filing_year" in schema:
        year_expr = pl.col("filing_year").cast(pl.Int32, strict=False).alias("filing_year")
    else:
        year_expr = pl.lit(None, dtype=pl.Int32).alias("filing_year")

    out = (
        lf.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
            year_expr,
        )
        .unique(maintain_order=True)
    )
    if year_filter is not None and ("filing_date" in schema or "filing_year" in schema):
        out = out.filter(_year_filter_expr("filing_year", year_filter))
    return out.collect()


def _doc_universe_fingerprint(df: pl.DataFrame) -> str:
    return stable_string_fingerprint(df["doc_id"].to_list() if df.height else [])


def _expected_packet_d_contract(
    *,
    section_cfg: FinbertSectionUniverseConfig,
    sample_backbone_path: Path,
    effective_year_filter: tuple[int, ...] | None,
) -> dict[str, Any]:
    sample_backbone_df = _load_backbone_doc_years(sample_backbone_path, effective_year_filter)
    return {
        "effective_year_filter": list(effective_year_filter) if effective_year_filter is not None else None,
        "accepted_universe_contract_fingerprint": json_sha256(
            section_universe_contract_payload(
                section_cfg,
                target_doc_universe_path=sample_backbone_path,
            )
        ),
        "normalized_relative_backbone_path": normalize_contract_path(
            sample_backbone_path,
            base_path=Path.cwd(),
        ),
        "filtered_backbone_doc_universe_fingerprint": _doc_universe_fingerprint(sample_backbone_df),
    }


def _candidate_backbone_contract(
    run_manifest: dict[str, Any],
    *,
    run_manifest_path: Path | None = None,
    effective_year_filter: tuple[int, ...] | None,
    section_cfg: FinbertSectionUniverseConfig | None = None,
    sample_backbone_path: Path | None = None,
) -> dict[str, Any] | None:
    stored_contract = run_manifest.get("backbone_contract")
    if isinstance(stored_contract, dict):
        return stored_contract

    backbone_path_raw = run_manifest.get("backbone_path")
    if not backbone_path_raw:
        backbone_path_raw = (run_manifest.get("nonportable_diagnostics") or {}).get("backbone_path")
    if not backbone_path_raw:
        return None
    if run_manifest_path is not None:
        backbone_path = resolve_manifest_path(
            backbone_path_raw,
            manifest_path=run_manifest_path,
            path_semantics=run_manifest.get("path_semantics"),
        )
    else:
        backbone_path = Path(str(backbone_path_raw))
    if backbone_path is None:
        return None
    if not backbone_path.exists():
        return None
    manifest_source = run_manifest.get("source_sentence_dataset_manifest") or {}
    accepted_universe_contract = manifest_source.get("accepted_universe_contract")
    if accepted_universe_contract is None and section_cfg is not None and sample_backbone_path is not None:
        accepted_universe_contract = section_universe_contract_payload(
            section_cfg,
            target_doc_universe_path=sample_backbone_path,
        )
    return {
        "effective_year_filter": list(effective_year_filter) if effective_year_filter is not None else None,
        "accepted_universe_contract_fingerprint": (
            json_sha256(accepted_universe_contract)
            if accepted_universe_contract is not None
            else None
        ),
        "normalized_relative_backbone_path": normalize_contract_path(
            backbone_path,
            base_path=Path.cwd(),
        ),
    }


def _packet_d_contract_match(
    candidate_contract: dict[str, Any] | None,
    *,
    expected_contract: dict[str, Any],
) -> bool:
    if candidate_contract is None:
        return False
    if candidate_contract.get("effective_year_filter") != expected_contract["effective_year_filter"]:
        return False
    if (
        candidate_contract.get("accepted_universe_contract_fingerprint")
        != expected_contract["accepted_universe_contract_fingerprint"]
    ):
        return False
    candidate_filtered_fingerprint = candidate_contract.get("filtered_backbone_doc_universe_fingerprint")
    if candidate_filtered_fingerprint is not None:
        return candidate_filtered_fingerprint == expected_contract["filtered_backbone_doc_universe_fingerprint"]
    return (
        candidate_contract.get("normalized_relative_backbone_path")
        == expected_contract["normalized_relative_backbone_path"]
    )


def _load_feature_doc_years(
    feature_path: Path,
    year_filter: tuple[int, ...] | None,
) -> pl.DataFrame:
    return (
        pl.scan_parquet(feature_path)
        .select(
            pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
            pl.col("filing_date").dt.year().cast(pl.Int32).alias("filing_year"),
        )
        .filter(_year_filter_expr("filing_year", year_filter))
        .unique(maintain_order=True)
        .collect()
    )


def _load_doc_year_frame(path: Path, year_filter: tuple[int, ...] | None) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select(
            pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
            pl.col("filing_date").dt.year().cast(pl.Int32).alias("filing_year"),
        )
        .filter(_year_filter_expr("filing_year", year_filter))
        .unique(maintain_order=True)
        .collect()
    )


def _doc_year_groups(df: pl.DataFrame) -> dict[int, pl.DataFrame]:
    if df.is_empty():
        return {}
    groups: dict[int, pl.DataFrame] = {}
    for year_key, group in df.group_by("filing_year", maintain_order=True):
        filing_year = year_key[0] if isinstance(year_key, tuple) else year_key
        groups[int(filing_year)] = group.select("doc_id").unique(maintain_order=True)
    return groups


def _marker_flags_and_snippet(text: str | None, snippet_char_limit: int) -> tuple[dict[str, bool], str | None]:
    if text is None:
        return {name: False for name, _ in _MARKER_PATTERNS}, None
    flags: dict[str, bool] = {}
    snippet: str | None = None
    first_match_start: int | None = None
    for name, pattern in _MARKER_PATTERNS:
        match = pattern.search(text)
        flags[name] = match is not None
        if match is not None and first_match_start is None:
            first_match_start = match.start()
    if first_match_start is not None:
        start = max(0, first_match_start - 80)
        end = min(len(text), first_match_start + snippet_char_limit)
        snippet = _truncate_text(text[start:end], snippet_char_limit)
    return flags, snippet


def _truncate_text(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(limit - 3, 0)].rstrip()}..."


def _count_true(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _count_series(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _collect_form_counts_from_year_merged(
    year_merged_dir: Path,
    year_filter: tuple[int, ...] | None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for year_path in _resolve_year_paths(year_merged_dir, year_filter):
        year_df = (
            pl.scan_parquet(year_path)
            .select(pl.col("document_type_filename").cast(pl.Utf8, strict=False).alias("document_type_filename"))
            .collect()
        )
        forms = [
            normalize_lm2011_form_value(value, other_value="Other")
            for value in year_df.get_column("document_type_filename").drop_nulls().to_list()
        ]
        for key, value in _count_series(forms).items():
            counts[key] = counts.get(key, 0) + value
    return counts


def _form_count_rows(corpus_label: str, counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {
            "corpus_label": corpus_label,
            "normalized_form": key,
            "doc_count": int(value),
        }
        for key, value in sorted(counts.items())
    ]


def _update_term_counts(stats: dict[str, int], year_df: pl.DataFrame) -> None:
    if year_df.is_empty():
        return
    stats["doc_count"] += int(year_df.height)
    for text in year_df.get_column("full_text").to_list():
        token_set = set(tokenize_lm2011_text(text if isinstance(text, str) else None))
        for _, term in _REPRESENTATIVE_TERMS:
            if term in token_set:
                stats[term] += 1


def _parquet_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _markdown_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "_No rows._"
    headers = df.columns
    rows = df.fill_null("").rows()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _markdown_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def _suggested_next_target(packet_results: dict[str, _PacketResult]) -> str:
    packet_a = packet_results.get("A")
    if packet_a is not None and packet_a.status in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}:
        dirty = int(packet_a.summary.get("dirty_full_10k_doc_count") or 0)
        flips = int(packet_a.summary.get("full_10k_threshold_flip_count") or 0)
        mda_flips = int(packet_a.summary.get("mda_threshold_flip_count") or 0)
        if dirty or flips or mda_flips:
            return "Remediate `R-01` first: clean the accepted full 10-K text object; Packet A threshold flips now track current-vs-Appendix total-token screens while Appendix-vs-recognized deltas remain diagnostic."

    packet_b = packet_results.get("B")
    if packet_b is not None and packet_b.status in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}:
        attrition = int(packet_b.summary.get("event_panel_attrition_count") or 0)
        if attrition:
            return "Remediate `R-03` and `R-04` next: pin the tf-idf corpus and separate IA.II sample construction from event-panel availability."

    packet_d = packet_results.get("D")
    if packet_d is not None and packet_d.status in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}:
        return "Remediate `R-09` and `R-10` next: unify the item-section contract and align FinBERT denominators with the actual filtered universe."

    return "Start with Packet A once the accepted sampled benchmark surface is available."


def _packet_a_full_text_rows(
    cfg: Phase0ValidationAuditConfig,
    paths: _ResolvedAuditPaths,
    accepted_backbone: pl.DataFrame,
    master_dictionary: frozenset[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    year_doc_map = _doc_year_groups(accepted_backbone)
    for year_path in _resolve_year_paths(paths.year_merged_dir, cfg.year_filter):
        filing_year = int(year_path.stem)
        accepted_year_docs = year_doc_map.get(filing_year)
        if accepted_year_docs is None or accepted_year_docs.is_empty():
            continue
        year_df = (
            pl.scan_parquet(year_path)
            .select(
                pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
                pl.col("full_text").cast(pl.Utf8, strict=False).alias("full_text"),
                pl.col("document_type_filename").cast(pl.Utf8, strict=False).alias("document_type_filename"),
            )
            .join(accepted_year_docs.lazy(), on="doc_id", how="inner")
            .collect()
        )
        for row in year_df.iter_rows(named=True):
            full_text = row.get("full_text")
            stripped_text = _strip_edgar_metadata(full_text) if isinstance(full_text, str) else full_text
            paper_cleaned_text, paper_cleaned_decision = _apply_lm2011_paper_cleaning(
                full_text if isinstance(full_text, str) else None
            )
            current_tokens = _legacy_tokenize_lm2011_text(full_text if isinstance(full_text, str) else None)
            appendix_tokens = tokenize_lm2011_text(full_text if isinstance(full_text, str) else None)
            stripped_current_tokens = _legacy_tokenize_lm2011_text(stripped_text if isinstance(stripped_text, str) else None)
            stripped_appendix_tokens = tokenize_lm2011_text(stripped_text if isinstance(stripped_text, str) else None)
            paper_cleaned_current_tokens = _legacy_tokenize_lm2011_text(
                paper_cleaned_text if isinstance(paper_cleaned_text, str) else None
            )
            paper_cleaned_appendix_tokens = tokenize_lm2011_text(
                paper_cleaned_text if isinstance(paper_cleaned_text, str) else None
            )
            marker_flags, snippet = _marker_flags_and_snippet(full_text, cfg.snippet_char_limit)
            stripped_marker_flags, stripped_snippet = _marker_flags_and_snippet(stripped_text, cfg.snippet_char_limit)
            paper_cleaned_marker_flags, paper_cleaned_snippet = _marker_flags_and_snippet(
                paper_cleaned_text,
                cfg.snippet_char_limit,
            )
            recognized_word_count = sum(1 for token in appendix_tokens if token in master_dictionary)
            stripped_recognized_word_count = sum(1 for token in stripped_appendix_tokens if token in master_dictionary)
            paper_cleaned_recognized_word_count = sum(
                1 for token in paper_cleaned_appendix_tokens if token in master_dictionary
            )
            cut_start = paper_cleaned_decision.start
            cut_share = (
                float(cut_start) / float(len(stripped_text))
                if cut_start is not None and isinstance(stripped_text, str) and len(stripped_text) > 0
                else None
            )
            pre_cut_tail_snippet = None
            if cut_start is not None and isinstance(stripped_text, str):
                pre_cut_tail_snippet = _truncate_text(
                    stripped_text[max(0, cut_start - 80) : min(len(stripped_text), cut_start + cfg.snippet_char_limit)],
                    cfg.snippet_char_limit,
                )
            post_cut_tail_snippet = _truncate_text(
                paper_cleaned_text[-cfg.snippet_char_limit :]
                if isinstance(paper_cleaned_text, str) and paper_cleaned_text
                else None,
                cfg.snippet_char_limit,
            )
            rows.append(
                {
                    "text_scope": "full_10k",
                    "doc_id": row["doc_id"],
                    "filing_year": filing_year,
                    "threshold": FULL_10K_THRESHOLD,
                    "current_token_count": len(current_tokens),
                    "appendix_token_count": len(appendix_tokens),
                    "recognized_word_count": recognized_word_count,
                    "current_threshold_pass": len(current_tokens) >= FULL_10K_THRESHOLD,
                    "appendix_threshold_pass": len(appendix_tokens) >= FULL_10K_THRESHOLD,
                    "recognized_threshold_pass": recognized_word_count >= FULL_10K_THRESHOLD,
                    "threshold_flip": (len(current_tokens) >= FULL_10K_THRESHOLD) != (len(appendix_tokens) >= FULL_10K_THRESHOLD),
                    "has_sec_header": marker_flags["sec_header"],
                    "has_html_marker": marker_flags["html"],
                    "has_table_marker": marker_flags["table"],
                    "has_exhibit_marker": marker_flags["exhibit"],
                    "any_marker": any(marker_flags.values()),
                    "edgar_stripped_current_token_count": len(stripped_current_tokens),
                    "edgar_stripped_appendix_token_count": len(stripped_appendix_tokens),
                    "edgar_stripped_recognized_word_count": stripped_recognized_word_count,
                    "edgar_stripped_has_sec_header": stripped_marker_flags["sec_header"],
                    "edgar_stripped_has_html_marker": stripped_marker_flags["html"],
                    "edgar_stripped_has_table_marker": stripped_marker_flags["table"],
                    "edgar_stripped_has_exhibit_marker": stripped_marker_flags["exhibit"],
                    "edgar_stripped_any_marker": any(stripped_marker_flags.values()),
                    "paper_cleaned_current_token_count": len(paper_cleaned_current_tokens),
                    "paper_cleaned_appendix_token_count": len(paper_cleaned_appendix_tokens),
                    "paper_cleaned_recognized_word_count": paper_cleaned_recognized_word_count,
                    "paper_cleaned_current_threshold_pass": len(paper_cleaned_current_tokens) >= FULL_10K_THRESHOLD,
                    "paper_cleaned_appendix_threshold_pass": len(paper_cleaned_appendix_tokens) >= FULL_10K_THRESHOLD,
                    "paper_cleaned_recognized_threshold_pass": paper_cleaned_recognized_word_count >= FULL_10K_THRESHOLD,
                    "paper_cleaned_threshold_flip": (
                        (len(paper_cleaned_current_tokens) >= FULL_10K_THRESHOLD)
                        != (len(paper_cleaned_appendix_tokens) >= FULL_10K_THRESHOLD)
                    ),
                    "paper_cleaned_has_sec_header": paper_cleaned_marker_flags["sec_header"],
                    "paper_cleaned_has_html_marker": paper_cleaned_marker_flags["html"],
                    "paper_cleaned_has_table_marker": paper_cleaned_marker_flags["table"],
                    "paper_cleaned_has_exhibit_marker": paper_cleaned_marker_flags["exhibit"],
                    "paper_cleaned_any_marker": any(paper_cleaned_marker_flags.values()),
                    "paper_cleaned_cut_reason": paper_cleaned_decision.reason,
                    "paper_cleaned_cut_start": cut_start,
                    "paper_cleaned_cut_share": cut_share,
                    "paper_cleaned_anchor_text": _truncate_text(
                        paper_cleaned_decision.anchor_text,
                        min(cfg.snippet_char_limit, 160),
                    ),
                    "example_snippet": snippet,
                    "edgar_stripped_example_snippet": stripped_snippet,
                    "paper_cleaned_example_snippet": paper_cleaned_snippet,
                    "paper_cleaned_pre_cut_tail_snippet": pre_cut_tail_snippet,
                    "paper_cleaned_post_cut_tail_snippet": post_cut_tail_snippet,
                }
            )
    return rows


def _packet_a_mda_rows(
    cfg: Phase0ValidationAuditConfig,
    paths: _ResolvedAuditPaths,
    accepted_backbone: pl.DataFrame,
    master_dictionary: frozenset[str],
) -> list[dict[str, Any]]:
    if not paths.items_analysis_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    year_doc_map = _doc_year_groups(accepted_backbone)
    for year_path in _resolve_year_paths(paths.items_analysis_dir, cfg.year_filter):
        filing_year = int(year_path.stem)
        accepted_year_docs = year_doc_map.get(filing_year)
        if accepted_year_docs is None or accepted_year_docs.is_empty():
            continue
        schema_names = set(pl.scan_parquet(year_path).collect_schema().names())
        doc_type_col = "document_type_filename" if "document_type_filename" in schema_names else "document_type"
        sort_cols = ["doc_id", "char_count"]
        if "canonical_item" in schema_names:
            sort_cols.append("canonical_item")
        if "filename" in schema_names:
            sort_cols.append("filename")
        items_df = (
            pl.scan_parquet(year_path)
            .with_columns(
                pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
                pl.col("item_id").cast(pl.Utf8, strict=False).str.to_uppercase().alias("item_id"),
                pl.col(doc_type_col).cast(pl.Utf8, strict=False).alias("document_type_raw"),
                pl.col("full_text").cast(pl.Utf8, strict=False).alias("full_text"),
            )
            .join(accepted_year_docs.lazy(), on="doc_id", how="inner")
            .filter(
                (pl.col("item_id") == "7")
                & pl.col("document_type_raw").is_in(_RAW_FORM_10K_ONLY)
                & pl.col("full_text").is_not_null()
            )
            .with_columns(pl.col("full_text").str.len_chars().cast(pl.Int32).alias("char_count"))
            .sort(by=sort_cols, descending=[False, True, False, False][: len(sort_cols)], nulls_last=True)
            .unique(subset=["doc_id"], keep="first", maintain_order=True)
            .collect()
        )
        for row in items_df.iter_rows(named=True):
            full_text = row.get("full_text")
            current_tokens = _legacy_tokenize_lm2011_text(full_text if isinstance(full_text, str) else None)
            appendix_tokens = tokenize_lm2011_text(full_text if isinstance(full_text, str) else None)
            recognized_word_count = sum(1 for token in appendix_tokens if token in master_dictionary)
            rows.append(
                {
                    "text_scope": "mda",
                    "doc_id": row["doc_id"],
                    "filing_year": filing_year,
                    "threshold": MDA_THRESHOLD,
                    "current_token_count": len(current_tokens),
                    "appendix_token_count": len(appendix_tokens),
                    "recognized_word_count": recognized_word_count,
                    "current_threshold_pass": len(current_tokens) >= MDA_THRESHOLD,
                    "appendix_threshold_pass": len(appendix_tokens) >= MDA_THRESHOLD,
                    "recognized_threshold_pass": recognized_word_count >= MDA_THRESHOLD,
                    "threshold_flip": (len(current_tokens) >= MDA_THRESHOLD) != (len(appendix_tokens) >= MDA_THRESHOLD),
                    "has_sec_header": False,
                    "has_html_marker": False,
                    "has_table_marker": False,
                    "has_exhibit_marker": False,
                    "any_marker": False,
                    "edgar_stripped_current_token_count": len(current_tokens),
                    "edgar_stripped_appendix_token_count": len(appendix_tokens),
                    "edgar_stripped_recognized_word_count": recognized_word_count,
                    "edgar_stripped_has_sec_header": False,
                    "edgar_stripped_has_html_marker": False,
                    "edgar_stripped_has_table_marker": False,
                    "edgar_stripped_has_exhibit_marker": False,
                    "edgar_stripped_any_marker": False,
                    "paper_cleaned_current_token_count": len(current_tokens),
                    "paper_cleaned_appendix_token_count": len(appendix_tokens),
                    "paper_cleaned_recognized_word_count": recognized_word_count,
                    "paper_cleaned_current_threshold_pass": len(current_tokens) >= MDA_THRESHOLD,
                    "paper_cleaned_appendix_threshold_pass": len(appendix_tokens) >= MDA_THRESHOLD,
                    "paper_cleaned_recognized_threshold_pass": recognized_word_count >= MDA_THRESHOLD,
                    "paper_cleaned_threshold_flip": (len(current_tokens) >= MDA_THRESHOLD) != (len(appendix_tokens) >= MDA_THRESHOLD),
                    "paper_cleaned_has_sec_header": False,
                    "paper_cleaned_has_html_marker": False,
                    "paper_cleaned_has_table_marker": False,
                    "paper_cleaned_has_exhibit_marker": False,
                    "paper_cleaned_any_marker": False,
                    "paper_cleaned_cut_reason": "no_tail_anchor",
                    "paper_cleaned_cut_start": None,
                    "paper_cleaned_cut_share": None,
                    "paper_cleaned_anchor_text": None,
                    "example_snippet": _truncate_text(full_text, cfg.snippet_char_limit),
                    "edgar_stripped_example_snippet": _truncate_text(full_text, cfg.snippet_char_limit),
                    "paper_cleaned_example_snippet": _truncate_text(full_text, cfg.snippet_char_limit),
                    "paper_cleaned_pre_cut_tail_snippet": None,
                    "paper_cleaned_post_cut_tail_snippet": _truncate_text(full_text, cfg.snippet_char_limit),
                }
            )
    return rows


def _packet_a_summary_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "text_scope": pl.Utf8,
                "filing_year": pl.Int32,
                "doc_count": pl.Int64,
                "docs_with_any_marker": pl.Int64,
                "docs_with_any_marker_after_edgar_strip": pl.Int64,
                "docs_with_any_marker_after_paper_cleaning": pl.Int64,
                "docs_with_exhibit_marker_after_paper_cleaning": pl.Int64,
                "current_threshold_pass_count": pl.Int64,
                "appendix_threshold_pass_count": pl.Int64,
                "recognized_threshold_pass_count": pl.Int64,
                "threshold_flip_count": pl.Int64,
                "mean_token_drop_after_edgar_strip": pl.Float64,
                "mean_token_drop_after_paper_cleaning": pl.Float64,
                "mean_recognized_word_drop_after_paper_cleaning": pl.Float64,
                "paper_cleaned_threshold_flip_count": pl.Int64,
                "paper_cleaned_appendix_threshold_pass_count": pl.Int64,
                "paper_cleaned_truncated_doc_count": pl.Int64,
            }
        )
    base = pl.DataFrame(rows)
    per_year = (
        base.group_by("text_scope", "filing_year")
        .agg(
            pl.len().alias("doc_count"),
            pl.col("any_marker").sum().cast(pl.Int64).alias("docs_with_any_marker"),
            pl.col("edgar_stripped_any_marker").sum().cast(pl.Int64).alias("docs_with_any_marker_after_edgar_strip"),
            pl.col("paper_cleaned_any_marker").sum().cast(pl.Int64).alias("docs_with_any_marker_after_paper_cleaning"),
            pl.col("paper_cleaned_has_exhibit_marker").sum().cast(pl.Int64).alias("docs_with_exhibit_marker_after_paper_cleaning"),
            pl.col("current_threshold_pass").sum().cast(pl.Int64).alias("current_threshold_pass_count"),
            pl.col("appendix_threshold_pass").sum().cast(pl.Int64).alias("appendix_threshold_pass_count"),
            pl.col("recognized_threshold_pass").sum().cast(pl.Int64).alias("recognized_threshold_pass_count"),
            pl.col("threshold_flip").sum().cast(pl.Int64).alias("threshold_flip_count"),
            pl.col("paper_cleaned_threshold_flip").sum().cast(pl.Int64).alias("paper_cleaned_threshold_flip_count"),
            pl.col("paper_cleaned_appendix_threshold_pass").sum().cast(pl.Int64).alias("paper_cleaned_appendix_threshold_pass_count"),
            (
                pl.col("paper_cleaned_cut_reason") != "no_tail_anchor"
            ).sum().cast(pl.Int64).alias("paper_cleaned_truncated_doc_count"),
            (
                pl.col("current_token_count") - pl.col("edgar_stripped_current_token_count")
            ).mean().alias("mean_token_drop_after_edgar_strip"),
            (
                pl.col("edgar_stripped_current_token_count") - pl.col("paper_cleaned_current_token_count")
            ).mean().alias("mean_token_drop_after_paper_cleaning"),
            (
                pl.col("edgar_stripped_recognized_word_count") - pl.col("paper_cleaned_recognized_word_count")
            ).mean().alias("mean_recognized_word_drop_after_paper_cleaning"),
        )
        .sort("text_scope", "filing_year")
    )
    overall = (
        base.group_by("text_scope")
        .agg(
            pl.len().alias("doc_count"),
            pl.col("any_marker").sum().cast(pl.Int64).alias("docs_with_any_marker"),
            pl.col("edgar_stripped_any_marker").sum().cast(pl.Int64).alias("docs_with_any_marker_after_edgar_strip"),
            pl.col("paper_cleaned_any_marker").sum().cast(pl.Int64).alias("docs_with_any_marker_after_paper_cleaning"),
            pl.col("paper_cleaned_has_exhibit_marker").sum().cast(pl.Int64).alias("docs_with_exhibit_marker_after_paper_cleaning"),
            pl.col("current_threshold_pass").sum().cast(pl.Int64).alias("current_threshold_pass_count"),
            pl.col("appendix_threshold_pass").sum().cast(pl.Int64).alias("appendix_threshold_pass_count"),
            pl.col("recognized_threshold_pass").sum().cast(pl.Int64).alias("recognized_threshold_pass_count"),
            pl.col("threshold_flip").sum().cast(pl.Int64).alias("threshold_flip_count"),
            pl.col("paper_cleaned_threshold_flip").sum().cast(pl.Int64).alias("paper_cleaned_threshold_flip_count"),
            pl.col("paper_cleaned_appendix_threshold_pass").sum().cast(pl.Int64).alias("paper_cleaned_appendix_threshold_pass_count"),
            (
                pl.col("paper_cleaned_cut_reason") != "no_tail_anchor"
            ).sum().cast(pl.Int64).alias("paper_cleaned_truncated_doc_count"),
            (
                pl.col("current_token_count") - pl.col("edgar_stripped_current_token_count")
            ).mean().alias("mean_token_drop_after_edgar_strip"),
            (
                pl.col("edgar_stripped_current_token_count") - pl.col("paper_cleaned_current_token_count")
            ).mean().alias("mean_token_drop_after_paper_cleaning"),
            (
                pl.col("edgar_stripped_recognized_word_count") - pl.col("paper_cleaned_recognized_word_count")
            ).mean().alias("mean_recognized_word_drop_after_paper_cleaning"),
        )
        .with_columns(pl.lit(None, dtype=pl.Int32).alias("filing_year"))
        .select(per_year.columns)
        .sort("text_scope")
    )
    return pl.concat([per_year, overall], how="vertical_relaxed")


def _packet_a_threshold_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "text_scope": pl.Utf8,
                "doc_id": pl.Utf8,
                "filing_year": pl.Int32,
                "threshold": pl.Int32,
                "current_token_count": pl.Int32,
                "appendix_token_count": pl.Int32,
                "recognized_word_count": pl.Int32,
                "current_threshold_pass": pl.Boolean,
                "appendix_threshold_pass": pl.Boolean,
                "recognized_threshold_pass": pl.Boolean,
                "threshold_flip": pl.Boolean,
                "has_sec_header": pl.Boolean,
                "has_html_marker": pl.Boolean,
                "has_table_marker": pl.Boolean,
                "has_exhibit_marker": pl.Boolean,
                "any_marker": pl.Boolean,
                "edgar_stripped_current_token_count": pl.Int32,
                "edgar_stripped_appendix_token_count": pl.Int32,
                "edgar_stripped_recognized_word_count": pl.Int32,
                "edgar_stripped_has_sec_header": pl.Boolean,
                "edgar_stripped_has_html_marker": pl.Boolean,
                "edgar_stripped_has_table_marker": pl.Boolean,
                "edgar_stripped_has_exhibit_marker": pl.Boolean,
                "edgar_stripped_any_marker": pl.Boolean,
                "paper_cleaned_current_token_count": pl.Int32,
                "paper_cleaned_appendix_token_count": pl.Int32,
                "paper_cleaned_recognized_word_count": pl.Int32,
                "paper_cleaned_current_threshold_pass": pl.Boolean,
                "paper_cleaned_appendix_threshold_pass": pl.Boolean,
                "paper_cleaned_recognized_threshold_pass": pl.Boolean,
                "paper_cleaned_threshold_flip": pl.Boolean,
                "paper_cleaned_has_sec_header": pl.Boolean,
                "paper_cleaned_has_html_marker": pl.Boolean,
                "paper_cleaned_has_table_marker": pl.Boolean,
                "paper_cleaned_has_exhibit_marker": pl.Boolean,
                "paper_cleaned_any_marker": pl.Boolean,
                "paper_cleaned_cut_reason": pl.Utf8,
                "paper_cleaned_cut_start": pl.Int64,
                "paper_cleaned_cut_share": pl.Float64,
                "paper_cleaned_anchor_text": pl.Utf8,
            }
        )
    return (
        pl.DataFrame(rows)
        .select(
            "text_scope",
            "doc_id",
            "filing_year",
            "threshold",
            "current_token_count",
            "appendix_token_count",
            "recognized_word_count",
            "current_threshold_pass",
            "appendix_threshold_pass",
            "recognized_threshold_pass",
            "threshold_flip",
            "has_sec_header",
            "has_html_marker",
            "has_table_marker",
            "has_exhibit_marker",
            "any_marker",
            "edgar_stripped_current_token_count",
            "edgar_stripped_appendix_token_count",
            "edgar_stripped_recognized_word_count",
            "edgar_stripped_has_sec_header",
            "edgar_stripped_has_html_marker",
            "edgar_stripped_has_table_marker",
            "edgar_stripped_has_exhibit_marker",
            "edgar_stripped_any_marker",
            "paper_cleaned_current_token_count",
            "paper_cleaned_appendix_token_count",
            "paper_cleaned_recognized_word_count",
            "paper_cleaned_current_threshold_pass",
            "paper_cleaned_appendix_threshold_pass",
            "paper_cleaned_recognized_threshold_pass",
            "paper_cleaned_threshold_flip",
            "paper_cleaned_has_sec_header",
            "paper_cleaned_has_html_marker",
            "paper_cleaned_has_table_marker",
            "paper_cleaned_has_exhibit_marker",
            "paper_cleaned_any_marker",
            "paper_cleaned_cut_reason",
            "paper_cleaned_cut_start",
            "paper_cleaned_cut_share",
            "paper_cleaned_anchor_text",
        )
        .sort("text_scope", "filing_year", "doc_id")
    )


def _packet_a_strip_comparison_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "text_scope": pl.Utf8,
                "filing_year": pl.Int32,
                "doc_count": pl.Int64,
                "docs_with_any_marker_before": pl.Int64,
                "docs_with_any_marker_after_edgar_strip": pl.Int64,
                "docs_with_any_marker_after_paper_cleaning": pl.Int64,
                "docs_with_sec_header_after_edgar_strip": pl.Int64,
                "docs_with_html_marker_after_edgar_strip": pl.Int64,
                "docs_with_table_marker_after_edgar_strip": pl.Int64,
                "docs_with_exhibit_marker_after_edgar_strip": pl.Int64,
                "docs_with_sec_header_after_paper_cleaning": pl.Int64,
                "docs_with_html_marker_after_paper_cleaning": pl.Int64,
                "docs_with_table_marker_after_paper_cleaning": pl.Int64,
                "docs_with_exhibit_marker_after_paper_cleaning": pl.Int64,
                "truncated_doc_count": pl.Int64,
            }
        )
    base = pl.DataFrame(rows)
    per_year = (
        base.group_by("text_scope", "filing_year")
        .agg(
            pl.len().alias("doc_count"),
            pl.col("any_marker").sum().cast(pl.Int64).alias("docs_with_any_marker_before"),
            pl.col("edgar_stripped_any_marker").sum().cast(pl.Int64).alias("docs_with_any_marker_after_edgar_strip"),
            pl.col("paper_cleaned_any_marker").sum().cast(pl.Int64).alias("docs_with_any_marker_after_paper_cleaning"),
            pl.col("edgar_stripped_has_sec_header").sum().cast(pl.Int64).alias("docs_with_sec_header_after_edgar_strip"),
            pl.col("edgar_stripped_has_html_marker").sum().cast(pl.Int64).alias("docs_with_html_marker_after_edgar_strip"),
            pl.col("edgar_stripped_has_table_marker").sum().cast(pl.Int64).alias("docs_with_table_marker_after_edgar_strip"),
            pl.col("edgar_stripped_has_exhibit_marker").sum().cast(pl.Int64).alias("docs_with_exhibit_marker_after_edgar_strip"),
            pl.col("paper_cleaned_has_sec_header").sum().cast(pl.Int64).alias("docs_with_sec_header_after_paper_cleaning"),
            pl.col("paper_cleaned_has_html_marker").sum().cast(pl.Int64).alias("docs_with_html_marker_after_paper_cleaning"),
            pl.col("paper_cleaned_has_table_marker").sum().cast(pl.Int64).alias("docs_with_table_marker_after_paper_cleaning"),
            pl.col("paper_cleaned_has_exhibit_marker").sum().cast(pl.Int64).alias("docs_with_exhibit_marker_after_paper_cleaning"),
            (
                pl.col("paper_cleaned_cut_reason") != "no_tail_anchor"
            ).sum().cast(pl.Int64).alias("truncated_doc_count"),
        )
        .sort("text_scope", "filing_year")
    )
    overall = (
        base.group_by("text_scope")
        .agg(
            pl.len().alias("doc_count"),
            pl.col("any_marker").sum().cast(pl.Int64).alias("docs_with_any_marker_before"),
            pl.col("edgar_stripped_any_marker").sum().cast(pl.Int64).alias("docs_with_any_marker_after_edgar_strip"),
            pl.col("paper_cleaned_any_marker").sum().cast(pl.Int64).alias("docs_with_any_marker_after_paper_cleaning"),
            pl.col("edgar_stripped_has_sec_header").sum().cast(pl.Int64).alias("docs_with_sec_header_after_edgar_strip"),
            pl.col("edgar_stripped_has_html_marker").sum().cast(pl.Int64).alias("docs_with_html_marker_after_edgar_strip"),
            pl.col("edgar_stripped_has_table_marker").sum().cast(pl.Int64).alias("docs_with_table_marker_after_edgar_strip"),
            pl.col("edgar_stripped_has_exhibit_marker").sum().cast(pl.Int64).alias("docs_with_exhibit_marker_after_edgar_strip"),
            pl.col("paper_cleaned_has_sec_header").sum().cast(pl.Int64).alias("docs_with_sec_header_after_paper_cleaning"),
            pl.col("paper_cleaned_has_html_marker").sum().cast(pl.Int64).alias("docs_with_html_marker_after_paper_cleaning"),
            pl.col("paper_cleaned_has_table_marker").sum().cast(pl.Int64).alias("docs_with_table_marker_after_paper_cleaning"),
            pl.col("paper_cleaned_has_exhibit_marker").sum().cast(pl.Int64).alias("docs_with_exhibit_marker_after_paper_cleaning"),
            (
                pl.col("paper_cleaned_cut_reason") != "no_tail_anchor"
            ).sum().cast(pl.Int64).alias("truncated_doc_count"),
        )
        .with_columns(pl.lit(None, dtype=pl.Int32).alias("filing_year"))
        .select(per_year.columns)
        .sort("text_scope")
    )
    return pl.concat([per_year, overall], how="vertical_relaxed")


def _packet_a_examples_frame(rows: list[dict[str, Any]], max_rows: int) -> pl.DataFrame:
    example_rows = [
        row
        for row in rows
        if row["threshold_flip"]
        or row["paper_cleaned_threshold_flip"]
        or row["any_marker"]
        or row["edgar_stripped_any_marker"]
        or row["paper_cleaned_any_marker"]
        or row["paper_cleaned_cut_reason"] != "no_tail_anchor"
    ]
    if not example_rows:
        return pl.DataFrame(
            schema={
                "text_scope": pl.Utf8,
                "doc_id": pl.Utf8,
                "filing_year": pl.Int32,
                "threshold_flip": pl.Boolean,
                "any_marker": pl.Boolean,
                "edgar_stripped_any_marker": pl.Boolean,
                "paper_cleaned_any_marker": pl.Boolean,
                "paper_cleaned_threshold_flip": pl.Boolean,
                "paper_cleaned_cut_reason": pl.Utf8,
                "paper_cleaned_cut_share": pl.Float64,
                "example_snippet": pl.Utf8,
                "edgar_stripped_example_snippet": pl.Utf8,
                "paper_cleaned_example_snippet": pl.Utf8,
                "paper_cleaned_pre_cut_tail_snippet": pl.Utf8,
                "paper_cleaned_post_cut_tail_snippet": pl.Utf8,
            }
        )
    ordered = sorted(
        example_rows,
        key=lambda row: (
            0 if row["paper_cleaned_cut_reason"] != "no_tail_anchor" else 1,
            0 if row["threshold_flip"] else 1,
            0 if row["paper_cleaned_threshold_flip"] else 1,
            0 if row["any_marker"] else 1,
            0 if row["edgar_stripped_any_marker"] else 1,
            0 if row["paper_cleaned_any_marker"] else 1,
            row["text_scope"],
            row["doc_id"],
        ),
    )[:max_rows]
    return pl.DataFrame(ordered).select(
        "text_scope",
        "doc_id",
        "filing_year",
        "threshold_flip",
        "any_marker",
        "edgar_stripped_any_marker",
        "paper_cleaned_any_marker",
        "paper_cleaned_threshold_flip",
        "paper_cleaned_cut_reason",
        "paper_cleaned_cut_share",
        "example_snippet",
        "edgar_stripped_example_snippet",
        "paper_cleaned_example_snippet",
        "paper_cleaned_pre_cut_tail_snippet",
        "paper_cleaned_post_cut_tail_snippet",
    )


def _packet_a_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scope in ("full_10k", "mda"):
        scoped_rows = [row for row in rows if row["text_scope"] == scope]
        doc_count = len(scoped_rows)
        current_vs_appendix_doc_count = sum(
            1 for row in scoped_rows if row["current_token_count"] != row["appendix_token_count"]
        )
        appendix_vs_recognized_doc_count = sum(
            1 for row in scoped_rows if row["appendix_token_count"] != row["recognized_word_count"]
        )
        mean_current_minus_appendix = (
            sum(row["current_token_count"] - row["appendix_token_count"] for row in scoped_rows) / doc_count
            if doc_count
            else 0.0
        )
        mean_appendix_minus_recognized = (
            sum(row["appendix_token_count"] - row["recognized_word_count"] for row in scoped_rows) / doc_count
            if doc_count
            else 0.0
        )
        summary[f"{scope}_doc_count"] = int(doc_count)
        summary[f"{scope}_current_vs_appendix_doc_count"] = int(current_vs_appendix_doc_count)
        summary[f"{scope}_appendix_vs_recognized_doc_count"] = int(appendix_vs_recognized_doc_count)
        summary[f"{scope}_mean_current_minus_appendix_token_delta"] = float(mean_current_minus_appendix)
        summary[f"{scope}_mean_appendix_minus_recognized_token_delta"] = float(mean_appendix_minus_recognized)
    return summary


def _format_delta_finding(
    scope_label: str,
    comparison_label: str,
    differing_doc_count: int,
    doc_count: int,
    mean_delta: float,
) -> str:
    mean_label = "current-minus-Appendix" if comparison_label == "current-vs-Appendix" else "Appendix-minus-recognized"
    return (
        f"Accepted sampled {scope_label} {comparison_label} token delta: "
        f"{differing_doc_count}/{doc_count} docs differ; mean {mean_label} = {mean_delta:.2f}"
    )


def _packet_b_corpus_summary(
    paths: _ResolvedAuditPaths,
    accepted_backbone: pl.DataFrame,
    year_filter: tuple[int, ...] | None,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    broad_years = _resolve_year_paths(paths.year_merged_dir, year_filter)
    broad_doc_count = 0
    min_year: int | None = None
    max_year: int | None = None
    for year_path in broad_years:
        year = int(year_path.stem)
        count = pl.scan_parquet(year_path).select(pl.len()).collect().item()
        broad_doc_count += int(count)
        min_year = year if min_year is None else min(min_year, year)
        max_year = year if max_year is None else max(max_year, year)
    rows.append(
        {
            "corpus_label": "year_merged_broad",
            "doc_count": int(broad_doc_count),
            "min_filing_year": min_year,
            "max_filing_year": max_year,
        }
    )
    rows.append(
        {
            "corpus_label": "accepted_backbone",
            "doc_count": int(accepted_backbone.height),
            "min_filing_year": int(accepted_backbone["filing_year"].min()) if accepted_backbone.height else None,
            "max_filing_year": int(accepted_backbone["filing_year"].max()) if accepted_backbone.height else None,
        }
    )
    if paths.text_features_full_10k_path.exists():
        feature_df = _load_feature_doc_years(paths.text_features_full_10k_path, year_filter)
        rows.append(
            {
                "corpus_label": "text_features_full_10k",
                "doc_count": int(feature_df.height),
                "min_filing_year": int(feature_df["filing_year"].min()) if feature_df.height else None,
                "max_filing_year": int(feature_df["filing_year"].max()) if feature_df.height else None,
            }
        )
    if paths.event_panel_path.exists():
        event_df = _load_doc_year_frame(paths.event_panel_path, year_filter)
        rows.append(
            {
                "corpus_label": "event_panel",
                "doc_count": int(event_df.height),
                "min_filing_year": int(event_df["filing_year"].min()) if event_df.height else None,
                "max_filing_year": int(event_df["filing_year"].max()) if event_df.height else None,
            }
        )
    return pl.DataFrame(rows).sort("corpus_label")


def _packet_b_form_mix(
    paths: _ResolvedAuditPaths,
    year_filter: tuple[int, ...] | None,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    broad_counts = _collect_form_counts_from_year_merged(paths.year_merged_dir, year_filter)
    rows.extend(_form_count_rows("year_merged_broad", broad_counts))

    backbone_df = (
        pl.scan_parquet(paths.sample_backbone_path)
        .select(
            pl.col("normalized_form").cast(pl.Utf8, strict=False).alias("normalized_form"),
            pl.col("filing_date").dt.year().cast(pl.Int32).alias("filing_year"),
        )
        .filter(_year_filter_expr("filing_year", year_filter))
        .collect()
    )
    rows.extend(
        _form_count_rows(
            "accepted_backbone",
            _count_series(backbone_df.get_column("normalized_form").drop_nulls().to_list()),
        )
    )

    if paths.text_features_full_10k_path.exists():
        feature_df = (
            pl.scan_parquet(paths.text_features_full_10k_path)
            .select(
                pl.col("normalized_form").cast(pl.Utf8, strict=False).alias("normalized_form"),
                pl.col("filing_date").dt.year().cast(pl.Int32).alias("filing_year"),
            )
            .filter(_year_filter_expr("filing_year", year_filter))
            .collect()
        )
        rows.extend(
            _form_count_rows(
                "text_features_full_10k",
                _count_series(feature_df.get_column("normalized_form").drop_nulls().to_list()),
            )
        )
    return pl.DataFrame(rows).sort("corpus_label", "normalized_form")


def _packet_b_representative_terms(
    year_paths: list[Path],
    accepted_backbone: pl.DataFrame,
    feature_doc_df: pl.DataFrame,
) -> pl.DataFrame:
    accepted_doc_map = _doc_year_groups(accepted_backbone)
    feature_doc_map = _doc_year_groups(feature_doc_df)
    universe_counts: dict[str, dict[str, int]] = {
        "year_merged_broad": {"doc_count": 0, **{term: 0 for _, term in _REPRESENTATIVE_TERMS}},
        "accepted_backbone": {"doc_count": 0, **{term: 0 for _, term in _REPRESENTATIVE_TERMS}},
        "text_features_full_10k": {"doc_count": 0, **{term: 0 for _, term in _REPRESENTATIVE_TERMS}},
    }
    for year_path in year_paths:
        year_df = (
            pl.scan_parquet(year_path)
            .select(
                pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
                pl.col("full_text").cast(pl.Utf8, strict=False).alias("full_text"),
            )
            .collect()
        )
        _update_term_counts(universe_counts["year_merged_broad"], year_df)
        accepted_docs = accepted_doc_map.get(int(year_path.stem))
        if accepted_docs is not None and not accepted_docs.is_empty():
            accepted_year_df = year_df.join(accepted_docs.select("doc_id"), on="doc_id", how="inner")
            _update_term_counts(universe_counts["accepted_backbone"], accepted_year_df)
        feature_docs = feature_doc_map.get(int(year_path.stem))
        if feature_docs is not None and not feature_docs.is_empty():
            feature_year_df = year_df.join(feature_docs.select("doc_id"), on="doc_id", how="inner")
            _update_term_counts(universe_counts["text_features_full_10k"], feature_year_df)

    rows: list[dict[str, Any]] = []
    for corpus_label, stats in universe_counts.items():
        doc_count = int(stats["doc_count"])
        for signal_name, term in _REPRESENTATIVE_TERMS:
            df_i = int(stats[term])
            rows.append(
                {
                    "corpus_label": corpus_label,
                    "signal_name": signal_name,
                    "term": term,
                    "doc_count": doc_count,
                    "document_frequency": df_i,
                    "idf": math.log(doc_count / df_i) if doc_count > 0 and df_i > 0 else None,
                }
            )
    return pl.DataFrame(rows).sort("corpus_label", "signal_name")


def _packet_b_event_attrition(
    paths: _ResolvedAuditPaths,
    accepted_backbone: pl.DataFrame,
    year_filter: tuple[int, ...] | None,
) -> pl.DataFrame:
    event_df = _load_doc_year_frame(paths.event_panel_path, year_filter)
    rows: list[dict[str, Any]] = []
    for filing_year in sorted(accepted_backbone["filing_year"].unique().to_list()):
        backbone_year = accepted_backbone.filter(pl.col("filing_year") == filing_year)
        event_year = event_df.filter(pl.col("filing_year") == filing_year)
        rows.append(
            {
                "filing_year": int(filing_year),
                "backbone_doc_count": int(backbone_year.height),
                "event_panel_doc_count": int(event_year.height),
                "lost_doc_count": int(backbone_year.join(event_year.select("doc_id"), on="doc_id", how="anti").height),
            }
        )
    rows.append(
        {
            "filing_year": None,
            "backbone_doc_count": int(accepted_backbone.height),
            "event_panel_doc_count": int(event_df.height),
            "lost_doc_count": int(accepted_backbone.join(event_df.select("doc_id"), on="doc_id", how="anti").height),
        }
    )
    return pl.DataFrame(rows).sort("filing_year")


def _packet_c_units_audit(paths: _ResolvedAuditPaths) -> pl.DataFrame:
    event_df = pl.read_parquet(paths.event_panel_path)
    sue_df = pl.read_parquet(paths.sue_panel_path)
    rows = [
        _units_row("lm2011_event_panel", "filing_period_excess_return", event_df, 100.0),
        _units_row("lm2011_sue_panel", "sue", sue_df, 100.0),
        _units_row("lm2011_sue_panel", "analyst_dispersion", sue_df, 100.0),
        _units_row("lm2011_sue_panel", "analyst_revisions", sue_df, 100.0),
    ]
    return pl.DataFrame(rows).sort("artifact_name", "field_name")


def _packet_c_denominator_audit(
    cfg: Phase0ValidationAuditConfig,
    paths: _ResolvedAuditPaths,
) -> pl.DataFrame:
    event_panel_df = pl.read_parquet(paths.event_panel_path)
    if cfg.year_filter is not None:
        event_panel_df = event_panel_df.filter(pl.col("filing_date").dt.year().is_in(cfg.year_filter))
    quarterly_lf = pl.scan_parquet(paths.quarterly_accounting_panel_path)
    docs_df = (
        attach_eligible_quarterly_accounting(
            event_panel_df.lazy().with_columns(pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("_lm2011_gvkey_int")),
            quarterly_lf.with_columns(pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int")),
            filing_gvkey_col="_lm2011_gvkey_int",
        )
        .with_columns(pl.col("_lm2011_gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int"))
        .collect()
    )
    docs_df = docs_df.filter(pl.col("quarter_report_date").is_not_null())
    docs_df = _attach_pre_filing_price_and_prior_month_price(docs_df, pl.scan_parquet(paths.daily_panel_path))
    docs_df = docs_df.with_columns(
        pl.coalesce(
            [
                (
                    pl.col("quarter_fiscal_period_end").cast(pl.Date, strict=False)
                    if "quarter_fiscal_period_end" in docs_df.columns
                    else pl.lit(None, dtype=pl.Date)
                ),
                (
                    pl.col("APDEDATEQ").cast(pl.Date, strict=False)
                    if "APDEDATEQ" in docs_df.columns
                    else pl.lit(None, dtype=pl.Date)
                ),
                (
                    pl.col("PDATEQ").cast(pl.Date, strict=False)
                    if "PDATEQ" in docs_df.columns
                    else pl.lit(None, dtype=pl.Date)
                ),
            ]
        ).alias("_quarter_fiscal_period_end")
    )
    analyst_df = _prepare_selected_analyst_input_df(paths.doc_analyst_selected_path)
    matched_df = select_refinitiv_lm2011_doc_analyst_inputs(
        docs_df.rename({"_quarter_fiscal_period_end": "quarter_fiscal_period_end"}),
        analyst_df,
    ).filter(pl.col("analyst_match_status") == "MATCHED")
    sue_doc_ids = pl.read_parquet(paths.sue_panel_path).select("doc_id").unique()
    matched_df = matched_df.join(sue_doc_ids, on="doc_id", how="inner")
    if matched_df.is_empty():
        return pl.DataFrame(
            schema={
                "doc_id": pl.Utf8,
                "matched_announcement_date": pl.Date,
                "pre_filing_price": pl.Float64,
                "pre_announcement_price": pl.Float64,
                "prior_month_price": pl.Float64,
                "announcement_prior_month_price": pl.Float64,
                "filing_anchor_sue": pl.Float64,
                "announcement_anchor_sue": pl.Float64,
                "abs_diff_sue": pl.Float64,
                "filing_anchor_dispersion": pl.Float64,
                "announcement_anchor_dispersion": pl.Float64,
                "filing_anchor_revisions": pl.Float64,
                "announcement_anchor_revisions": pl.Float64,
            }
        )

    permnos = [int(value) for value in matched_df.get_column("KYPERMNO").drop_nulls().unique().to_list()]
    daily_price = _prepare_price_lookup_df(paths.daily_panel_path, permnos)
    matched_df = matched_df.sort("KYPERMNO", "matched_announcement_date")
    matched_df = matched_df.join_asof(
        daily_price.rename({"trade_date": "_announcement_trade_date", "_price": "pre_announcement_price"}),
        left_on="matched_announcement_date",
        right_on="_announcement_trade_date",
        by="KYPERMNO",
        strategy="backward",
        check_sortedness=False,
    )
    matched_df = matched_df.with_columns(
        pl.col("matched_announcement_date")
        .map_elements(_previous_month_end, return_dtype=pl.Date)
        .alias("_announcement_prior_month_end")
    )
    matched_df = matched_df.join_asof(
        daily_price.rename({"trade_date": "_announcement_prior_month_trade_date", "_price": "announcement_prior_month_price"}),
        left_on="_announcement_prior_month_end",
        right_on="_announcement_prior_month_trade_date",
        by="KYPERMNO",
        strategy="backward",
        check_sortedness=False,
    )
    return (
        matched_df.with_columns(
            ((pl.col("actual_eps") - pl.col("forecast_consensus_mean")) / pl.col("pre_filing_price")).alias("filing_anchor_sue"),
            ((pl.col("actual_eps") - pl.col("forecast_consensus_mean")) / pl.col("pre_announcement_price")).alias("announcement_anchor_sue"),
            (pl.col("forecast_dispersion") / pl.col("pre_filing_price")).alias("filing_anchor_dispersion"),
            (pl.col("forecast_dispersion") / pl.col("pre_announcement_price")).alias("announcement_anchor_dispersion"),
            (pl.col("forecast_revision_4m") / pl.col("prior_month_price")).alias("filing_anchor_revisions"),
            (pl.col("forecast_revision_4m") / pl.col("announcement_prior_month_price")).alias("announcement_anchor_revisions"),
        )
        .with_columns(
            (pl.col("filing_anchor_sue") - pl.col("announcement_anchor_sue")).abs().alias("abs_diff_sue"),
        )
        .select(
            "doc_id",
            "matched_announcement_date",
            "pre_filing_price",
            "pre_announcement_price",
            "prior_month_price",
            "announcement_prior_month_price",
            "filing_anchor_sue",
            "announcement_anchor_sue",
            "abs_diff_sue",
            "filing_anchor_dispersion",
            "announcement_anchor_dispersion",
            "filing_anchor_revisions",
            "announcement_anchor_revisions",
        )
        .sort("doc_id")
    )


def _packet_c_output_classification(paths: _ResolvedAuditPaths) -> pl.DataFrame:
    rows = [
        {
            "artifact_name": "lm2011_event_panel",
            "field_name": "filing_period_excess_return",
            "classification": "clearly_internal_unit",
            "note": "Return-side event_panel values are stored in decimal units and require x100 for paper-style display.",
        },
        {
            "artifact_name": "lm2011_sue_panel",
            "field_name": "sue",
            "classification": "clearly_internal_unit",
            "note": "SUE values are stored in decimal units and likely require x100 for paper-style display.",
        },
        {
            "artifact_name": "lm2011_sue_panel",
            "field_name": "analyst_dispersion",
            "classification": "clearly_internal_unit",
            "note": "Dispersion values are stored in decimal units.",
        },
        {
            "artifact_name": "lm2011_sue_panel",
            "field_name": "analyst_revisions",
            "classification": "clearly_internal_unit",
            "note": "Revision values are stored in decimal units.",
        },
        {
            "artifact_name": "lm2011_table_iv_results",
            "field_name": "estimate",
            "classification": "unresolved_empty_artifact" if _parquet_row_count(paths.table_iv_results_path) == 0 else "requires_table_level_review",
            "note": "Sampled table artifact is empty, so display-unit behavior cannot be inferred from stored estimates.",
        },
        {
            "artifact_name": "lm2011_table_viii_results",
            "field_name": "estimate",
            "classification": "unresolved_empty_artifact" if _parquet_row_count(paths.table_viii_results_path) == 0 else "requires_table_level_review",
            "note": "Sampled Table VIII artifact is empty, so display-unit behavior cannot be inferred from stored estimates.",
        },
    ]
    return pl.DataFrame(rows).sort("artifact_name", "field_name")


def _prepare_selected_analyst_input_df(path: Path) -> pl.DataFrame:
    schema_names = set(pl.scan_parquet(path).collect_schema().names())
    return (
        pl.scan_parquet(path)
        .filter(pl.col("analyst_match_status").cast(pl.Utf8, strict=False) == pl.lit("MATCHED"))
        .select(
            pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int"),
            pl.col("matched_announcement_date").cast(pl.Date, strict=False).alias("announcement_date"),
            pl.col("matched_fiscal_period_end").cast(pl.Date, strict=False).alias("fiscal_period_end"),
            pl.col("actual_eps").cast(pl.Float64, strict=False).alias("actual_eps"),
            pl.col("forecast_consensus_mean").cast(pl.Float64, strict=False).alias("forecast_consensus_mean"),
            pl.col("forecast_dispersion").cast(pl.Float64, strict=False).alias("forecast_dispersion"),
            pl.col("forecast_revision_4m").cast(pl.Float64, strict=False).alias("forecast_revision_4m"),
            (
                pl.col("forecast_revision_1m").cast(pl.Float64, strict=False)
                if "forecast_revision_1m" in schema_names
                else pl.lit(None, dtype=pl.Float64)
            ).alias("forecast_revision_1m"),
        )
        .drop_nulls(subset=["gvkey_int", "announcement_date", "fiscal_period_end"])
        .unique(subset=["gvkey_int", "announcement_date", "fiscal_period_end"], keep="first")
        .collect()
    )


def _prepare_price_lookup_df(daily_panel_path: Path, permnos: list[int]) -> pl.DataFrame:
    daily_lf = pl.scan_parquet(daily_panel_path)
    schema = daily_lf.collect_schema()
    price_col = "FINAL_PRC" if "FINAL_PRC" in schema else "PRC"
    return (
        daily_lf.filter(pl.col("KYPERMNO").cast(pl.Int32, strict=False).is_in(permnos))
        .select(
            pl.col("KYPERMNO").cast(pl.Int32, strict=False).alias("KYPERMNO"),
            pl.col("CALDT").cast(pl.Date, strict=False).alias("trade_date"),
            pl.col(price_col).cast(pl.Float64, strict=False).abs().alias("_price"),
        )
        .drop_nulls(subset=["KYPERMNO", "trade_date", "_price"])
        .collect()
        .sort("KYPERMNO", "trade_date")
    )


def _units_row(
    artifact_name: str,
    field_name: str,
    df: pl.DataFrame,
    paper_multiplier: float,
) -> dict[str, Any]:
    series = df.get_column(field_name).drop_nulls()
    mean_abs = float(series.abs().mean()) if series.len() > 0 else None
    return {
        "artifact_name": artifact_name,
        "field_name": field_name,
        "nonnull_row_count": int(series.len()),
        "mean_abs_internal": mean_abs,
        "paper_display_multiplier": paper_multiplier,
        "mean_abs_paper_display_equivalent": (mean_abs * paper_multiplier) if mean_abs is not None else None,
        "classification": "clearly_internal_unit",
    }


def _packet_d_reconciliation_table(
    dictionary_universe_df: pl.DataFrame,
    item_features_long_path: Path,
) -> pl.DataFrame:
    finbert_df = (
        pl.read_parquet(item_features_long_path)
        .select(
            pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
            pl.col("benchmark_item_code").cast(pl.Utf8, strict=False).alias("benchmark_item_code"),
        )
        .unique(maintain_order=True)
        .with_columns(_benchmark_item_to_item_id_expr().alias("item_id"))
        .select("doc_id", "item_id")
    )
    dictionary_pairs = dictionary_universe_df.select("doc_id", "item_id").unique(maintain_order=True)
    full_join = (
        dictionary_pairs.with_columns(pl.lit(True).alias("_in_dictionary"))
        .join(
            finbert_df.with_columns(pl.lit(True).alias("_in_finbert")),
            on=["doc_id", "item_id"],
            how="full",
            coalesce=True,
        )
        .with_columns(
            pl.col("_in_dictionary").fill_null(False),
            pl.col("_in_finbert").fill_null(False),
            pl.when(pl.col("_in_dictionary") & pl.col("_in_finbert"))
            .then(pl.lit("both"))
            .when(pl.col("_in_dictionary"))
            .then(pl.lit("dictionary_only"))
            .otherwise(pl.lit("finbert_only"))
            .alias("classification"),
        )
    )
    return (
        full_join.group_by("item_id", "classification")
        .agg(
            pl.len().alias("row_count"),
            pl.col("doc_id").n_unique().alias("doc_count"),
        )
        .sort("item_id", "classification")
    )


def _packet_d_removal_waterfall(
    section_cfg: FinbertSectionUniverseConfig,
    year_paths: list[Path],
    benchmark_backbone_path: Path,
) -> pl.DataFrame:
    raw_lf = pl.concat([pl.scan_parquet(path) for path in year_paths], how="diagonal_relaxed")
    schema = set(raw_lf.collect_schema().names())
    doc_type_col = "document_type_filename" if "document_type_filename" in schema else "document_type"
    base = raw_lf.with_columns(
        pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
        pl.col("item_id").cast(pl.Utf8, strict=False).str.to_uppercase().alias("item_id"),
        pl.col(doc_type_col).cast(pl.Utf8, strict=False).alias("document_type_raw"),
        pl.col("full_text").cast(pl.Utf8, strict=False).alias("full_text"),
        (
            pl.col("item_status").cast(pl.Utf8, strict=False)
            if "item_status" in schema
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("item_status"),
        (
            pl.col("exists_by_regime").cast(pl.Boolean, strict=False)
            if "exists_by_regime" in schema
            else pl.lit(True, dtype=pl.Boolean)
        ).alias("exists_by_regime"),
        (
            pl.col("canonical_item").cast(pl.Utf8, strict=False)
            if "canonical_item" in schema
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("canonical_item"),
        (
            pl.col("filename").cast(pl.Utf8, strict=False)
            if "filename" in schema
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("filename"),
    )

    stages: list[tuple[str, pl.LazyFrame]] = []
    stages.append(("raw_target_items", base.filter(pl.col("item_id").is_in(_TARGET_ITEM_IDS))))
    stages.append(("form_filtered", stages[-1][1].filter(pl.col("document_type_raw").is_in(section_cfg.form_types))))
    active_expr = (
        (pl.col("item_status") == "active")
        if section_cfg.require_active_items
        else pl.lit(True)
    )
    regime_expr = pl.col("exists_by_regime") if section_cfg.require_exists_by_regime else pl.lit(True)
    stages.append(("status_regime_filtered", stages[-1][1].filter(active_expr & regime_expr)))
    stages.append(
        (
            "min_char_filtered",
            stages[-1][1].filter(pl.col("full_text").str.strip_chars().str.len_chars() >= section_cfg.min_char_count),
        )
    )
    backbone_lf = (
        pl.scan_parquet(benchmark_backbone_path)
        .select(pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"))
        .unique()
    )
    stages.append(("benchmark_backbone_filtered", stages[-1][1].join(backbone_lf, on="doc_id", how="inner")))
    deduped = (
        stages[-1][1]
        .with_columns(
            pl.when(pl.col("item_id") == "1")
            .then(pl.lit("item_1"))
            .when(pl.col("item_id") == "1A")
            .then(pl.lit("item_1a"))
            .otherwise(pl.lit("item_7"))
            .alias("benchmark_item_code"),
            pl.col("full_text").str.len_chars().cast(pl.Int32).alias("char_count"),
        )
        .sort(
            by=["doc_id", "benchmark_item_code", "char_count", "canonical_item", "filename"],
            descending=[False, False, True, False, False],
            nulls_last=True,
        )
        .unique(subset=["doc_id", "benchmark_item_code"], keep="first", maintain_order=True)
    )
    stages.append(("deduped_final", deduped))

    rows: list[dict[str, Any]] = []
    for stage_name, stage_lf in stages:
        stats_df = stage_lf.select(
            pl.len().alias("row_count"),
            pl.col("doc_id").n_unique().alias("doc_count"),
        ).collect()
        rows.append(
            {
                "stage_name": stage_name,
                "row_count": int(stats_df.item(0, "row_count")),
                "doc_count": int(stats_df.item(0, "doc_count")),
            }
        )
    return pl.DataFrame(rows)


def _packet_d_coverage_reconciliation(
    section_cfg: FinbertSectionUniverseConfig,
    year_paths: list[Path],
    run_manifest: dict[str, Any],
    run_manifest_path: Path,
    item_features_long_path: Path,
    *,
    sample_backbone_path: Path,
) -> pl.DataFrame:
    reported = run_manifest.get("coverage_summary") or {}
    effective_year_filter = tuple(sorted(int(path.stem) for path in year_paths)) or None
    expected_contract = _expected_packet_d_contract(
        section_cfg=section_cfg,
        sample_backbone_path=sample_backbone_path,
        effective_year_filter=effective_year_filter,
    )
    backbone_path_raw = run_manifest.get("backbone_path")
    if not backbone_path_raw:
        backbone_path_raw = (run_manifest.get("nonportable_diagnostics") or {}).get("backbone_path")
    backbone_path = resolve_manifest_path(
        backbone_path_raw,
        manifest_path=run_manifest_path,
        path_semantics=run_manifest.get("path_semantics"),
    )
    candidate_contract = _candidate_backbone_contract(
        run_manifest,
        run_manifest_path=run_manifest_path,
        effective_year_filter=effective_year_filter,
        section_cfg=section_cfg,
        sample_backbone_path=sample_backbone_path,
    )
    actual_filtered_doc_count: int | None = None
    if backbone_path is not None and backbone_path.exists():
        actual_filtered_doc_count = int(
            load_eligible_section_universe(
                section_cfg,
                year_paths=year_paths,
                target_doc_universe_path=backbone_path,
            )
            .select(pl.col("doc_id").n_unique())
            .collect()
            .item()
        )
    sample_backbone_filtered_doc_count: int | None = None
    if sample_backbone_path.exists():
        sample_backbone_filtered_doc_count = int(
            load_eligible_section_universe(
                section_cfg,
                year_paths=year_paths,
                target_doc_universe_path=sample_backbone_path,
            )
            .select(pl.col("doc_id").n_unique())
            .collect()
            .item()
        )
    covered_doc_count = int(
        pl.read_parquet(item_features_long_path)
        .select(pl.col("doc_id").n_unique())
        .item()
    )
    reported_backbone_doc_count = int(reported.get("backbone_doc_count") or 0)
    normalized_finbert_backbone = str(backbone_path) if backbone_path is not None else None
    normalized_sample_backbone = str(sample_backbone_path.resolve())
    return pl.DataFrame(
        [
            {
                "run_name": str(run_manifest.get("run_name") or ""),
                "finbert_backbone_path": normalized_finbert_backbone,
                "sample_backbone_path": normalized_sample_backbone,
                "finbert_backbone_matches_sample_backbone": (
                    _packet_d_contract_match(
                        candidate_contract,
                        expected_contract=expected_contract,
                    )
                ),
                "finbert_backbone_path_matches_sample_backbone": (
                    normalized_finbert_backbone == normalized_sample_backbone
                    if normalized_finbert_backbone is not None
                    else False
                ),
                "reported_backbone_doc_count": reported_backbone_doc_count,
                "actual_filtered_doc_count": actual_filtered_doc_count,
                "sample_backbone_filtered_doc_count": sample_backbone_filtered_doc_count,
                "reported_covered_doc_count": int(reported.get("covered_doc_count") or 0),
                "actual_covered_doc_count": covered_doc_count,
                "denominator_gap": (
                    reported_backbone_doc_count - actual_filtered_doc_count
                    if actual_filtered_doc_count is not None
                    else None
                ),
                "sample_backbone_denominator_gap": (
                    reported_backbone_doc_count - sample_backbone_filtered_doc_count
                    if sample_backbone_filtered_doc_count is not None
                    else None
                ),
            }
        ]
    )


def _packet_d_extraction_regime_comparison(
    cfg: Phase0ValidationAuditConfig,
    paths: _ResolvedAuditPaths,
    dictionary_universe_df: pl.DataFrame,
    year_filter: tuple[int, ...] | None,
) -> pl.DataFrame:
    if dictionary_universe_df.is_empty():
        return pl.DataFrame(
            schema={
                "doc_id": pl.Utf8,
                "filing_year": pl.Int32,
                "legacy_item_count": pl.Int32,
                "v2_item_count": pl.Int32,
                "legacy_item_ids": pl.Utf8,
                "v2_item_ids": pl.Utf8,
                "same_item_set": pl.Boolean,
                "legacy_total_chars": pl.Int32,
                "v2_total_chars": pl.Int32,
                "error": pl.Utf8,
            }
        )
    sample_doc_ids = dictionary_universe_df.select("doc_id").unique().head(cfg.regime_compare_doc_limit)
    docs_df = (
        pl.scan_parquet(paths.sample_backbone_path)
        .select(
            pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
            pl.col("filing_date").cast(pl.Date, strict=False).alias("filing_date"),
            pl.col("period_end").cast(pl.Date, strict=False).alias("period_end"),
            pl.col("document_type_filename").cast(pl.Utf8, strict=False).alias("document_type_filename"),
            pl.col("full_text").cast(pl.Utf8, strict=False).alias("full_text"),
        )
        .filter(_year_filter_expr("filing_date", year_filter, use_date_year=True))
        .join(sample_doc_ids.lazy(), on="doc_id", how="inner")
        .collect()
    )
    rows: list[dict[str, Any]] = []
    for row in docs_df.iter_rows(named=True):
        full_text = row.get("full_text")
        try:
            legacy_items = extract_filing_items(
                full_text if isinstance(full_text, str) else "",
                form_type=row.get("document_type_filename"),
                filing_date=row.get("filing_date"),
                period_end=row.get("period_end"),
                extraction_regime="legacy",
            )
            v2_items = extract_filing_items(
                full_text if isinstance(full_text, str) else "",
                form_type=row.get("document_type_filename"),
                filing_date=row.get("filing_date"),
                period_end=row.get("period_end"),
                extraction_regime="v2",
            )
            legacy_summary = _summarize_extracted_target_items(legacy_items)
            v2_summary = _summarize_extracted_target_items(v2_items)
            error = None
        except Exception as exc:
            legacy_summary = {"item_ids": "", "item_count": 0, "total_chars": 0}
            v2_summary = {"item_ids": "", "item_count": 0, "total_chars": 0}
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "doc_id": row["doc_id"],
                "filing_year": row["filing_date"].year if row.get("filing_date") is not None else None,
                "legacy_item_count": legacy_summary["item_count"],
                "v2_item_count": v2_summary["item_count"],
                "legacy_item_ids": legacy_summary["item_ids"],
                "v2_item_ids": v2_summary["item_ids"],
                "same_item_set": legacy_summary["item_ids"] == v2_summary["item_ids"],
                "legacy_total_chars": legacy_summary["total_chars"],
                "v2_total_chars": v2_summary["total_chars"],
                "error": error,
            }
        )
    return pl.DataFrame(rows).sort("doc_id")


def _resolve_latest_valid_finbert_run(
    finbert_output_root: Path,
    *,
    sampled_items_analysis_dir: Path,
    sample_backbone_path: Path,
    requested_year_filter: tuple[int, ...] | None,
) -> Path | None:
    if not finbert_output_root.exists():
        return None
    candidate_dirs = sorted(
        (path for path in finbert_output_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidate_dirs:
        run_manifest_path = path / "run_manifest.json"
        if not (
            run_manifest_path.exists()
            and (path / "item_features_long.parquet").exists()
            and (path / "coverage_report.parquet").exists()
        ):
            continue
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        section_cfg = _section_universe_from_manifest(sampled_items_analysis_dir, run_manifest)
        effective_year_filter = requested_year_filter or _manifest_year_filter(run_manifest)
        expected_contract = _expected_packet_d_contract(
            section_cfg=section_cfg,
            sample_backbone_path=sample_backbone_path,
            effective_year_filter=effective_year_filter,
        )
        candidate_contract = _candidate_backbone_contract(
            run_manifest,
            run_manifest_path=run_manifest_path,
            effective_year_filter=effective_year_filter,
            section_cfg=section_cfg,
            sample_backbone_path=sample_backbone_path,
        )
        if _packet_d_contract_match(candidate_contract, expected_contract=expected_contract):
            return path
    return None


def _section_universe_from_manifest(
    source_items_dir: Path,
    run_manifest: dict[str, Any],
) -> FinbertSectionUniverseConfig:
    manifest_section = run_manifest.get("section_universe") or {}
    raw_target_items = manifest_section.get("target_items") or []
    target_items = tuple(
        BenchmarkItemSpec(
            benchmark_item_code=str(item["benchmark_item_code"]),
            item_id=str(item["item_id"]),
            benchmark_item_label=str(item["benchmark_item_label"]),
        )
        for item in raw_target_items
    ) or tuple(
        BenchmarkItemSpec(_EXTRACTION_ITEM_LABELS[item_id], item_id, f"10-K Item {item_id}")
        for item_id in _TARGET_ITEM_IDS
    )
    return FinbertSectionUniverseConfig(
        source_items_dir=source_items_dir,
        form_types=tuple(manifest_section.get("form_types") or _RAW_FORM_10K_ONLY),
        target_items=target_items,
        require_active_items=bool(manifest_section.get("require_active_items", True)),
        require_exists_by_regime=bool(manifest_section.get("require_exists_by_regime", True)),
        min_char_count=int(manifest_section.get("min_char_count") or 250),
    )


def _manifest_year_filter(run_manifest: dict[str, Any]) -> tuple[int, ...] | None:
    raw_year_filter = run_manifest.get("year_filter")
    if raw_year_filter is None:
        return None
    return _normalize_year_filter(tuple(int(year) for year in raw_year_filter))


def _benchmark_item_to_item_id_expr() -> pl.Expr:
    return (
        pl.when(pl.col("benchmark_item_code") == "item_1")
        .then(pl.lit("1"))
        .when(pl.col("benchmark_item_code") == "item_1a")
        .then(pl.lit("1A"))
        .when(pl.col("benchmark_item_code") == "item_7")
        .then(pl.lit("7"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )


def _summarize_extracted_target_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    longest_by_item: dict[str, int] = {}
    for item in items:
        item_id = str(item.get("item_id") or "").upper()
        if item_id not in _TARGET_ITEM_IDS:
            continue
        full_text = item.get("full_text")
        char_count = len(full_text) if isinstance(full_text, str) else 0
        longest_by_item[item_id] = max(longest_by_item.get(item_id, 0), char_count)
    item_ids = ",".join(sorted(longest_by_item))
    return {
        "item_ids": item_ids,
        "item_count": len(longest_by_item),
        "total_chars": sum(longest_by_item.values()),
    }
