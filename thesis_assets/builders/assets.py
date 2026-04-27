from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

from thesis_assets.builders.artifacts import resolve_required_artifacts
from thesis_assets.builders.artifacts import resolve_run
from thesis_assets.builders.artifacts import scan_parquet_artifact
from thesis_assets.builders.sentence_summaries import build_finbert_sentence_summary
from thesis_assets.builders.sentence_summaries import build_lm_negative_sentence_summary
from thesis_assets.builders.sentence_summaries import sentence_batch_size_from_env
from thesis_assets.builders.sample_contracts import common_row_comparison
from thesis_assets.builders.sample_contracts import common_success_comparison
from thesis_assets.builders.sample_contracts import raw_available
from thesis_assets.config.constants import DEFAULT_COMMON_SUCCESS_POLICY
from thesis_assets.config.constants import DEFAULT_COMPARISON_JOIN_KEYS
from thesis_assets.config.constants import ARTIFACT_KEY_TABLE_IA_II_RESULTS
from thesis_assets.config.constants import ARTIFACT_KEY_TRADING_STRATEGY_MONTHLY_RETURNS
from thesis_assets.config.constants import RUN_FAMILY_FINBERT_RUN
from thesis_assets.config.constants import RUN_FAMILY_LM2011_POST_REFINITIV
from thesis_assets.errors import AssetBuildError
from thesis_assets.figures import build_concordance_figure
from thesis_assets.figures import build_concordance_by_scope_figure
from thesis_assets.figures import build_ecdf_lines_figure
from thesis_assets.figures import build_metric_panel_ecdf_figure
from thesis_assets.figures import build_multi_series_line_figure
from thesis_assets.figures import build_sample_attrition_figure
from thesis_assets.figures import build_sample_bridge_figure
from thesis_assets.figures import build_sample_funnel_figure
from thesis_assets.renderers import write_csv_table
from thesis_assets.renderers import write_figure_bundle
from thesis_assets.renderers import write_latex_table
from thesis_assets.renderers import write_markdown_table
from thesis_assets.specs import ArtifactRequirement
from thesis_assets.specs import AssetSpec
from thesis_assets.specs import BuildContext
from thesis_assets.specs import BuildResult
from thesis_assets.specs import ResolvedArtifact
from thesis_pkg.core.sec.lm2011_dictionary import load_lm2011_word_list


TARGET_TEXT_SCOPES = ("item_7_mda", "item_1a_risk_factors")
TABLE_VI_DEPENDENT_VARIABLES = (
    "filing_period_excess_return",
    "abnormal_volume",
    "postevent_return_volatility",
)
TABLE_VI_SIGNAL_COLUMNS = (
    "h4n_inf_prop",
    "lm_negative_prop",
    "lm_positive_prop",
    "lm_uncertainty_prop",
    "lm_litigious_prop",
    "lm_modal_strong_prop",
    "lm_modal_weak_prop",
    "h4n_inf_tfidf",
    "lm_negative_tfidf",
    "lm_positive_tfidf",
    "lm_uncertainty_tfidf",
    "lm_litigious_tfidf",
    "lm_modal_strong_tfidf",
    "lm_modal_weak_tfidf",
)
TABLE_VI_EXPECTED_SPEC_COUNT = len(TABLE_VI_DEPENDENT_VARIABLES) * len(TABLE_VI_SIGNAL_COLUMNS)
PORTFOLIO_SPREAD_DEFINITION = "Q5 - Q1; Q5 = most negative filings; Q1 = least negative filings"
NW_LAG_GRID = (1, 2, 3, 4)
NW_SIG_5_ABS_T = 1.959963984540054
NW_SIG_10_ABS_T = 1.6448536269514722


def build_asset(context: BuildContext, spec: AssetSpec) -> BuildResult:
    context.logger.info("Building asset %s", spec.asset_id)
    try:
        artifact_map = resolve_required_artifacts(context, spec)
        if spec.builder_id == "chapter4_sample_attrition":
            return _build_chapter4_sample_attrition(context, spec, artifact_map)
        if spec.builder_id == "chapter4_sample_funnel":
            return _build_chapter4_sample_funnel(context, spec, artifact_map)
        if spec.builder_id == "chapter4_sample_attrition_losses":
            return _build_chapter4_sample_attrition_losses(context, spec, artifact_map)
        if spec.builder_id == "chapter4_sample_stage_bridge":
            return _build_chapter4_sample_stage_bridge(context, spec, artifact_map)
        if spec.builder_id == "chapter4_full_10k_regression_sample_summary":
            return _build_chapter4_full_10k_regression_sample_summary(context, spec, artifact_map)
        if spec.builder_id == "chapter4_extension_attrition_ladder":
            return _build_chapter4_extension_attrition_ladder(context, spec, artifact_map)
        if spec.builder_id == "chapter4_no_ownership_c0_specification":
            return _build_chapter4_no_ownership_c0_specification(context, spec, artifact_map)
        if spec.builder_id == "chapter4_ownership_analyst_coverage_diagnostics":
            return _build_chapter4_ownership_analyst_coverage_diagnostics(context, spec, artifact_map)
        if spec.builder_id == "chapter4_ownership_coverage_by_year":
            return _build_chapter4_ownership_coverage_by_year(context, spec, artifact_map)
        if spec.builder_id == "chapter4_item_cleaning_eligibility_diagnostics":
            return _build_chapter4_item_cleaning_eligibility_diagnostics(context, spec, artifact_map)
        if spec.builder_id == "chapter4_item_cleaning_quality_by_year":
            return _build_chapter4_item_cleaning_quality_by_year(context, spec, artifact_map)
        if spec.builder_id == "chapter4_dictionary_provenance_summary":
            return _build_chapter4_dictionary_provenance_summary(context, spec, artifact_map)
        if spec.builder_id == "chapter4_finbert_inference_manifest_summary":
            return _build_chapter4_finbert_inference_manifest_summary(context, spec, artifact_map)
        if spec.builder_id == "chapter4_finbert_segment_token_diagnostics":
            return _build_chapter4_finbert_segment_token_diagnostics(context, spec, artifact_map)
        if spec.builder_id == "chapter4_score_family_descriptive_statistics":
            return _build_chapter4_score_family_descriptive_statistics(context, spec, artifact_map)
        if spec.builder_id == "chapter4_variable_definitions":
            return _build_chapter4_variable_definitions(context, spec, artifact_map)
        if spec.builder_id == "chapter5_lm2011_full_10k_return_coefficients":
            return _build_chapter5_lm2011_full_10k_return_coefficients(context, spec, artifact_map)
        if spec.builder_id == "chapter5_lm2011_portfolio_long_short":
            return _build_chapter5_lm2011_portfolio_long_short(context, spec, artifact_map)
        if spec.builder_id == "chapter5_lm2011_portfolio_formation_diagnostics":
            return _build_chapter5_lm2011_portfolio_formation_diagnostics(context, spec, artifact_map)
        if spec.builder_id == "chapter5_portfolio_cumulative_q5_minus_q1":
            return _build_chapter5_portfolio_cumulative_q5_minus_q1(context, spec, artifact_map)
        if spec.builder_id == "chapter5_fit_horserace":
            return _build_chapter5_fit_horserace(context, spec, artifact_map)
        if spec.builder_id == "chapter5_extension_c0_fit_summary":
            return _build_chapter5_extension_c0_fit_summary(context, spec, artifact_map)
        if spec.builder_id == "chapter5_extension_c0_fit_comparisons":
            return _build_chapter5_extension_c0_fit_comparisons(context, spec, artifact_map)
        if spec.builder_id == "chapter5_extension_fit_delta_path":
            return _build_chapter5_extension_fit_delta_path(context, spec, artifact_map)
        if spec.builder_id == "chapter5_lm2011_table_vi_no_ownership":
            return _build_chapter5_lm2011_table_vi_no_ownership(context, spec, artifact_map)
        if spec.builder_id == "chapter5_nw_lag_baseline_reconciliation":
            return _build_chapter5_nw_lag_baseline_reconciliation(context, spec, artifact_map)
        if spec.builder_id == "chapter5_nw_lag_core_no_ownership_appendix":
            return _build_chapter5_nw_lag_core_no_ownership_appendix(context, spec, artifact_map)
        if spec.builder_id == "chapter5_nw_lag_extension_coefficients_appendix":
            return _build_chapter5_nw_lag_extension_coefficients_appendix(context, spec, artifact_map)
        if spec.builder_id == "chapter5_nw_lag_extension_fit_comparisons_appendix":
            return _build_chapter5_nw_lag_extension_fit_comparisons_appendix(context, spec, artifact_map)
        if spec.builder_id == "chapter5_concordance":
            return _build_chapter5_concordance(context, spec, artifact_map)
        if spec.builder_id == "chapter5_lm_doc_score_ecdf":
            return _build_chapter5_lm_doc_score_ecdf(context, spec, artifact_map)
        if spec.builder_id == "chapter5_finbert_doc_score_ecdf":
            return _build_chapter5_finbert_doc_score_ecdf(context, spec, artifact_map)
        if spec.builder_id == "chapter5_finbert_sentence_ecdf":
            return _build_chapter5_finbert_sentence_ecdf(context, spec, artifact_map)
        if spec.builder_id == "chapter5_lm_sentence_ecdf":
            return _build_chapter5_lm_sentence_ecdf(context, spec, artifact_map)
        if spec.builder_id == "chapter5_high_negative_sentence_share":
            return _build_chapter5_high_negative_sentence_share(context, spec, artifact_map)
        if spec.builder_id == "chapter5_concordance_by_scope":
            return _build_chapter5_concordance_by_scope(context, spec, artifact_map)
        if spec.builder_id == "chapter5_score_drift_by_year":
            return _build_chapter5_score_drift_by_year(context, spec, artifact_map)
        if spec.builder_id == "chapter5_finbert_robustness_coefficients":
            return _build_chapter5_finbert_robustness_coefficients(context, spec, artifact_map)
        if spec.builder_id == "chapter5_finbert_robustness_fit_comparisons":
            return _build_chapter5_finbert_robustness_fit_comparisons(context, spec, artifact_map)
        if spec.builder_id == "chapter5_matched_dictionary_finbert_coefficients_full":
            return _build_chapter5_matched_dictionary_finbert_coefficients_full(context, spec, artifact_map)
        if spec.builder_id == "chapter5_fama_macbeth_skipped_quarter_diagnostics":
            return _build_chapter5_fama_macbeth_skipped_quarter_diagnostics(context, spec, artifact_map)
        if spec.builder_id == "chapter5_alternative_signal_robustness_full_grid":
            return _build_chapter5_alternative_signal_robustness_full_grid(context, spec, artifact_map)
        if spec.builder_id == "chapter5_full_controls_coefficient_appendix":
            return _build_chapter5_full_controls_coefficient_appendix(context, spec, artifact_map)
        if spec.builder_id == "chapter5_text_score_control_correlation_matrix":
            return _build_chapter5_text_score_control_correlation_matrix(context, spec, artifact_map)
        if spec.builder_id == "chapter5_research_question_evidence_map":
            return _build_chapter5_research_question_evidence_map(context, spec, artifact_map)
        raise AssetBuildError(f"Unsupported builder_id {spec.builder_id!r}")
    except Exception as exc:  # pragma: no cover - exercised through failure behavior tests
        context.logger.exception("Asset %s failed", spec.asset_id)
        return BuildResult(
            asset_id=spec.asset_id,
            chapter=spec.chapter,
            asset_kind=spec.asset_kind,
            sample_contract_id=spec.sample_contract_id,
            status="failed",
            resolved_inputs={},
            output_paths={},
            row_counts={},
            failure_reason=str(exc),
        )


def _build_chapter4_sample_attrition(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "raw_available")
    source_artifact = artifacts["table_i_sample_creation"]
    selected = (
        raw_available(scan_parquet_artifact(source_artifact))
        .sort("section_order", "row_order")
        .select(
            pl.col("section_label").alias("section"),
            pl.col("display_label").alias("row_label"),
            pl.col("sample_size_value").alias("sample_size"),
            pl.col("observations_removed").alias("observations_removed"),
            pl.col("availability_status").alias("availability_status"),
            pl.col("availability_reason").alias("availability_reason"),
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Chapter 4 sample attrition selection returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"table_i_sample_creation": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter4_sample_funnel(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "raw_available")
    source_artifact = artifacts["table_i_sample_creation"]
    selected = _sample_creation_lf(source_artifact).filter(
        (pl.col("sample_size_kind").cast(pl.Utf8, strict=False) == pl.lit("count"))
        & ~pl.col("display_label").str.contains("Number of unique firms", literal=True)
        & ~pl.col("display_label").str.contains("Average number of years", literal=True)
    ).collect()
    if selected.is_empty():
        raise AssetBuildError("Chapter 4 sample funnel selection returned zero rows.")

    figure = build_sample_funnel_figure(selected)
    output_paths = _write_figure_outputs(context, spec, selected, figure)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"table_i_sample_creation": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"figure_rows": selected.height},
    )


def _build_chapter4_sample_attrition_losses(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "raw_available")
    source_artifact = artifacts["table_i_sample_creation"]
    selected = (
        _sample_creation_lf(source_artifact)
        .filter(pl.col("observations_removed").cast(pl.Float64, strict=False).fill_null(0.0) > 0.0)
        .with_columns(pl.col("observations_removed").cast(pl.Float64, strict=False))
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Chapter 4 sample attrition loss selection returned zero rows.")

    figure = build_sample_attrition_figure(selected)
    output_paths = _write_figure_outputs(context, spec, selected, figure)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"table_i_sample_creation": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"figure_rows": selected.height},
    )


def _build_chapter4_sample_stage_bridge(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "raw_available")
    source_artifact = artifacts["table_i_sample_creation"]
    base = _sample_creation_lf(source_artifact).collect()
    selected = _sample_stage_bridge_rows(base)
    if selected.is_empty():
        raise AssetBuildError("Chapter 4 sample stage bridge selection returned zero rows.")

    figure = build_sample_bridge_figure(selected)
    output_paths = _write_figure_outputs(context, spec, selected, figure)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"table_i_sample_creation": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"figure_rows": selected.height},
    )


def _sample_creation_lf(source_artifact: ResolvedArtifact) -> pl.LazyFrame:
    return (
        raw_available(scan_parquet_artifact(source_artifact))
        .sort("section_order", "row_order")
        .select(
            pl.col("section_label"),
            pl.col("section_order"),
            pl.col("row_order"),
            pl.col("display_label"),
            pl.col("sample_size_value").cast(pl.Float64, strict=False),
            pl.col("sample_size_kind").cast(pl.Utf8, strict=False),
            pl.col("observations_removed").cast(pl.Float64, strict=False),
        )
    )


def _sample_stage_bridge_rows(df: pl.DataFrame) -> pl.DataFrame:
    stages = (
        ("Raw 10-K filings", lambda row: int(row["section_order"]) == 1 and int(row["row_order"]) == 1),
        ("Final full 10-K sample", lambda row: int(row["section_order"]) == 1),
        ("Firm-year sample", lambda row: str(row["display_label"]) == "Firm-Year Sample"),
        ("MD&A identified", lambda row: "MD&A section could be identified" in str(row["display_label"])),
        ("MD&A >= 250 words", lambda row: "MD&A section >= 250 words" in str(row["display_label"])),
    )
    rows: list[dict[str, object]] = []
    row_dicts = df.sort("section_order", "row_order").to_dicts()
    for order, (stage_label, predicate) in enumerate(stages, start=1):
        matches = [row for row in row_dicts if predicate(row)]
        if not matches:
            continue
        row = matches[-1]
        rows.append(
            {
                "bridge_order": order,
                "bridge_stage": stage_label,
                "source_section": row["section_label"],
                "source_row_label": row["display_label"],
                "sample_size_value": row["sample_size_value"],
            }
        )
    return pl.DataFrame(rows).sort("bridge_order") if rows else pl.DataFrame()


def _build_chapter4_full_10k_regression_sample_summary(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    source_artifact = artifacts["return_regression_panel_full_10k"]
    summary = _summary_statistics_table(
        scan_parquet_artifact(source_artifact),
        variables=(
            ("filing_period_excess_return", "Filing-period excess return"),
            ("abnormal_volume", "Abnormal volume"),
            ("postevent_return_volatility", "Postevent return volatility"),
            ("total_token_count_full_10k", "Total full-10-K tokens"),
            ("token_count_full_10k", "Recognized dictionary tokens"),
            ("lm_negative_prop", "LM negative proportion"),
            ("lm_negative_tfidf", "LM negative tf-idf"),
            ("h4n_inf_prop", "H4N-Inf proportion"),
            ("h4n_inf_tfidf", "H4N-Inf tf-idf"),
            ("log_size", "Log size"),
            ("log_book_to_market", "Log book-to-market"),
            ("log_share_turnover", "Log share turnover"),
            ("pre_ffalpha", "Pre-filing FF alpha"),
            ("nasdaq_dummy", "NASDAQ indicator"),
        ),
    )
    if summary.is_empty():
        raise AssetBuildError("Full-10-K regression sample summary returned zero rows.")

    output_paths = _write_table_outputs(context, spec, summary)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"return_regression_panel_full_10k": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": summary.height},
    )


def _build_chapter4_extension_attrition_ladder(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    loss_artifact = artifacts["extension_sample_loss"]
    panel_artifact = artifacts["extension_analysis_panel"]
    loss_lf = scan_parquet_artifact(loss_artifact)
    panel_lf = scan_parquet_artifact(panel_artifact)
    sample_loss = (
        loss_lf.filter(
            (pl.col("text_scope").is_in(TARGET_TEXT_SCOPES))
            & (pl.col("outcome_name") == pl.lit("filing_period_excess_return"))
            & (pl.col("control_set_id") == pl.lit("C0"))
            & (pl.col("feature_family") == pl.lit("dictionary_plus_finbert"))
            & (pl.col("specification_name") == pl.lit("dictionary_finbert_joint"))
        )
        .group_by("text_scope")
        .agg(
            pl.col("n_control_set_rows").sum().alias("control_set_rows"),
            pl.col("n_outcome_available").sum().alias("outcome_available_rows"),
            pl.col("n_signal_available").sum().alias("signal_available_rows"),
            pl.col("n_controls_available").sum().alias("controls_available_rows"),
            pl.col("n_industry_available").sum().alias("industry_available_rows"),
            pl.col("n_estimation_rows").sum().alias("estimation_rows"),
        )
    )
    matched = (
        panel_lf.filter(pl.col("text_scope").is_in(TARGET_TEXT_SCOPES))
        .with_columns(pl.col("filing_date").dt.year().alias("filing_year"))
        .group_by("text_scope")
        .agg(
            pl.len().alias("matched_panel_rows"),
            pl.col("doc_id").n_unique().alias("unique_docs"),
            pl.col("filing_year").min().alias("first_year"),
            pl.col("filing_year").max().alias("last_year"),
        )
    )
    selected = (
        sample_loss.join(matched, on="text_scope", how="outer", coalesce=True)
        .with_columns(_scope_label_expr().alias("scope"))
        .sort("text_scope")
        .select(
            "scope",
            "first_year",
            "last_year",
            "unique_docs",
            "matched_panel_rows",
            "control_set_rows",
            "outcome_available_rows",
            "signal_available_rows",
            "controls_available_rows",
            "industry_available_rows",
            "estimation_rows",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Extension attrition ladder returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={
            "extension_sample_loss": str(loss_artifact.path),
            "extension_analysis_panel": str(panel_artifact.path),
        },
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter4_no_ownership_c0_specification(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    source_artifact = artifacts["extension_control_ladder"]
    control_ladder = scan_parquet_artifact(source_artifact)
    c0_rows = (
        control_ladder.filter(pl.col("control_set_id") == pl.lit("C0"))
        .with_columns(
            pl.lit("C0 no-ownership").alias("specification"),
            pl.lit("log_size; log_book_to_market; log_share_turnover; pre_ffalpha; nasdaq_dummy").alias(
                "base_controls"
            ),
            pl.lit("FF48 industry dummies").alias("fixed_effects"),
            pl.lit("Quarterly Fama-MacBeth, Newey-West lag 1").alias("estimator"),
            pl.col("includes_ownership_control").cast(pl.Utf8, strict=False).alias("includes_ownership_control_text"),
        )
        .select(
            "control_set_id",
            "specification",
            "spec_alias",
            "base_controls",
            "fixed_effects",
            "estimator",
            "sample_rule",
            "common_support_ownership",
            pl.col("includes_ownership_control_text").alias("includes_ownership_control"),
        )
        .collect()
    )
    if c0_rows.is_empty():
        raise AssetBuildError("No C0 row found in extension control ladder.")

    output_paths = _write_table_outputs(context, spec, c0_rows)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_control_ladder": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": c0_rows.height},
    )


def _build_chapter4_ownership_analyst_coverage_diagnostics(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    panel_artifact = artifacts["extension_analysis_panel"]
    panel_rows = (
        scan_parquet_artifact(panel_artifact)
        .filter(pl.col("text_scope").is_in(TARGET_TEXT_SCOPES))
        .with_columns(pl.col("filing_date").dt.year().alias("calendar_year"))
        .group_by("calendar_year", "text_scope")
        .agg(
            pl.len().alias("rows"),
            pl.col("doc_id").n_unique().alias("unique_docs"),
            pl.col("ownership_proxy_available").cast(pl.Int64).sum().alias("ownership_available_rows"),
            pl.col("common_support_flag_ownership").cast(pl.Int64).sum().alias("ownership_common_support_rows"),
        )
        .with_columns(
            pl.lit("extension_analysis_panel").alias("coverage_source"),
            _scope_label_expr().alias("scope"),
            (pl.col("ownership_available_rows") / pl.col("rows")).alias("ownership_available_rate"),
            (pl.col("ownership_common_support_rows") / pl.col("rows")).alias("ownership_common_support_rate"),
            pl.lit(None).cast(pl.Int64).alias("analyst_request_rows"),
            pl.lit(None).cast(pl.Int64).alias("analyst_normalized_rows"),
        )
        .select(
            "coverage_source",
            "calendar_year",
            "text_scope",
            "scope",
            "unique_docs",
            "rows",
            "ownership_available_rows",
            "ownership_available_rate",
            "ownership_common_support_rows",
            "ownership_common_support_rate",
            "analyst_request_rows",
            "analyst_normalized_rows",
        )
    )
    analyst_rows = _analyst_coverage_rows(context)
    frames = [panel_rows]
    warnings: list[str] = []
    if analyst_rows is not None:
        frames.append(analyst_rows.lazy())
    else:
        warnings.append("No Refinitiv analyst coverage artifact was found; analyst columns are left blank.")
    selected = (
        pl.concat(frames, how="vertical_relaxed")
        .sort("coverage_source", "calendar_year", "text_scope")
        .select(
            "coverage_source",
            "calendar_year",
            "scope",
            "unique_docs",
            "rows",
            "ownership_available_rows",
            "ownership_available_rate",
            "ownership_common_support_rows",
            "ownership_common_support_rate",
            "analyst_request_rows",
            "analyst_normalized_rows",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Ownership/analyst coverage diagnostics returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_analysis_panel": str(panel_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
        warnings=tuple(warnings),
    )


def _build_chapter4_ownership_coverage_by_year(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    panel_artifact = artifacts["extension_analysis_panel"]
    base = (
        scan_parquet_artifact(panel_artifact)
        .filter(pl.col("text_scope").is_in(TARGET_TEXT_SCOPES))
        .select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("filing_date").cast(pl.Date, strict=False).alias("filing_date"),
            pl.col("text_scope").cast(pl.Utf8, strict=False),
            pl.col("ownership_proxy_available").cast(pl.Int64, strict=False).alias("ownership_available"),
            pl.col("common_support_flag_ownership").cast(pl.Int64, strict=False).alias("ownership_common_support"),
        )
        .with_columns(pl.col("filing_date").dt.year().alias("calendar_year"))
        .group_by("calendar_year", "text_scope")
        .agg(
            pl.len().alias("panel_rows"),
            pl.col("doc_id").n_unique().alias("unique_docs"),
            pl.col("ownership_available").sum().alias("ownership_available_rows"),
            pl.col("ownership_common_support").sum().alias("ownership_common_support_rows"),
        )
        .with_columns(_scope_label_expr().alias("scope"))
    )
    long_lf = pl.concat(
        [
            base.select(
                "calendar_year",
                "text_scope",
                "scope",
                pl.lit("unrestricted_panel").alias("coverage_metric"),
                pl.lit("Unrestricted C0 panel").alias("metric_label"),
                pl.col("panel_rows").alias("numerator_rows"),
                pl.col("panel_rows").alias("denominator_rows"),
                pl.lit(1.0).alias("coverage_rate"),
                pl.lit(1).alias("_metric_order"),
            ),
            base.select(
                "calendar_year",
                "text_scope",
                "scope",
                pl.lit("ownership_available").alias("coverage_metric"),
                pl.lit("Ownership proxy available").alias("metric_label"),
                pl.col("ownership_available_rows").alias("numerator_rows"),
                pl.col("panel_rows").alias("denominator_rows"),
                (pl.col("ownership_available_rows") / pl.col("panel_rows")).alias("coverage_rate"),
                pl.lit(2).alias("_metric_order"),
            ),
            base.select(
                "calendar_year",
                "text_scope",
                "scope",
                pl.lit("ownership_common_support").alias("coverage_metric"),
                pl.lit("Ownership common support").alias("metric_label"),
                pl.col("ownership_common_support_rows").alias("numerator_rows"),
                pl.col("panel_rows").alias("denominator_rows"),
                (pl.col("ownership_common_support_rows") / pl.col("panel_rows")).alias("coverage_rate"),
                pl.lit(3).alias("_metric_order"),
            ),
        ],
        how="vertical_relaxed",
    ).with_columns(
        pl.concat_str([pl.col("scope"), pl.lit(" | "), pl.col("metric_label")]).alias("series_label")
    )
    selected = (
        long_lf.sort("calendar_year", "text_scope", "_metric_order")
        .select(
            "calendar_year",
            "scope",
            "coverage_metric",
            "metric_label",
            "numerator_rows",
            "denominator_rows",
            "coverage_rate",
            "series_label",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Ownership coverage-by-year figure returned zero rows.")

    figure = build_multi_series_line_figure(
        selected,
        x_col="calendar_year",
        y_col="coverage_rate",
        series_col="series_label",
        x_label="Filing year",
        y_label="Coverage rate",
    )
    output_paths = _write_figure_outputs(context, spec, selected, figure)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_analysis_panel": str(panel_artifact.path)},
        output_paths=output_paths,
        row_counts={"figure_rows": selected.height},
        warnings=(
            "Coverage rates are diagnostics only; C1/C2 ownership-conditioned specifications remain outside the central thesis tables.",
        ),
    )


def _build_chapter4_item_cleaning_eligibility_diagnostics(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    finbert_artifact = artifacts["model_inference_yearly_summary"]
    diagnostics_path = _resolve_finbert_preprocessing_path(
        context,
        finbert_artifact,
        "item_scope_cleaning_diagnostics.parquet",
    )
    count_columns = (
        "n_filings_candidate",
        "n_filings_extracted",
        "n_rows_after_cleaning",
        "toc_trimmed_rows",
        "tail_truncated_rows",
        "reference_stub_rows",
        "empty_after_cleaning_rows",
        "large_removal_warning_rows",
        "manual_audit_queue_n",
    )
    token_weight = pl.col("n_rows_after_cleaning").cast(pl.Float64)
    diagnostics = (
        pl.scan_parquet(str(diagnostics_path))
        .filter(pl.col("text_scope").is_in(TARGET_TEXT_SCOPES))
        .group_by("calendar_year", "text_scope")
        .agg(
            *(pl.col(column).sum().alias(column) for column in count_columns),
            ((pl.col("token_count_mean") * token_weight).sum() / token_weight.sum()).alias("token_count_mean"),
            ((pl.col("token_count_median") * token_weight).sum() / token_weight.sum()).alias("token_count_median"),
            ((pl.col("token_count_p05") * token_weight).sum() / token_weight.sum()).alias("token_count_p05"),
            pl.col("activation_status").unique().sort().alias("_activation_statuses"),
        )
        .with_columns(
            (pl.col("n_filings_extracted") / pl.col("n_filings_candidate")).alias("extraction_rate"),
            _scope_label_expr().alias("scope"),
            pl.col("_activation_statuses").list.join("; ").alias("activation_status"),
        )
        .sort("calendar_year", "text_scope")
        .select(
            "calendar_year",
            "scope",
            "n_filings_candidate",
            "n_filings_extracted",
            "extraction_rate",
            "n_rows_after_cleaning",
            "token_count_mean",
            "token_count_median",
            "token_count_p05",
            "toc_trimmed_rows",
            "tail_truncated_rows",
            "reference_stub_rows",
            "empty_after_cleaning_rows",
            "large_removal_warning_rows",
            "manual_audit_queue_n",
            "activation_status",
        )
        .collect()
    )
    if diagnostics.is_empty():
        raise AssetBuildError("Item cleaning diagnostics returned zero Item 1A/Item 7 rows.")

    output_paths = _write_table_outputs(context, spec, diagnostics)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={
            "model_inference_yearly_summary": str(finbert_artifact.path),
            "item_scope_cleaning_diagnostics": str(diagnostics_path),
        },
        output_paths=output_paths,
        row_counts={"table_rows": diagnostics.height},
    )


def _build_chapter4_item_cleaning_quality_by_year(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    finbert_artifact = artifacts["model_inference_yearly_summary"]
    diagnostics_path = _resolve_finbert_preprocessing_path(
        context,
        finbert_artifact,
        "item_scope_cleaning_diagnostics.parquet",
    )
    base = (
        pl.scan_parquet(str(diagnostics_path))
        .filter(pl.col("text_scope").is_in(TARGET_TEXT_SCOPES))
        .group_by("calendar_year", "text_scope")
        .agg(
            pl.col("n_filings_candidate").sum().alias("candidate_filings"),
            pl.col("n_filings_extracted").sum().alias("extracted_filings"),
            pl.col("n_rows_after_cleaning").sum().alias("cleaned_rows"),
            pl.col("manual_audit_queue_n").sum().alias("manual_review_rows"),
        )
        .with_columns(_scope_label_expr().alias("scope"))
    )
    long_lf = pl.concat(
        [
            base.select(
                "calendar_year",
                "text_scope",
                "scope",
                pl.lit("extraction_rate").alias("quality_metric"),
                pl.lit("Extraction rate").alias("metric_label"),
                pl.col("extracted_filings").alias("numerator_rows"),
                pl.col("candidate_filings").alias("denominator_rows"),
                (pl.col("extracted_filings") / pl.col("candidate_filings")).alias("metric_value"),
                pl.lit(1).alias("_metric_order"),
            ),
            base.select(
                "calendar_year",
                "text_scope",
                "scope",
                pl.lit("cleaned_scope_rate").alias("quality_metric"),
                pl.lit("Cleaned-scope rate").alias("metric_label"),
                pl.col("cleaned_rows").alias("numerator_rows"),
                pl.col("candidate_filings").alias("denominator_rows"),
                (pl.col("cleaned_rows") / pl.col("candidate_filings")).alias("metric_value"),
                pl.lit(2).alias("_metric_order"),
            ),
            base.select(
                "calendar_year",
                "text_scope",
                "scope",
                pl.lit("manual_review_share").alias("quality_metric"),
                pl.lit("Manual-review queue share").alias("metric_label"),
                pl.col("manual_review_rows").alias("numerator_rows"),
                pl.col("candidate_filings").alias("denominator_rows"),
                (pl.col("manual_review_rows") / pl.col("candidate_filings")).alias("metric_value"),
                pl.lit(3).alias("_metric_order"),
            ),
        ],
        how="vertical_relaxed",
    ).with_columns(
        pl.concat_str([pl.col("scope"), pl.lit(" | "), pl.col("metric_label")]).alias("series_label")
    )
    selected = (
        long_lf.sort("calendar_year", "text_scope", "_metric_order")
        .select(
            "calendar_year",
            "scope",
            "quality_metric",
            "metric_label",
            "numerator_rows",
            "denominator_rows",
            "metric_value",
            "series_label",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Item cleaning quality-by-year figure returned zero rows.")

    figure = build_multi_series_line_figure(
        selected,
        x_col="calendar_year",
        y_col="metric_value",
        series_col="series_label",
        x_label="Filing year",
        y_label="Rate",
    )
    output_paths = _write_figure_outputs(context, spec, selected, figure)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={
            "model_inference_yearly_summary": str(finbert_artifact.path),
            "item_scope_cleaning_diagnostics": str(diagnostics_path),
        },
        output_paths=output_paths,
        row_counts={"figure_rows": selected.height},
    )


def _build_chapter4_dictionary_provenance_summary(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    dictionary_artifact = artifacts["extension_dictionary_surface"]
    generated_root = _resolve_generated_dictionary_family_root(context)
    rows: list[dict[str, object]] = []
    for family_dir in sorted(path for path in generated_root.iterdir() if path.is_dir()):
        list_files = sorted(path for path in family_dir.glob("*.txt") if path.is_file())
        rows.append(
            {
                "dictionary_family": family_dir.name,
                "source": str(family_dir),
                "list_count": len(list_files),
                "list_names": "; ".join(path.stem for path in list_files),
                "total_terms": sum(_count_nonblank_lines(path) for path in list_files),
            }
        )
    surface_counts = (
        scan_parquet_artifact(dictionary_artifact)
        .group_by("dictionary_family", "text_scope")
        .len()
        .collect()
    )
    family_docs = (
        surface_counts.group_by("dictionary_family")
        .agg(
            pl.col("len").sum().alias("surface_rows"),
            pl.col("text_scope").n_unique().alias("scope_count"),
        )
    )
    selected = pl.DataFrame(rows).join(family_docs, on="dictionary_family", how="left").sort("dictionary_family")
    if selected.is_empty():
        raise AssetBuildError("Dictionary provenance summary returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={
            "extension_dictionary_surface": str(dictionary_artifact.path),
            "generated_dictionary_families": str(generated_root),
        },
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter4_finbert_inference_manifest_summary(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    summary_artifact = artifacts["model_inference_yearly_summary"]
    run = resolve_run(context, RUN_FAMILY_FINBERT_RUN)
    manifest = run.manifest or {}
    yearly = scan_parquet_artifact(summary_artifact)
    status_summary = (
        yearly.group_by("status")
        .agg(
            pl.len().alias("years"),
            pl.col("sentence_rows").sum().alias("sentence_rows"),
            pl.col("item_feature_rows").sum().alias("item_feature_rows"),
            pl.col("doc_rows").sum().alias("doc_rows"),
        )
        .with_columns(pl.lit("yearly_status").alias("section"), pl.col("status").alias("metric"))
        .select("section", "metric", "years", "sentence_rows", "item_feature_rows", "doc_rows")
    )
    manifest_rows = pl.DataFrame(
        [
            {
                "section": "manifest",
                "metric": "run_name",
                "years": None,
                "sentence_rows": None,
                "item_feature_rows": None,
                "doc_rows": None,
                "value": manifest.get("run_name"),
            },
            {
                "section": "manifest",
                "metric": "model",
                "years": None,
                "sentence_rows": None,
                "item_feature_rows": None,
                "doc_rows": None,
                "value": _manifest_nested_value(manifest, ("runtime", "model_name"))
                or _manifest_nested_value(manifest, ("authority", "model_name"))
                or _manifest_nested_value(manifest, ("batch_config", "model_name")),
            },
            {
                "section": "manifest",
                "metric": "bucket_lengths",
                "years": None,
                "sentence_rows": None,
                "item_feature_rows": None,
                "doc_rows": None,
                "value": json.dumps(manifest.get("bucket_lengths"), sort_keys=True)
                if manifest.get("bucket_lengths") is not None
                else None,
            },
        ],
        schema={
            "section": pl.Utf8,
            "metric": pl.Utf8,
            "years": pl.Int64,
            "sentence_rows": pl.Int64,
            "item_feature_rows": pl.Int64,
            "doc_rows": pl.Int64,
            "value": pl.Utf8,
        },
    )
    selected = pl.concat(
        [status_summary.collect().with_columns(pl.lit(None).cast(pl.Utf8).alias("value")), manifest_rows],
        how="vertical_relaxed",
    )
    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={
            "model_inference_yearly_summary": str(summary_artifact.path),
            "run_manifest": str(run.manifest_path) if run.manifest_path is not None else "",
        },
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter4_finbert_segment_token_diagnostics(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    source_artifact = artifacts["item_features_long"]
    selected = (
        scan_parquet_artifact(source_artifact)
        .filter(pl.col("text_scope").is_in(TARGET_TEXT_SCOPES))
        .select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("filing_year").cast(pl.Int64, strict=False),
            pl.col("text_scope").cast(pl.Utf8, strict=False),
            pl.col("sentence_count").cast(pl.Float64, strict=False),
            pl.col("finbert_segment_count").cast(pl.Float64, strict=False),
            pl.col("finbert_token_count_512_sum").cast(pl.Float64, strict=False),
        )
        .group_by("filing_year", "text_scope")
        .agg(
            pl.len().alias("item_scope_rows"),
            pl.col("doc_id").n_unique().alias("unique_docs"),
            pl.col("sentence_count").mean().alias("sentence_count_mean"),
            pl.col("sentence_count").median().alias("sentence_count_median"),
            pl.col("finbert_segment_count").mean().alias("segment_count_mean"),
            pl.col("finbert_segment_count").median().alias("segment_count_median"),
            pl.col("finbert_segment_count").quantile(0.90).alias("segment_count_p90"),
            pl.col("finbert_token_count_512_sum").mean().alias("token_count_512_mean"),
            pl.col("finbert_token_count_512_sum").median().alias("token_count_512_median"),
            pl.col("finbert_token_count_512_sum").quantile(0.90).alias("token_count_512_p90"),
        )
        .with_columns(_scope_label_expr().alias("scope"))
        .sort("filing_year", "text_scope")
        .select(
            "filing_year",
            "scope",
            "item_scope_rows",
            "unique_docs",
            "sentence_count_mean",
            "sentence_count_median",
            "segment_count_mean",
            "segment_count_median",
            "segment_count_p90",
            "token_count_512_mean",
            "token_count_512_median",
            "token_count_512_p90",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("FinBERT segment/token diagnostics returned zero Item 1A/Item 7 rows.")

    figure_df = selected.select(
        "filing_year",
        pl.col("scope").alias("series_label"),
        pl.col("token_count_512_median").alias("metric_value"),
    )
    figure = build_multi_series_line_figure(
        figure_df,
        x_col="filing_year",
        y_col="metric_value",
        series_col="series_label",
        x_label="Filing year",
        y_label="Median FinBERT 512-token count",
    )
    output_paths = _write_figure_outputs(context, spec, selected, figure)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"item_features_long": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"summary_rows": selected.height},
    )


def _build_chapter4_score_family_descriptive_statistics(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    frames = [
        _score_summary_lf(
            scan_parquet_artifact(artifacts["return_regression_panel_full_10k"]),
            source_label="LM2011 benchmark",
            score_columns=("lm_negative_prop", "lm_negative_tfidf", "h4n_inf_prop", "h4n_inf_tfidf"),
        ),
        _score_summary_lf(
            scan_parquet_artifact(artifacts["return_regression_panel_mda"]),
            source_label="LM2011 MD&A",
            score_columns=("lm_negative_prop", "lm_negative_tfidf", "h4n_inf_prop", "h4n_inf_tfidf"),
        ),
        _score_summary_lf(
            scan_parquet_artifact(artifacts["extension_dictionary_surface"]),
            source_label="Extension dictionary",
            score_columns=("lm_negative_prop", "lm_negative_tfidf"),
        ),
        _score_summary_lf(
            scan_parquet_artifact(artifacts["extension_finbert_surface"]),
            source_label="Extension FinBERT",
            score_columns=("finbert_neg_prob_lenw_mean", "finbert_net_negative_lenw_mean"),
        ),
    ]
    selected = pl.concat(frames, how="vertical_relaxed").sort("source", "text_scope", "score_name").collect()
    if selected.is_empty():
        raise AssetBuildError("Score family descriptive statistics returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={key: str(value.path) for key, value in artifacts.items()},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter4_variable_definitions(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    rows = [
        {
            "category": "Outcome",
            "variable_or_rule": "filing_period_excess_return",
            "definition": "Buy-and-hold filing-period excess return used in the extension return regressions.",
            "source_or_artifact": "lm2011_extension_analysis_panel / extension regression outputs",
            "reporting_note": "Primary Chapter 5 outcome.",
        },
        {
            "category": "Text signal",
            "variable_or_rule": "lm_negative_tfidf",
            "definition": "LM2011 negative-word score with tf-idf weighting, measured on the selected text scope.",
            "source_or_artifact": "lm2011_extension_dictionary_surface",
            "reporting_note": "Dictionary benchmark signal.",
        },
        {
            "category": "Text signal",
            "variable_or_rule": "finbert_neg_prob_lenw_mean",
            "definition": "Length-weighted mean FinBERT negative probability across sentences in the selected item scope.",
            "source_or_artifact": "item_features_long / lm2011_extension_finbert_surface",
            "reporting_note": "FinBERT comparison signal.",
        },
        {
            "category": "Control set",
            "variable_or_rule": "C0",
            "definition": "No-ownership control set: log size, log book-to-market, log share turnover, pre-filing FF alpha, NASDAQ indicator, and FF48 industry dummies.",
            "source_or_artifact": "lm2011_extension_control_ladder",
            "reporting_note": "Central specification because C1/C2 coverage is limited.",
        },
        {
            "category": "Portfolio convention",
            "variable_or_rule": "Q5 - Q1",
            "definition": PORTFOLIO_SPREAD_DEFINITION,
            "source_or_artifact": "lm2011_trading_strategy_monthly_returns",
            "reporting_note": "Do not flip signs when reporting portfolio spreads.",
        },
        {
            "category": "Timing",
            "variable_or_rule": "portfolio sort year",
            "definition": "Latest eligible filing by KYPERMNO/sort year; July-December filings map to the following sort year and January-June filings stay in the current sort year.",
            "source_or_artifact": "lm2011_trading_strategy_monthly_returns",
            "reporting_note": "Stored monthly returns contain Q5-Q1 only, not separate Q1/Q5 legs.",
        },
    ]
    selected = pl.DataFrame(rows)
    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_lm2011_full_10k_return_coefficients(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    source_artifact = artifacts["table_iv_results_no_ownership"]
    selected = (
        scan_parquet_artifact(source_artifact)
        .filter(
            (pl.col("text_scope") == pl.lit("full_10k"))
            & (pl.col("dependent_variable") == pl.lit("filing_period_excess_return"))
            & (pl.col("coefficient_name") == pl.col("signal_name"))
            & pl.col("signal_name").is_in(("h4n_inf_prop", "lm_negative_prop", "h4n_inf_tfidf", "lm_negative_tfidf"))
        )
        .with_columns(
            _signal_label_expr().alias("signal"),
            _table_vi_weighting_label_expr().alias("weighting"),
            _table_vi_weight_order_expr().alias("_weight_order"),
            _table_vi_signal_order_expr().alias("_signal_order"),
            (pl.col("estimate").cast(pl.Float64, strict=False) * 100.0).alias("estimate_x100"),
            (pl.col("standard_error").cast(pl.Float64, strict=False) * 100.0).alias("std_error_x100"),
            pl.col("t_stat").cast(pl.Float64, strict=False),
            _normal_p_value_expr(pl.col("t_stat")).alias("p_value_normal_approx"),
            pl.lit("C0 no ownership").alias("controls"),
            pl.lit("FF48 industry dummies").alias("fixed_effects"),
        )
        .sort("_weight_order", "_signal_order")
        .select(
            "signal",
            "weighting",
            "estimate_x100",
            "std_error_x100",
            "t_stat",
            "p_value_normal_approx",
            "n_quarters",
            "mean_quarter_n",
            "nw_lags",
            "controls",
            "fixed_effects",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Full-10-K return coefficient selection returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"table_iv_results_no_ownership": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_lm2011_portfolio_long_short(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    source_artifact, warning = _resolve_portfolio_table_artifact(context, artifacts.get("table_ia_ii_results"))
    selected = (
        scan_parquet_artifact(source_artifact)
        .filter(pl.col("coefficient_name").is_in(("mean_long_short_return", "alpha_ff3_mom", "r2")))
        .with_columns(
            _portfolio_signal_label_expr().alias("signal"),
            _portfolio_metric_label_expr().alias("metric"),
            pl.lit(PORTFOLIO_SPREAD_DEFINITION).alias("spread_definition"),
            pl.lit("not stored").alias("q1_return"),
            pl.lit("not stored").alias("q5_return"),
        )
        .sort("signal_name", "coefficient_name")
        .select(
            "signal",
            "metric",
            "estimate",
            "t_stat",
            "spread_definition",
            "q1_return",
            "q5_return",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Portfolio long-short table returned zero rows.")

    warnings = (warning,) if warning else ()
    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"table_ia_ii_results": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
        warnings=warnings,
    )


def _build_chapter5_lm2011_portfolio_formation_diagnostics(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    source_artifact, warning = _resolve_portfolio_monthly_artifact(
        context,
        artifacts.get("trading_strategy_monthly_returns"),
    )
    selected = (
        scan_parquet_artifact(source_artifact)
        .with_columns(pl.col("portfolio_month").cast(pl.Date, strict=False))
        .group_by("sort_signal_name")
        .agg(
            pl.len().alias("months"),
            pl.col("portfolio_month").min().alias("first_month"),
            pl.col("portfolio_month").max().alias("last_month"),
            pl.col("long_short_return").mean().alias("mean_long_short_return"),
            pl.col("long_short_return").std().alias("sd_long_short_return"),
            pl.col("long_short_return").min().alias("min_long_short_return"),
            pl.col("long_short_return").max().alias("max_long_short_return"),
        )
        .with_columns(
            _portfolio_sort_signal_label_expr().alias("sort_signal"),
            pl.lit("latest eligible filing by KYPERMNO/sort year; July-Dec filings map to next sort year").alias(
                "formation_rule"
            ),
            pl.lit(PORTFOLIO_SPREAD_DEFINITION).alias("long_short_sign"),
            pl.lit(False).alias("q1_q5_legs_stored"),
        )
        .sort("sort_signal_name")
        .select(
            "sort_signal",
            "months",
            "first_month",
            "last_month",
            "mean_long_short_return",
            "sd_long_short_return",
            "min_long_short_return",
            "max_long_short_return",
            "formation_rule",
            "long_short_sign",
            "q1_q5_legs_stored",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Portfolio formation diagnostics returned zero rows.")

    warnings = [warning] if warning else []
    warnings.append(
        "Stored monthly strategy artifact contains Q5 - Q1 high-minus-low returns only; Q1/Q5 leg diagnostics are unavailable."
    )
    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"trading_strategy_monthly_returns": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
        warnings=tuple(warnings),
    )


def _build_chapter5_portfolio_cumulative_q5_minus_q1(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    source_artifact, warning = _resolve_portfolio_monthly_artifact(
        context,
        artifacts.get("trading_strategy_monthly_returns"),
    )
    selected = (
        scan_parquet_artifact(source_artifact)
        .select(
            pl.col("portfolio_month").cast(pl.Date, strict=False).alias("portfolio_month"),
            pl.col("sort_signal_name").cast(pl.Utf8, strict=False),
            pl.col("long_short_return").cast(pl.Float64, strict=False),
        )
        .drop_nulls(subset=["portfolio_month", "sort_signal_name", "long_short_return"])
        .sort("sort_signal_name", "portfolio_month")
        .with_columns(
            _portfolio_sort_signal_label_expr().alias("signal"),
            (
                (pl.col("long_short_return") + 1.0).cum_prod().over("sort_signal_name") - 1.0
            ).alias("cumulative_q5_minus_q1_return"),
            pl.lit(PORTFOLIO_SPREAD_DEFINITION).alias("spread_definition"),
        )
        .select(
            "portfolio_month",
            "sort_signal_name",
            "signal",
            "long_short_return",
            "cumulative_q5_minus_q1_return",
            "spread_definition",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Portfolio cumulative Q5-Q1 figure returned zero monthly rows.")

    figure = build_multi_series_line_figure(
        selected,
        x_col="portfolio_month",
        y_col="cumulative_q5_minus_q1_return",
        series_col="signal",
        x_label="Portfolio month",
        y_label="Cumulative Q5 - Q1 return",
        zero_line=True,
    )
    warnings = [warning] if warning else []
    warnings.append(
        "Stored monthly strategy artifact contains Q5 - Q1 high-minus-low returns only; Q5 is most negative filings."
    )
    output_paths = _write_figure_outputs(context, spec, selected, figure)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"trading_strategy_monthly_returns": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"monthly_rows": selected.height},
        warnings=tuple(warnings),
    )


def _build_chapter5_fit_horserace(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "common_success_comparison")
    source_artifact = artifacts["extension_fit_summary"]
    fit_lf = scan_parquet_artifact(source_artifact)
    schema = fit_lf.collect_schema()
    if "dictionary_family_source" not in schema:
        fit_lf = fit_lf.with_columns(pl.lit("replication").alias("dictionary_family_source"))

    spec_order = (
        pl.when(pl.col("specification_name") == "dictionary_only")
        .then(pl.lit(1))
        .when(pl.col("specification_name") == "finbert_only")
        .then(pl.lit(2))
        .when(pl.col("specification_name") == "dictionary_finbert_joint")
        .then(pl.lit(3))
        .otherwise(pl.lit(99))
        .alias("_spec_order")
    )

    selected = (
        common_success_comparison(
            fit_lf,
            expected_policy=DEFAULT_COMMON_SUCCESS_POLICY,
            filters=(
                pl.col("text_scope") == pl.lit("item_7_mda"),
                pl.col("outcome_name") == pl.lit("filing_period_excess_return"),
                pl.col("control_set_id") == pl.lit("C0"),
                pl.col("dictionary_family_source") == pl.lit("replication"),
            ),
        )
        .with_columns(spec_order)
        .sort("_spec_order")
        .select(
            pl.col("feature_family").alias("feature_family"),
            pl.col("specification_name").alias("specification"),
            pl.col("signal_name").alias("signal"),
            pl.col("n_quarters").alias("n_quarters"),
            pl.col("total_n_obs").alias("total_n_obs"),
            pl.col("weighted_avg_adj_r2").alias("weighted_avg_adj_r2"),
            pl.col("equal_quarter_avg_adj_r2").alias("equal_quarter_avg_adj_r2"),
            pl.col("estimator_status").alias("estimator_status"),
            pl.col("failure_reason").alias("failure_reason"),
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Chapter 5 fit horserace selection returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_fit_summary": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"comparison_rows": selected.height},
    )


def _build_chapter5_extension_c0_fit_summary(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "common_success_comparison")
    source_artifact = artifacts["extension_fit_summary"]
    fit_lf = _ensure_dictionary_family_source(scan_parquet_artifact(source_artifact))

    selected = (
        common_success_comparison(
            fit_lf,
            expected_policy=DEFAULT_COMMON_SUCCESS_POLICY,
            filters=_extension_c0_fit_filters(),
        )
        .with_columns(
            _scope_label_expr().alias("scope"),
            _scope_order_expr().alias("_scope_order"),
            _dictionary_family_label_expr().alias("dictionary_family"),
            _dictionary_family_order_expr().alias("_dictionary_family_order"),
            _specification_label_expr().alias("specification"),
            _specification_order_expr().alias("_spec_order"),
            _fit_signal_label_expr("signal_name").alias("signal"),
        )
        .sort("_scope_order", "_dictionary_family_order", "_spec_order")
        .select(
            "scope",
            "dictionary_family",
            "specification",
            "signal",
            "n_quarters",
            "total_n_obs",
            "mean_quarter_n",
            "weighted_avg_adj_r2",
            "equal_quarter_avg_adj_r2",
            "estimator_status",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Chapter 5 extension C0 fit-summary selection returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_fit_summary": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_extension_c0_fit_comparisons(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "common_success_comparison")
    source_artifact = artifacts["extension_fit_comparisons"]
    comparison_lf = _ensure_dictionary_family_source(scan_parquet_artifact(source_artifact))

    selected = (
        common_success_comparison(
            comparison_lf,
            expected_policy=DEFAULT_COMMON_SUCCESS_POLICY,
            filters=_extension_c0_fit_filters(),
        )
        .with_columns(
            _scope_label_expr().alias("scope"),
            _scope_order_expr().alias("_scope_order"),
            _dictionary_family_label_expr().alias("dictionary_family"),
            _dictionary_family_order_expr().alias("_dictionary_family_order"),
            _comparison_label_expr().alias("comparison"),
            _comparison_order_expr().alias("_comparison_order"),
            _stars_expr(pl.col("nw_p_value_delta_adj_r2")).alias("stars"),
        )
        .sort("_scope_order", "_dictionary_family_order", "_comparison_order")
        .select(
            "scope",
            "dictionary_family",
            "comparison",
            pl.col("weighted_avg_delta_adj_r2").alias("delta_adj_r2_weighted"),
            pl.col("equal_quarter_avg_delta_adj_r2").alias("delta_adj_r2_equal_quarter"),
            pl.col("nw_t_stat_delta_adj_r2").alias("nw_t_stat"),
            pl.col("nw_p_value_delta_adj_r2").alias("nw_p_value"),
            "stars",
            "n_quarters",
            "total_n_obs",
            "mean_quarter_n",
            "estimator_status",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Chapter 5 extension C0 fit-comparison selection returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_fit_comparisons": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_extension_fit_delta_path(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "common_success_comparison")
    source_artifact = artifacts["extension_fit_difference_quarterly"]
    fit_lf = _ensure_dictionary_family_source(scan_parquet_artifact(source_artifact))
    selected = (
        common_success_comparison(
            fit_lf,
            expected_policy=DEFAULT_COMMON_SUCCESS_POLICY,
            filters=(
                pl.col("text_scope").is_in(TARGET_TEXT_SCOPES),
                pl.col("outcome_name") == pl.lit("filing_period_excess_return"),
                pl.col("control_set_id") == pl.lit("C0"),
                pl.col("dictionary_family_source") == pl.lit("replication"),
                pl.col("comparison_name").is_in(
                    ("finbert_minus_dictionary", "joint_minus_dictionary", "joint_minus_finbert")
                ),
            ),
        )
        .with_columns(
            pl.col("quarter_start").cast(pl.Date, strict=False).alias("quarter_start"),
            _scope_label_expr().alias("scope"),
            _comparison_label_expr().alias("comparison"),
            _scope_order_expr().alias("_scope_order"),
            _comparison_order_expr().alias("_comparison_order"),
        )
        .with_columns(pl.concat_str([pl.col("scope"), pl.lit(" | "), pl.col("comparison")]).alias("series_label"))
        .sort("_scope_order", "_comparison_order", "quarter_start")
        .select(
            "quarter_start",
            "scope",
            "comparison",
            "series_label",
            "n_obs",
            "delta_adj_r2",
            "common_success_policy",
            "dictionary_family_source",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Extension fit-delta path figure returned zero quarterly rows.")

    figure = build_multi_series_line_figure(
        selected,
        x_col="quarter_start",
        y_col="delta_adj_r2",
        series_col="series_label",
        x_label="Quarter",
        y_label="Adjusted-R2 delta",
        zero_line=True,
    )
    output_paths = _write_figure_outputs(context, spec, selected, figure)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_fit_difference_quarterly": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"quarterly_rows": selected.height},
    )


def _build_chapter5_lm2011_table_vi_no_ownership(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "lm2011_table_vi_no_ownership")
    source_artifact = artifacts["table_vi_results_no_ownership"]
    selected_with_keys, resolved_artifact, warnings = _collect_table_vi_no_ownership_surface(
        context,
        source_artifact,
    )
    selected = selected_with_keys.select(
        "outcome",
        "signal",
        "weighting",
        "estimate",
        "std_error",
        "t_stat",
        "n_quarters",
        "mean_quarter_n",
        "nw_lags",
        "reported_scale",
    )

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"table_vi_results_no_ownership": str(resolved_artifact.path)},
        output_paths=output_paths,
        row_counts={
            "table_rows": selected.height,
            "outcome_count": len(TABLE_VI_DEPENDENT_VARIABLES),
            "signal_count": len(TABLE_VI_SIGNAL_COLUMNS),
        },
        warnings=warnings,
    )


def _build_chapter5_nw_lag_baseline_reconciliation(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    core_sensitivity = artifacts["core_tables_nw_lag_sensitivity"]
    extension_sensitivity = artifacts["extension_results_nw_lag_sensitivity"]
    fit_sensitivity = artifacts["extension_fit_comparisons_nw_lag_sensitivity"]
    table_iv_artifact = artifacts["table_iv_results_no_ownership"]
    _, table_vi_artifact, table_vi_warnings = _collect_table_vi_no_ownership_surface(
        context,
        artifacts["table_vi_results_no_ownership"],
    )
    extension_results = artifacts["extension_results"]
    extension_fit_comparisons = artifacts["extension_fit_comparisons"]

    core_lf = scan_parquet_artifact(core_sensitivity)
    rows = [
        _nw_baseline_comparison_row(
            asset="ch5_lm2011_full_10k_return_coefficients",
            baseline_input="lm2011_table_iv_results_no_ownership",
            canonical_lf=scan_parquet_artifact(table_iv_artifact),
            sensitivity_lf=core_lf.filter(pl.col("stage_name") == pl.lit("table_iv_results_no_ownership")),
            join_keys=(
                "table_id",
                "specification_id",
                "text_scope",
                "signal_name",
                "dependent_variable",
                "coefficient_name",
                "weighting_rule",
            ),
            estimate_col="estimate",
            t_col="t_stat",
        ),
        _nw_baseline_comparison_row(
            asset="ch5_lm2011_table_vi_no_ownership_outcomes",
            baseline_input="lm2011_table_vi_results_no_ownership",
            canonical_lf=scan_parquet_artifact(table_vi_artifact),
            sensitivity_lf=core_lf.filter(pl.col("stage_name") == pl.lit("table_vi_results_no_ownership")),
            join_keys=(
                "table_id",
                "specification_id",
                "text_scope",
                "signal_name",
                "dependent_variable",
                "coefficient_name",
                "weighting_rule",
            ),
            estimate_col="estimate",
            t_col="t_stat",
        ),
        _nw_baseline_comparison_row(
            asset="ch5_matched_dictionary_finbert_coefficients_full",
            baseline_input="lm2011_extension_results",
            canonical_lf=_ensure_dictionary_family_source(scan_parquet_artifact(extension_results)).filter(
                (pl.col("estimator_status") == pl.lit("estimated"))
                & (pl.col("dictionary_family_source") == pl.lit("replication"))
            ),
            sensitivity_lf=_ensure_dictionary_family_source(scan_parquet_artifact(extension_sensitivity)).filter(
                (pl.col("nw_lags") == 1)
                & (pl.col("estimator_status") == pl.lit("estimated"))
                & (pl.col("dictionary_family_source") == pl.lit("replication"))
            ),
            join_keys=(
                "text_scope",
                "outcome_name",
                "feature_family",
                "control_set_id",
                "specification_name",
                "coefficient_name",
                "signal_name",
                "weighting_rule",
            ),
            estimate_col="estimate",
            t_col="t_stat",
            sensitivity_already_nw1=True,
        ),
        _nw_baseline_comparison_row(
            asset="ch5_extension_fit_comparisons",
            baseline_input="lm2011_extension_fit_comparisons",
            canonical_lf=_ensure_dictionary_family_source(scan_parquet_artifact(extension_fit_comparisons)).filter(
                (pl.col("estimator_status") == pl.lit("estimated"))
                & (pl.col("dictionary_family_source") == pl.lit("replication"))
            ),
            sensitivity_lf=_ensure_dictionary_family_source(scan_parquet_artifact(fit_sensitivity)).filter(
                (pl.col("nw_lags") == 1)
                & (pl.col("estimator_status") == pl.lit("estimated"))
                & (pl.col("dictionary_family_source") == pl.lit("replication"))
            ),
            join_keys=(
                "text_scope",
                "outcome_name",
                "control_set_id",
                "comparison_name",
                "left_specification_name",
                "right_specification_name",
                "weighting_rule",
                "common_success_policy",
            ),
            estimate_col="weighted_avg_delta_adj_r2",
            t_col="nw_t_stat_delta_adj_r2",
            sensitivity_already_nw1=True,
        ),
    ]
    selected = pl.DataFrame(rows)
    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={
            "core_tables_nw_lag_sensitivity": str(core_sensitivity.path),
            "extension_results_nw_lag_sensitivity": str(extension_sensitivity.path),
            "extension_fit_comparisons_nw_lag_sensitivity": str(fit_sensitivity.path),
            "table_iv_results_no_ownership": str(table_iv_artifact.path),
            "table_vi_results_no_ownership": str(table_vi_artifact.path),
            "extension_results": str(extension_results.path),
            "extension_fit_comparisons": str(extension_fit_comparisons.path),
        },
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
        warnings=table_vi_warnings,
    )


def _build_chapter5_nw_lag_core_no_ownership_appendix(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    source_artifact = artifacts["core_tables_nw_lag_sensitivity"]
    no_ownership_stages = (
        "table_iv_results_no_ownership",
        "table_v_results_no_ownership",
        "table_vi_results_no_ownership",
        "table_viii_results_no_ownership",
        "table_ia_i_results_no_ownership",
    )
    base_lf = (
        scan_parquet_artifact(source_artifact)
        .filter(
            pl.col("stage_name").is_in(no_ownership_stages)
            & (pl.col("coefficient_name") == pl.col("signal_name"))
            & pl.col("nw_lags").is_in(NW_LAG_GRID)
        )
        .with_columns(
            _nw_core_table_label_expr().alias("table_or_surface"),
            _nw_scope_label_expr("text_scope").alias("scope"),
            _nw_outcome_label_expr("dependent_variable").alias("outcome"),
            pl.col("specification_id").cast(pl.Utf8, strict=False).alias("specification"),
            _nw_coefficient_label_expr().alias("coefficient"),
            _nw_scale_label_expr("dependent_variable").alias("reported_scale"),
            _nw_core_stage_order_expr().alias("_surface_order"),
            _nw_scope_order_expr("text_scope").alias("_scope_order"),
            _nw_outcome_order_expr("dependent_variable").alias("_outcome_order"),
            pl.lit(0).alias("_specification_order"),
            _nw_coefficient_order_expr().alias("_coefficient_order"),
        )
        .with_columns(
            (pl.col("estimate").cast(pl.Float64, strict=False) * _nw_scale_expr("dependent_variable")).alias(
                "estimate"
            ),
            (pl.col("standard_error").cast(pl.Float64, strict=False) * _nw_scale_expr("dependent_variable")).alias(
                "standard_error"
            ),
            pl.col("t_stat").cast(pl.Float64, strict=False).alias("t_stat"),
            pl.col("n_quarters").cast(pl.Int64, strict=False).alias("n_quarters"),
            pl.col("mean_quarter_n").cast(pl.Float64, strict=False).alias("mean_quarter_n"),
        )
    )
    selected = _collect_nw_lag_grid(base_lf)
    if selected.is_empty():
        raise AssetBuildError("Core no-ownership NW lag appendix selection returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"core_tables_nw_lag_sensitivity": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_nw_lag_extension_coefficients_appendix(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    source_artifact = artifacts["extension_results_nw_lag_sensitivity"]
    base_lf = (
        _ensure_dictionary_family_source(scan_parquet_artifact(source_artifact))
        .filter(
            (pl.col("estimator_status") == pl.lit("estimated"))
            & (pl.col("dictionary_family_source") == pl.lit("replication"))
            & pl.col("text_scope").is_in(TARGET_TEXT_SCOPES)
            & (pl.col("outcome_name") == pl.lit("filing_period_excess_return"))
            & pl.col("coefficient_name").is_in(("lm_negative_tfidf", "finbert_neg_prob_lenw_mean"))
            & pl.col("nw_lags").is_in(NW_LAG_GRID)
        )
        .with_columns(
            pl.lit("Extension matched coefficients (replication dictionary)").alias("table_or_surface"),
            _scope_label_expr().alias("scope"),
            _nw_outcome_label_expr("outcome_name").alias("outcome"),
            pl.concat_str([pl.col("control_set_id"), pl.lit(" "), _specification_label_expr()]).alias(
                "specification"
            ),
            _nw_coefficient_label_expr().alias("coefficient"),
            _nw_scale_label_expr("outcome_name").alias("reported_scale"),
            pl.lit(1).alias("_surface_order"),
            _scope_order_expr().alias("_scope_order"),
            _nw_control_set_order_expr().alias("_outcome_order"),
            _specification_order_expr().alias("_specification_order"),
            _nw_coefficient_order_expr().alias("_coefficient_order"),
        )
        .with_columns(
            (pl.col("estimate").cast(pl.Float64, strict=False) * _nw_scale_expr("outcome_name")).alias("estimate"),
            (pl.col("standard_error").cast(pl.Float64, strict=False) * _nw_scale_expr("outcome_name")).alias(
                "standard_error"
            ),
            pl.col("t_stat").cast(pl.Float64, strict=False).alias("t_stat"),
            pl.col("n_quarters").cast(pl.Int64, strict=False).alias("n_quarters"),
            pl.col("mean_quarter_n").cast(pl.Float64, strict=False).alias("mean_quarter_n"),
        )
    )
    selected = _collect_nw_lag_grid(base_lf)
    if selected.is_empty():
        raise AssetBuildError("Extension coefficient NW lag appendix selection returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_results_nw_lag_sensitivity": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_nw_lag_extension_fit_comparisons_appendix(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    source_artifact = artifacts["extension_fit_comparisons_nw_lag_sensitivity"]
    base_lf = (
        _ensure_dictionary_family_source(scan_parquet_artifact(source_artifact))
        .filter(
            (pl.col("estimator_status") == pl.lit("estimated"))
            & pl.col("text_scope").is_in(TARGET_TEXT_SCOPES)
            & (pl.col("outcome_name") == pl.lit("filing_period_excess_return"))
            & pl.col("control_set_id").is_in(("C0", "C1", "C2"))
            & pl.col("dictionary_family_source").is_in(("replication", "extended"))
            & pl.col("nw_lags").is_in(NW_LAG_GRID)
        )
        .with_columns(
            pl.concat_str([_dictionary_family_label_expr(), pl.lit(" fit comparison")]).alias("table_or_surface"),
            _scope_label_expr().alias("scope"),
            _nw_outcome_label_expr("outcome_name").alias("outcome"),
            pl.concat_str([pl.col("control_set_id"), pl.lit(" "), _comparison_label_expr()]).alias("specification"),
            pl.concat_str([pl.lit("Delta adj R2: "), _comparison_label_expr()]).alias("coefficient"),
            pl.lit("raw").alias("reported_scale"),
            _dictionary_family_order_expr().alias("_surface_order"),
            _scope_order_expr().alias("_scope_order"),
            _nw_control_set_order_expr().alias("_outcome_order"),
            _comparison_order_expr().alias("_specification_order"),
            _comparison_order_expr().alias("_coefficient_order"),
        )
        .with_columns(
            pl.col("weighted_avg_delta_adj_r2").cast(pl.Float64, strict=False).alias("estimate"),
            pl.col("nw_se_delta_adj_r2").cast(pl.Float64, strict=False).alias("standard_error"),
            pl.col("nw_t_stat_delta_adj_r2").cast(pl.Float64, strict=False).alias("t_stat"),
            pl.col("n_quarters").cast(pl.Int64, strict=False).alias("n_quarters"),
            pl.col("mean_quarter_n").cast(pl.Float64, strict=False).alias("mean_quarter_n"),
        )
    )
    selected = _collect_nw_lag_grid(base_lf)
    if selected.is_empty():
        raise AssetBuildError("Extension fit-comparison NW lag appendix selection returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_fit_comparisons_nw_lag_sensitivity": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_concordance(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "common_row_comparison")
    dictionary_artifact = artifacts["extension_dictionary_surface"]
    finbert_artifact = artifacts["extension_finbert_surface"]

    joined = (
        common_row_comparison(
            scan_parquet_artifact(dictionary_artifact),
            scan_parquet_artifact(finbert_artifact),
            join_keys=DEFAULT_COMPARISON_JOIN_KEYS,
            left_signal_columns=("lm_negative_tfidf",),
            right_signal_columns=("finbert_neg_prob_lenw_mean",),
            left_filters=(
                pl.col("text_scope") == pl.lit("item_7_mda"),
                pl.col("dictionary_family") == pl.lit("replication"),
            ),
            right_filters=(pl.col("text_scope") == pl.lit("item_7_mda"),),
        )
        .select(
            pl.col("doc_id"),
            pl.col("filing_date"),
            pl.col("text_scope"),
            pl.col("cleaning_policy_id"),
            pl.col("lm_negative_tfidf"),
            pl.col("finbert_neg_prob_lenw_mean"),
        )
        .sort("filing_date", "doc_id")
        .collect()
    )
    if joined.is_empty():
        raise AssetBuildError("Chapter 5 concordance selection returned zero matched common-sample rows.")

    csv_path = context.output_dirs["csv"] / f"{spec.output_stem}.csv"
    write_csv_table(joined, csv_path)
    figure = build_concordance_figure(
        joined,
        x_col="lm_negative_tfidf",
        y_col="finbert_neg_prob_lenw_mean",
        x_label="LM2011 negative tf-idf",
        y_label="FinBERT negative probability (length-weighted mean)",
    )
    figure_paths = write_figure_bundle(figure, context.output_dirs["figures"] / spec.output_stem)

    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={
            "extension_dictionary_surface": str(dictionary_artifact.path),
            "extension_finbert_surface": str(finbert_artifact.path),
        },
        output_paths={
            "csv": str(csv_path),
            "png": str(figure_paths["png"]),
            "pdf": str(figure_paths["pdf"]),
        },
        row_counts={"common_sample_rows": joined.height},
    )


def _build_chapter5_lm_doc_score_ecdf(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "raw_available")
    source_artifact = artifacts["extension_dictionary_surface"]
    long_df = _document_score_long_df(
        scan_parquet_artifact(source_artifact),
        metric_specs=(
            ("lm_negative_prop", "LM2011 negative proportion", "lm_negative_prop"),
            ("lm_negative_tfidf", "LM2011 negative tf-idf", "lm_negative_tfidf"),
        ),
        filters=(
            pl.col("text_scope").is_in(TARGET_TEXT_SCOPES),
            pl.col("dictionary_family").cast(pl.Utf8, strict=False) == pl.lit("replication"),
        ),
    )
    ecdf_df = _exact_ecdf_frame(long_df, score_col="score")
    if ecdf_df.is_empty():
        raise AssetBuildError("LM2011 document score ECDF selection returned zero rows.")

    figure = build_metric_panel_ecdf_figure(
        ecdf_df,
        metric_panels=(
            ("lm_negative_prop", "Proportion", "LM2011 negative proportion"),
            ("lm_negative_tfidf", "tf-idf", "LM2011 negative tf-idf"),
        ),
        x_col="score",
    )
    output_paths = _write_figure_outputs(context, spec, ecdf_df, figure)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_dictionary_surface": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"ecdf_rows": ecdf_df.height, "score_rows": long_df.height},
    )


def _build_chapter5_finbert_doc_score_ecdf(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "raw_available")
    source_artifact = artifacts["extension_finbert_surface"]
    long_df = _document_score_long_df(
        scan_parquet_artifact(source_artifact),
        metric_specs=(
            ("finbert_neg_prob_lenw_mean", "FinBERT negative probability", "finbert_neg_prob_lenw_mean"),
            ("finbert_net_negative_lenw_mean", "FinBERT net negative", "finbert_net_negative_lenw_mean"),
        ),
        filters=(pl.col("text_scope").is_in(TARGET_TEXT_SCOPES),),
    )
    ecdf_df = _exact_ecdf_frame(long_df, score_col="score")
    if ecdf_df.is_empty():
        raise AssetBuildError("FinBERT document score ECDF selection returned zero rows.")

    figure = build_ecdf_lines_figure(ecdf_df, x_col="score", x_label="Document score")
    output_paths = _write_figure_outputs(context, spec, ecdf_df, figure)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_finbert_surface": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"ecdf_rows": ecdf_df.height, "score_rows": long_df.height},
    )


def _build_chapter5_finbert_sentence_ecdf(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "analysis_panel_sentence_scores")
    summary = _cached_finbert_sentence_summary(context, artifacts)
    if summary.ecdf.is_empty():
        raise AssetBuildError("FinBERT sentence ECDF summary returned zero rows.")

    figure = build_ecdf_lines_figure(
        summary.ecdf,
        x_col="score_bin_right",
        x_label="Sentence-level FinBERT negative probability",
    )
    output_paths = _write_figure_outputs(context, spec, summary.ecdf, figure)
    return _sentence_figure_result(
        context,
        spec,
        artifacts,
        output_paths,
        row_counts={"ecdf_rows": summary.ecdf.height, "sentence_rows": summary.sentence_count},
    )


def _build_chapter5_lm_sentence_ecdf(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "analysis_panel_sentence_scores")
    summary = _cached_lm_sentence_summary(context, artifacts)
    if summary.ecdf.is_empty():
        raise AssetBuildError("LM2011 sentence ECDF summary returned zero rows.")

    figure = build_ecdf_lines_figure(
        summary.ecdf,
        x_col="score_bin_right",
        x_label="Sentence-level LM2011 negative word share",
    )
    output_paths = _write_figure_outputs(context, spec, summary.ecdf, figure)
    return _sentence_figure_result(
        context,
        spec,
        artifacts,
        output_paths,
        row_counts={"ecdf_rows": summary.ecdf.height, "sentence_rows": summary.sentence_count},
    )


def _build_chapter5_high_negative_sentence_share(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "analysis_panel_sentence_scores")
    finbert_summary = _cached_finbert_sentence_summary(context, artifacts)
    lm_summary = _cached_lm_sentence_summary(context, artifacts)
    high_share = pl.concat(
        [finbert_summary.high_share, lm_summary.high_share],
        how="vertical_relaxed",
    )
    ecdf_df = _exact_ecdf_frame(high_share, score_col="high_sentence_share")
    if ecdf_df.is_empty():
        raise AssetBuildError("High-negative sentence share selection returned zero rows.")

    figure = build_ecdf_lines_figure(
        ecdf_df,
        x_col="score",
        x_label="Per-filing share of high-negative sentences",
    )
    output_paths = _write_figure_outputs(context, spec, ecdf_df, figure)
    return _sentence_figure_result(
        context,
        spec,
        artifacts,
        output_paths,
        row_counts={
            "ecdf_rows": ecdf_df.height,
            "finbert_sentence_rows": finbert_summary.sentence_count,
            "lm_sentence_rows": lm_summary.sentence_count,
        },
    )


def _build_chapter5_concordance_by_scope(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "common_row_comparison")
    dictionary_artifact = artifacts["extension_dictionary_surface"]
    finbert_artifact = artifacts["extension_finbert_surface"]

    joined = (
        common_row_comparison(
            scan_parquet_artifact(dictionary_artifact),
            scan_parquet_artifact(finbert_artifact),
            join_keys=DEFAULT_COMPARISON_JOIN_KEYS,
            left_signal_columns=("lm_negative_tfidf",),
            right_signal_columns=("finbert_neg_prob_lenw_mean",),
            left_filters=(
                pl.col("text_scope").is_in(TARGET_TEXT_SCOPES),
                pl.col("dictionary_family") == pl.lit("replication"),
            ),
            right_filters=(pl.col("text_scope").is_in(TARGET_TEXT_SCOPES),),
        )
        .select(
            pl.col("doc_id"),
            pl.col("filing_date"),
            pl.col("text_scope"),
            pl.col("cleaning_policy_id"),
            pl.col("lm_negative_tfidf"),
            pl.col("finbert_neg_prob_lenw_mean"),
        )
        .sort("text_scope", "filing_date", "doc_id")
        .collect()
    )
    if joined.is_empty():
        raise AssetBuildError("Chapter 5 concordance by scope returned zero matched rows.")

    figure = build_concordance_by_scope_figure(
        joined,
        x_col="lm_negative_tfidf",
        y_col="finbert_neg_prob_lenw_mean",
        x_label="LM2011 negative tf-idf",
        y_label="FinBERT negative probability (length-weighted mean)",
    )
    output_paths = _write_figure_outputs(context, spec, joined, figure)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={
            "extension_dictionary_surface": str(dictionary_artifact.path),
            "extension_finbert_surface": str(finbert_artifact.path),
        },
        output_paths=output_paths,
        row_counts={"common_sample_rows": joined.height},
    )


def _build_chapter5_score_drift_by_year(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    panel_artifact = artifacts["extension_analysis_panel"]
    base = (
        scan_parquet_artifact(panel_artifact)
        .filter(pl.col("text_scope").is_in(TARGET_TEXT_SCOPES))
        .select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("filing_date").cast(pl.Date, strict=False).alias("filing_date"),
            pl.col("text_scope").cast(pl.Utf8, strict=False),
            pl.col("lm_negative_tfidf").cast(pl.Float64, strict=False),
            pl.col("finbert_neg_prob_lenw_mean").cast(pl.Float64, strict=False),
        )
        .with_columns(pl.col("filing_date").dt.year().alias("filing_year"))
        .group_by("filing_year", "text_scope")
        .agg(
            pl.len().alias("panel_rows"),
            pl.col("doc_id").n_unique().alias("unique_docs"),
            pl.col("lm_negative_tfidf").mean().alias("lm_negative_tfidf_mean"),
            pl.col("lm_negative_tfidf").median().alias("lm_negative_tfidf_median"),
            pl.col("finbert_neg_prob_lenw_mean").mean().alias("finbert_negative_mean"),
            pl.col("finbert_neg_prob_lenw_mean").median().alias("finbert_negative_median"),
        )
        .with_columns(_scope_label_expr().alias("scope"))
    )
    long_lf = pl.concat(
        [
            base.select(
                "filing_year",
                "text_scope",
                "scope",
                "panel_rows",
                "unique_docs",
                pl.lit("lm_negative_tfidf").alias("score_metric"),
                pl.lit("LM negative tf-idf").alias("metric_label"),
                pl.col("lm_negative_tfidf_mean").alias("mean_score"),
                pl.col("lm_negative_tfidf_median").alias("median_score"),
                pl.lit(1).alias("_metric_order"),
            ),
            base.select(
                "filing_year",
                "text_scope",
                "scope",
                "panel_rows",
                "unique_docs",
                pl.lit("finbert_neg_prob_lenw_mean").alias("score_metric"),
                pl.lit("FinBERT negative probability").alias("metric_label"),
                pl.col("finbert_negative_mean").alias("mean_score"),
                pl.col("finbert_negative_median").alias("median_score"),
                pl.lit(2).alias("_metric_order"),
            ),
        ],
        how="vertical_relaxed",
    ).with_columns(
        pl.concat_str([pl.col("scope"), pl.lit(" | "), pl.col("metric_label")]).alias("series_label")
    )
    selected = (
        long_lf.sort("filing_year", "text_scope", "_metric_order")
        .select(
            "filing_year",
            "scope",
            "score_metric",
            "metric_label",
            "panel_rows",
            "unique_docs",
            "mean_score",
            "median_score",
            "series_label",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Score drift-by-year figure returned zero rows.")

    figure = build_multi_series_line_figure(
        selected,
        x_col="filing_year",
        y_col="mean_score",
        series_col="series_label",
        x_label="Filing year",
        y_label="Mean score",
    )
    output_paths = _write_figure_outputs(context, spec, selected, figure)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_analysis_panel": str(panel_artifact.path)},
        output_paths=output_paths,
        row_counts={"figure_rows": selected.height},
    )


def _build_chapter5_finbert_robustness_coefficients(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "finbert_robustness_regression_results")
    frames = (
        _robustness_coefficients_lf(
            scan_parquet_artifact(artifacts["existing_scale_coefficients"]),
            family_label="Existing scale",
            signal_coefficient_expr=pl.col("coefficient_name") == pl.lit("finbert_neg_prob_lenw_mean"),
        ),
        _robustness_coefficients_lf(
            scan_parquet_artifact(artifacts["tail_coefficients"]),
            family_label="Tail signal",
            signal_coefficient_expr=pl.col("coefficient_name") == pl.lit("finbert_neg_prob_lenw_mean"),
        ),
        _robustness_coefficients_lf(
            scan_parquet_artifact(artifacts["quantile_coefficients"]),
            family_label="Quantile signal",
            signal_coefficient_expr=pl.col("coefficient_name") == pl.col("variant_id"),
        ),
    )
    selected = (
        pl.concat(frames)
        .sort("_scope_order", "_family_order", "_signal_order", "_spec_order", "_weight_order")
        .select(
            "text_scope",
            "signal_family",
            "signal",
            "specification",
            "quarter_weighting",
            "estimate",
            "std_error",
            "t_stat",
            "p_value",
            "stars",
            "n_obs",
            "n_quarters",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Chapter 5 FinBERT robustness coefficient selection returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={
            "existing_scale_coefficients": str(artifacts["existing_scale_coefficients"].path),
            "tail_coefficients": str(artifacts["tail_coefficients"].path),
            "quantile_coefficients": str(artifacts["quantile_coefficients"].path),
        },
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_finbert_robustness_fit_comparisons(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    _require_contract(spec, "finbert_robustness_fit_comparisons")
    frames = (
        _robustness_fit_comparisons_lf(
            scan_parquet_artifact(artifacts["existing_scale_fit_comparisons"]),
            family_label="Existing scale",
        ),
        _robustness_fit_comparisons_lf(
            scan_parquet_artifact(artifacts["tail_fit_comparisons"]),
            family_label="Tail signal",
        ),
        _robustness_fit_comparisons_lf(
            scan_parquet_artifact(artifacts["quantile_fit_comparisons"]),
            family_label="Quantile signal",
        ),
    )
    selected = (
        pl.concat(frames)
        .sort("_scope_order", "_family_order", "_signal_order", "_comparison_order")
        .select(
            "text_scope",
            "signal_family",
            "signal",
            "comparison",
            "delta_adj_r2_weighted",
            "delta_adj_r2_equal_quarter",
            "nw_t_stat",
            "nw_p_value",
            "stars",
            "total_n_obs",
            "n_quarters",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Chapter 5 FinBERT robustness fit-comparison selection returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={
            "existing_scale_fit_comparisons": str(artifacts["existing_scale_fit_comparisons"].path),
            "tail_fit_comparisons": str(artifacts["tail_fit_comparisons"].path),
            "quantile_fit_comparisons": str(artifacts["quantile_fit_comparisons"].path),
        },
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_matched_dictionary_finbert_coefficients_full(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    source_artifact = artifacts["extension_results"]
    selected = (
        scan_parquet_artifact(source_artifact)
        .filter(
            (pl.col("estimator_status") == pl.lit("estimated"))
            & pl.col("text_scope").is_in(TARGET_TEXT_SCOPES)
            & (
                (pl.col("coefficient_name") == pl.col("signal_name"))
                | pl.col("coefficient_name").is_in(("lm_negative_tfidf", "finbert_neg_prob_lenw_mean"))
            )
            & pl.col("coefficient_name").is_in(("lm_negative_tfidf", "finbert_neg_prob_lenw_mean"))
        )
        .with_columns(
            _scope_label_expr().alias("scope"),
            _specification_label_expr().alias("specification"),
            _signal_label_expr().alias("coefficient"),
            _stars_expr(pl.col("p_value")).alias("stars"),
            _scope_order_expr().alias("_scope_order"),
            _specification_order_expr().alias("_spec_order"),
        )
        .sort("_scope_order", "_spec_order", "coefficient_name")
        .select(
            "scope",
            "specification",
            "coefficient",
            "estimate",
            pl.col("standard_error").alias("std_error"),
            "t_stat",
            "p_value",
            "stars",
            "n_obs",
            "n_quarters",
            "mean_quarter_n",
            "weighting_rule",
        )
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Matched dictionary-versus-FinBERT coefficient table returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"extension_results": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_fama_macbeth_skipped_quarter_diagnostics(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    frames: list[pl.LazyFrame] = []
    for logical_name, artifact in artifacts.items():
        frames.append(_skipped_quarter_summary_lf(scan_parquet_artifact(artifact), logical_name))
    if not frames:
        raise AssetBuildError("No skipped-quarter artifacts were resolved.")
    selected = (
        pl.concat(frames, how="vertical_relaxed")
        .sort("source", "text_scope", "skip_reason")
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Skipped-quarter diagnostics returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={key: str(value.path) for key, value in artifacts.items()},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_alternative_signal_robustness_full_grid(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    frames = [
        _robustness_full_grid_lf(scan_parquet_artifact(artifacts["existing_scale_coefficients"]), "existing_scale"),
        _robustness_full_grid_lf(scan_parquet_artifact(artifacts["tail_coefficients"]), "tail_signal"),
        _robustness_full_grid_lf(scan_parquet_artifact(artifacts["quantile_coefficients"]), "quantile_signal"),
    ]
    selected = (
        pl.concat(frames, how="vertical_relaxed")
        .sort("signal_family", "text_scope", "variant_id", "specification", "coefficient_name")
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Alternative signal robustness grid returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={key: str(value.path) for key, value in artifacts.items()},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_full_controls_coefficient_appendix(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    control_columns = ("log_size", "log_book_to_market", "log_share_turnover", "pre_ffalpha", "nasdaq_dummy")
    frames = [
        _control_coefficients_lf(
            scan_parquet_artifact(artifacts["table_iv_results_no_ownership"]),
            "lm2011_table_iv_full_10k",
            control_columns,
        ),
        _control_coefficients_lf(
            scan_parquet_artifact(artifacts["extension_results"]),
            "extension_matched",
            control_columns,
        ),
        _control_coefficients_lf(
            scan_parquet_artifact(artifacts["existing_scale_coefficients"]),
            "finbert_robustness_existing_scale",
            control_columns,
        ),
    ]
    selected = (
        pl.concat(frames, how="vertical_relaxed")
        .sort("source", "text_scope", "signal_name", "coefficient_name")
        .collect()
    )
    if selected.is_empty():
        raise AssetBuildError("Full controls coefficient appendix returned zero rows.")

    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={key: str(value.path) for key, value in artifacts.items()},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_text_score_control_correlation_matrix(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    source_artifact = artifacts["return_regression_panel_full_10k"]
    columns = (
        "lm_negative_prop",
        "lm_negative_tfidf",
        "h4n_inf_prop",
        "h4n_inf_tfidf",
        "log_size",
        "log_book_to_market",
        "log_share_turnover",
        "pre_ffalpha",
        "nasdaq_dummy",
    )
    selected = _correlation_matrix_table(scan_parquet_artifact(source_artifact), columns=columns)
    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={"return_regression_panel_full_10k": str(source_artifact.path)},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _build_chapter5_research_question_evidence_map(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
) -> BuildResult:
    rows = [
        {
            "research_question": "RQ1: Can the LM2011 benchmark be carried into the post-2009 sample?",
            "evidence_asset": "ch5_lm2011_full_10k_return_coefficients; ch5_lm2011_table_vi_no_ownership_outcomes",
            "evidence_status": "Generated from stored no-ownership LM2011 outputs",
            "claim_guardrail": "Interpret as a benchmark extension, not as a full recreation of unavailable ownership-conditioned specifications.",
            "recommended_thesis_placement": "Chapter 5 main results / appendix",
        },
        {
            "research_question": "RQ2: Do Item 1A and Item 7 provide distinct text-signal evidence?",
            "evidence_asset": "ch4_score_family_descriptive_statistics; ch5_score_drift_by_year; ch5_concordance_negative_scores_by_scope",
            "evidence_status": "Generated for Item 1A and Item 7 matched surfaces",
            "claim_guardrail": "Keep claims descriptive unless tied to the matched regression tables.",
            "recommended_thesis_placement": "Chapter 4 diagnostics and Chapter 5 appendix",
        },
        {
            "research_question": "RQ3: Does FinBERT improve model fit relative to dictionary scores?",
            "evidence_asset": "ch5_extension_c0_fit_summary; ch5_extension_c0_fit_comparisons; ch5_extension_fit_delta_path",
            "evidence_status": "Generated from common-success C0 fit artifacts",
            "claim_guardrail": "Report Q5-Q1 and fit deltas on stored sign/scaling; do not infer C1/C2 centrality.",
            "recommended_thesis_placement": "Chapter 5 main results / appendix",
        },
        {
            "research_question": "RQ4: Are high-minus-low negativity portfolio returns interpretable from current artifacts?",
            "evidence_asset": "ch5_lm2011_portfolio_long_short; ch5_lm2011_portfolio_formation_diagnostics; ch5_portfolio_cumulative_q5_minus_q1",
            "evidence_status": "Generated from stored Q5-Q1 monthly and summary portfolio artifacts",
            "claim_guardrail": PORTFOLIO_SPREAD_DEFINITION + "; separate Q1/Q5 legs are not stored.",
            "recommended_thesis_placement": "Chapter 5 portfolio subsection",
        },
        {
            "research_question": "RQ5: What data limitations affect the extension claims?",
            "evidence_asset": "ch4_item_cleaning_quality_by_year; ch4_ownership_coverage_by_year; ch4_finbert_segment_token_diagnostics",
            "evidence_status": "Generated from cleaning, coverage, and FinBERT feature diagnostics",
            "claim_guardrail": "Use as transparency evidence; defer new regressions requiring unavailable reconstructed panels.",
            "recommended_thesis_placement": "Chapter 4/5 limitations appendix",
        },
    ]
    selected = pl.DataFrame(rows)
    output_paths = _write_table_outputs(context, spec, selected)
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={},
        output_paths=output_paths,
        row_counts={"table_rows": selected.height},
    )


def _table_vi_signal_coefficients_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.filter(
            pl.col("dependent_variable").is_in(TABLE_VI_DEPENDENT_VARIABLES)
            & pl.col("signal_name").is_in(TABLE_VI_SIGNAL_COLUMNS)
            & (pl.col("coefficient_name") == pl.col("signal_name"))
        )
        .with_columns(
            _table_vi_outcome_order_expr().alias("_outcome_order"),
            _table_vi_signal_order_expr().alias("_signal_order"),
            _table_vi_weight_order_expr().alias("_weight_order"),
            _table_vi_outcome_scale_expr().alias("_scale"),
            _table_vi_outcome_label_expr().alias("outcome"),
            _table_vi_signal_label_expr().alias("signal"),
            _table_vi_weighting_label_expr().alias("weighting"),
            _table_vi_reported_scale_expr().alias("reported_scale"),
        )
        .with_columns(
            (pl.col("estimate").cast(pl.Float64, strict=False) * pl.col("_scale")).alias("estimate"),
            (pl.col("standard_error").cast(pl.Float64, strict=False) * pl.col("_scale")).alias("std_error"),
            pl.col("t_stat").cast(pl.Float64, strict=False).alias("t_stat"),
            pl.col("mean_quarter_n").cast(pl.Float64, strict=False).alias("mean_quarter_n"),
        )
        .select(
            "dependent_variable",
            "signal_name",
            "outcome",
            "signal",
            "weighting",
            "estimate",
            "std_error",
            "t_stat",
            "n_quarters",
            "mean_quarter_n",
            "nw_lags",
            "reported_scale",
            "_outcome_order",
            "_weight_order",
            "_signal_order",
        )
    )


def _collect_table_vi_no_ownership_surface(
    context: BuildContext,
    source_artifact: ResolvedArtifact,
) -> tuple[pl.DataFrame, ResolvedArtifact, tuple[str, ...]]:
    selected = _collect_table_vi_signal_coefficients(source_artifact)
    try:
        _validate_table_vi_surface(selected)
        return selected, source_artifact, ()
    except AssetBuildError as original_error:
        fallback_path = _resolve_table_vi_validation_fallback_path(context, source_artifact)
        if fallback_path is None:
            raise original_error

    fallback_artifact = ResolvedArtifact(
        requirement=source_artifact.requirement,
        path=fallback_path,
        run=source_artifact.run,
    )
    selected = _collect_table_vi_signal_coefficients(fallback_artifact)
    _validate_table_vi_surface(selected)
    warning = (
        "Resolved canonical Table VI no-ownership artifact did not contain the full three-outcome "
        f"surface; used validation artifact {fallback_path}."
    )
    return selected, fallback_artifact, (warning,)


def _collect_table_vi_signal_coefficients(source_artifact: ResolvedArtifact) -> pl.DataFrame:
    return (
        _table_vi_signal_coefficients_lf(scan_parquet_artifact(source_artifact))
        .sort("_outcome_order", "_weight_order", "_signal_order")
        .collect()
    )


def _resolve_table_vi_validation_fallback_path(
    context: BuildContext,
    source_artifact: ResolvedArtifact,
) -> Path | None:
    filenames = (
        "lm2011_table_vi_results_no_ownership_validation.parquet",
        "lm2011_table_vi_results_no_ownership.parquet",
    )
    validation_dirs = ("lm2011_table_vi_validation_second_pass", "lm2011_table_vi_validation")
    candidate_roots: list[Path] = [context.repo_root / "full_data_run"]
    candidate_roots.extend(source_artifact.run.root.parents)
    candidate_roots.append(source_artifact.run.root)

    seen: set[Path] = set()
    for root in candidate_roots:
        for dirname in validation_dirs:
            for filename in filenames:
                candidate = (root / dirname / filename).resolve()
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate == source_artifact.path.resolve():
                    continue
                if candidate.exists() and candidate.is_file():
                    return candidate
    return None


def _validate_table_vi_surface(df: pl.DataFrame) -> None:
    if df.is_empty():
        raise AssetBuildError("LM2011 Table VI no-ownership selection returned zero signal rows.")

    observed_pairs = {
        (row["dependent_variable"], row["signal_name"])
        for row in df.select("dependent_variable", "signal_name").to_dicts()
    }
    expected_pairs = {
        (dependent_variable, signal_name)
        for dependent_variable in TABLE_VI_DEPENDENT_VARIABLES
        for signal_name in TABLE_VI_SIGNAL_COLUMNS
    }
    missing_pairs = sorted(expected_pairs.difference(observed_pairs))
    if missing_pairs:
        preview = ", ".join(f"{dependent_variable}:{signal_name}" for dependent_variable, signal_name in missing_pairs[:8])
        suffix = "" if len(missing_pairs) <= 8 else f", ... ({len(missing_pairs)} missing total)"
        raise AssetBuildError(f"LM2011 Table VI no-ownership artifact is missing expected specs: {preview}{suffix}")

    duplicate_specs = (
        df.group_by("dependent_variable", "signal_name")
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate_specs.height:
        preview = ", ".join(
            f"{row['dependent_variable']}:{row['signal_name']}"
            for row in duplicate_specs.select("dependent_variable", "signal_name").head(8).to_dicts()
        )
        raise AssetBuildError(f"LM2011 Table VI no-ownership artifact has duplicate signal specs: {preview}")

    if df.height != TABLE_VI_EXPECTED_SPEC_COUNT:
        raise AssetBuildError(
            f"LM2011 Table VI no-ownership selection returned {df.height} rows; "
            f"expected {TABLE_VI_EXPECTED_SPEC_COUNT}."
        )


def _nw_baseline_comparison_row(
    *,
    asset: str,
    baseline_input: str,
    canonical_lf: pl.LazyFrame,
    sensitivity_lf: pl.LazyFrame,
    join_keys: tuple[str, ...],
    estimate_col: str,
    t_col: str,
    sensitivity_already_nw1: bool = False,
) -> dict[str, object]:
    canonical_selected = canonical_lf.select(
        *[pl.col(column).cast(pl.Utf8, strict=False) for column in join_keys],
        pl.col(estimate_col).cast(pl.Float64, strict=False).alias("_canonical_estimate"),
        pl.col(t_col).cast(pl.Float64, strict=False).alias("_canonical_t"),
    )
    sensitivity_source = sensitivity_lf if sensitivity_already_nw1 else sensitivity_lf.filter(pl.col("nw_lags") == 1)
    sensitivity_selected = sensitivity_source.select(
        *[pl.col(column).cast(pl.Utf8, strict=False) for column in join_keys],
        pl.col(estimate_col).cast(pl.Float64, strict=False).alias("_sensitivity_estimate"),
        pl.col(t_col).cast(pl.Float64, strict=False).alias("_sensitivity_t"),
    )
    canonical_rows = _lazy_count(canonical_selected)
    sensitivity_rows = _lazy_count(sensitivity_selected)
    joined = canonical_selected.join(sensitivity_selected, on=list(join_keys), how="inner")
    summary = (
        joined.select(
            pl.len().alias("joined_rows"),
            (pl.col("_canonical_estimate") - pl.col("_sensitivity_estimate"))
            .abs()
            .max()
            .alias("max_abs_estimate_diff"),
            (pl.col("_canonical_t") - pl.col("_sensitivity_t")).abs().max().alias("max_abs_t_diff"),
        )
        .collect()
        .row(0, named=True)
    )
    return {
        "asset": asset,
        "baseline_input": baseline_input,
        "canonical_rows": canonical_rows,
        "sensitivity_nw1_rows": sensitivity_rows,
        "joined_rows": int(summary["joined_rows"]),
        "max_abs_estimate_diff": summary["max_abs_estimate_diff"],
        "max_abs_t_diff": summary["max_abs_t_diff"],
    }


def _lazy_count(lf: pl.LazyFrame) -> int:
    return int(lf.select(pl.len().alias("_n")).collect().item())


def _collect_nw_lag_grid(base_lf: pl.LazyFrame) -> pl.DataFrame:
    identity_columns = (
        "table_or_surface",
        "scope",
        "outcome",
        "specification",
        "coefficient",
        "reported_scale",
        "_surface_order",
        "_scope_order",
        "_outcome_order",
        "_specification_order",
        "_coefficient_order",
    )
    aggregations: list[pl.Expr] = [
        pl.col("estimate").first().alias("estimate"),
        pl.col("n_quarters").first().alias("n_quarters"),
        pl.col("mean_quarter_n").first().alias("mean_quarter_n"),
    ]
    for lag in NW_LAG_GRID:
        aggregations.extend(
            (
                pl.col("standard_error").filter(pl.col("nw_lags") == lag).first().alias(f"se_nw{lag}"),
                pl.col("t_stat").filter(pl.col("nw_lags") == lag).first().alias(f"t_nw{lag}"),
                pl.col("_stars").filter(pl.col("nw_lags") == lag).first().alias(f"stars_nw{lag}"),
            )
        )

    return (
        base_lf.with_columns(
            _stars_expr(_normal_p_value_expr(pl.col("t_stat"))).alias("_stars"),
        )
        .group_by(*identity_columns)
        .agg(*aggregations)
        .with_columns(
            (pl.col("se_nw4") / pl.col("se_nw1")).alias("se_ratio_nw4_to_nw1"),
            pl.all_horizontal([pl.col(f"t_nw{lag}").abs() >= NW_SIG_5_ABS_T for lag in NW_LAG_GRID])
            .fill_null(False)
            .alias("sig5_all_lags"),
            ((pl.col("t_nw1").abs() >= NW_SIG_5_ABS_T) & (pl.col("t_nw4").abs() < NW_SIG_5_ABS_T))
            .fill_null(False)
            .alias("lost_5pct_by_nw4"),
            ((pl.col("t_nw1").abs() < NW_SIG_10_ABS_T) & (pl.col("t_nw4").abs() >= NW_SIG_10_ABS_T))
            .fill_null(False)
            .alias("gained_10pct_by_nw4"),
        )
        .sort("_surface_order", "_scope_order", "_outcome_order", "_specification_order", "_coefficient_order")
        .select(
            "table_or_surface",
            "scope",
            "outcome",
            "specification",
            "coefficient",
            "estimate",
            "t_nw1",
            "t_nw2",
            "t_nw3",
            "t_nw4",
            "stars_nw1",
            "stars_nw2",
            "stars_nw3",
            "stars_nw4",
            "se_ratio_nw4_to_nw1",
            "sig5_all_lags",
            "lost_5pct_by_nw4",
            "gained_10pct_by_nw4",
            "n_quarters",
            "mean_quarter_n",
            "reported_scale",
        )
        .collect()
    )


def _nw_core_table_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("stage_name") == "table_iv_results_no_ownership")
        .then(pl.lit("Table IV full 10-K returns"))
        .when(pl.col("stage_name") == "table_v_results_no_ownership")
        .then(pl.lit("Table V MD&A returns"))
        .when(pl.col("stage_name") == "table_vi_results_no_ownership")
        .then(pl.lit("Table VI full 10-K outcomes"))
        .when(pl.col("stage_name") == "table_viii_results_no_ownership")
        .then(pl.lit("Table VIII SUE"))
        .when(pl.col("stage_name") == "table_ia_i_results_no_ownership")
        .then(pl.lit("Table IA.I normalized differences"))
        .otherwise(pl.col("stage_name"))
    )


def _nw_core_stage_order_expr() -> pl.Expr:
    return (
        pl.when(pl.col("stage_name") == "table_iv_results_no_ownership")
        .then(pl.lit(1))
        .when(pl.col("stage_name") == "table_v_results_no_ownership")
        .then(pl.lit(2))
        .when(pl.col("stage_name") == "table_vi_results_no_ownership")
        .then(pl.lit(3))
        .when(pl.col("stage_name") == "table_viii_results_no_ownership")
        .then(pl.lit(4))
        .when(pl.col("stage_name") == "table_ia_i_results_no_ownership")
        .then(pl.lit(5))
        .otherwise(pl.lit(99))
    )


def _nw_scope_label_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column) == "full_10k")
        .then(pl.lit("Full 10-K"))
        .when(pl.col(column) == "mda_item_7")
        .then(pl.lit("MD&A Item 7"))
        .when(pl.col(column) == "item_7_mda")
        .then(pl.lit("Item 7 MD&A"))
        .when(pl.col(column) == "item_1a_risk_factors")
        .then(pl.lit("Item 1A risk factors"))
        .otherwise(pl.col(column))
    )


def _nw_scope_order_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column) == "full_10k")
        .then(pl.lit(1))
        .when(pl.col(column).is_in(("mda_item_7", "item_7_mda")))
        .then(pl.lit(2))
        .when(pl.col(column) == "item_1a_risk_factors")
        .then(pl.lit(3))
        .otherwise(pl.lit(99))
    )


def _nw_outcome_label_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column) == "filing_period_excess_return")
        .then(pl.lit("Filing-period excess return"))
        .when(pl.col(column) == "abnormal_volume")
        .then(pl.lit("Abnormal volume"))
        .when(pl.col(column) == "postevent_return_volatility")
        .then(pl.lit("Postevent return volatility"))
        .when(pl.col(column) == "sue")
        .then(pl.lit("SUE"))
        .otherwise(pl.col(column))
    )


def _nw_outcome_order_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column) == "filing_period_excess_return")
        .then(pl.lit(1))
        .when(pl.col(column) == "abnormal_volume")
        .then(pl.lit(2))
        .when(pl.col(column) == "postevent_return_volatility")
        .then(pl.lit(3))
        .when(pl.col(column) == "sue")
        .then(pl.lit(4))
        .otherwise(pl.lit(99))
    )


def _nw_control_set_order_expr() -> pl.Expr:
    return (
        pl.when(pl.col("control_set_id") == "C0")
        .then(pl.lit(1))
        .when(pl.col("control_set_id") == "C1")
        .then(pl.lit(2))
        .when(pl.col("control_set_id") == "C2")
        .then(pl.lit(3))
        .otherwise(pl.lit(99))
    )


def _nw_coefficient_label_expr() -> pl.Expr:
    labels = {
        "h4n_inf_prop": "H4N-Inf proportion",
        "h4n_inf_tfidf": "H4N-Inf tf-idf",
        "lm_negative_prop": "LM negative proportion",
        "lm_negative_tfidf": "LM negative tf-idf",
        "lm_positive_prop": "LM positive proportion",
        "lm_positive_tfidf": "LM positive tf-idf",
        "lm_uncertainty_prop": "LM uncertainty proportion",
        "lm_uncertainty_tfidf": "LM uncertainty tf-idf",
        "lm_litigious_prop": "LM litigious proportion",
        "lm_litigious_tfidf": "LM litigious tf-idf",
        "lm_modal_strong_prop": "LM modal strong proportion",
        "lm_modal_strong_tfidf": "LM modal strong tf-idf",
        "lm_modal_weak_prop": "LM modal weak proportion",
        "lm_modal_weak_tfidf": "LM modal weak tf-idf",
        "finbert_neg_prob_lenw_mean": "FinBERT negative probability",
        "normalized_difference_h4n_inf": "H4N-Inf normalized difference",
        "normalized_difference_negative": "LM negative normalized difference",
    }
    expr = pl.col("coefficient_name")
    for raw_value, label in reversed(tuple(labels.items())):
        expr = pl.when(pl.col("coefficient_name") == raw_value).then(pl.lit(label)).otherwise(expr)
    return expr


def _nw_coefficient_order_expr() -> pl.Expr:
    order = {
        "h4n_inf_prop": 10,
        "lm_negative_prop": 20,
        "lm_positive_prop": 30,
        "lm_uncertainty_prop": 40,
        "lm_litigious_prop": 50,
        "lm_modal_strong_prop": 60,
        "lm_modal_weak_prop": 70,
        "h4n_inf_tfidf": 110,
        "lm_negative_tfidf": 120,
        "lm_positive_tfidf": 130,
        "lm_uncertainty_tfidf": 140,
        "lm_litigious_tfidf": 150,
        "lm_modal_strong_tfidf": 160,
        "lm_modal_weak_tfidf": 170,
        "finbert_neg_prob_lenw_mean": 220,
        "normalized_difference_h4n_inf": 310,
        "normalized_difference_negative": 320,
    }
    expr = pl.lit(999)
    for raw_value, order_value in reversed(tuple(order.items())):
        expr = pl.when(pl.col("coefficient_name") == raw_value).then(pl.lit(order_value)).otherwise(expr)
    return expr


def _nw_scale_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column).is_in(("filing_period_excess_return", "postevent_return_volatility")))
        .then(pl.lit(100.0))
        .otherwise(pl.lit(1.0))
    )


def _nw_scale_label_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column).is_in(("filing_period_excess_return", "postevent_return_volatility")))
        .then(pl.lit("x100"))
        .otherwise(pl.lit("raw"))
    )


def _table_vi_outcome_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("dependent_variable") == "filing_period_excess_return")
        .then(pl.lit("Filing-period excess return"))
        .when(pl.col("dependent_variable") == "abnormal_volume")
        .then(pl.lit("Abnormal volume"))
        .when(pl.col("dependent_variable") == "postevent_return_volatility")
        .then(pl.lit("Postevent return volatility"))
        .otherwise(pl.col("dependent_variable"))
    )


def _table_vi_outcome_order_expr() -> pl.Expr:
    expr = pl.lit(999)
    for order, dependent_variable in reversed(tuple(enumerate(TABLE_VI_DEPENDENT_VARIABLES, start=1))):
        expr = pl.when(pl.col("dependent_variable") == dependent_variable).then(pl.lit(order)).otherwise(expr)
    return expr


def _table_vi_signal_label_expr() -> pl.Expr:
    base_labels = {
        "h4n_inf": "H4N-Inf",
        "lm_negative": "LM negative",
        "lm_positive": "LM positive",
        "lm_uncertainty": "LM uncertainty",
        "lm_litigious": "LM litigious",
        "lm_modal_strong": "LM modal strong",
        "lm_modal_weak": "LM modal weak",
    }
    expr = pl.col("signal_name")
    for signal_name in reversed(TABLE_VI_SIGNAL_COLUMNS):
        base_name = signal_name.removesuffix("_prop").removesuffix("_tfidf")
        expr = pl.when(pl.col("signal_name") == signal_name).then(pl.lit(base_labels[base_name])).otherwise(expr)
    return expr


def _table_vi_signal_order_expr() -> pl.Expr:
    base_order = {
        "h4n_inf": 1,
        "lm_negative": 2,
        "lm_positive": 3,
        "lm_uncertainty": 4,
        "lm_litigious": 5,
        "lm_modal_strong": 6,
        "lm_modal_weak": 7,
    }
    expr = pl.lit(999)
    for signal_name in reversed(TABLE_VI_SIGNAL_COLUMNS):
        base_name = signal_name.removesuffix("_prop").removesuffix("_tfidf")
        expr = pl.when(pl.col("signal_name") == signal_name).then(pl.lit(base_order[base_name])).otherwise(expr)
    return expr


def _table_vi_weighting_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("signal_name").str.ends_with("_prop"))
        .then(pl.lit("Proportional"))
        .when(pl.col("signal_name").str.ends_with("_tfidf"))
        .then(pl.lit("tf-idf"))
        .otherwise(pl.lit("unknown"))
    )


def _table_vi_weight_order_expr() -> pl.Expr:
    return (
        pl.when(pl.col("signal_name").str.ends_with("_prop"))
        .then(pl.lit(1))
        .when(pl.col("signal_name").str.ends_with("_tfidf"))
        .then(pl.lit(2))
        .otherwise(pl.lit(99))
    )


def _table_vi_outcome_scale_expr() -> pl.Expr:
    return (
        pl.when(pl.col("dependent_variable").is_in(("filing_period_excess_return", "postevent_return_volatility")))
        .then(pl.lit(100.0))
        .otherwise(pl.lit(1.0))
    )


def _table_vi_reported_scale_expr() -> pl.Expr:
    return (
        pl.when(pl.col("dependent_variable").is_in(("filing_period_excess_return", "postevent_return_volatility")))
        .then(pl.lit("x100"))
        .otherwise(pl.lit("raw"))
    )


def _robustness_coefficients_lf(
    lf: pl.LazyFrame,
    *,
    family_label: str,
    signal_coefficient_expr: pl.Expr,
) -> pl.LazyFrame:
    return (
        lf.filter(
            (pl.col("estimator_status") == pl.lit("estimated"))
            & pl.col("specification_name").is_in(("finbert_only", "dictionary_finbert_joint"))
            & (pl.col("variant_id") != pl.lit("top_5_sentences_neg_mean"))
            & signal_coefficient_expr
        )
        .with_columns(
            pl.lit(family_label).alias("signal_family"),
            _robustness_signal_label_expr().alias("signal"),
            _scope_label_expr().alias("text_scope"),
            _specification_label_expr().alias("specification"),
            _quarter_weighting_label_expr().alias("quarter_weighting"),
            _stars_expr(pl.col("p_value")).alias("stars"),
            _scope_order_expr().alias("_scope_order"),
            _family_order_expr(family_label).alias("_family_order"),
            _robustness_signal_order_expr().alias("_signal_order"),
            _specification_order_expr().alias("_spec_order"),
            _quarter_weighting_order_expr().alias("_weight_order"),
        )
        .select(
            "text_scope",
            "signal_family",
            "signal",
            "specification",
            "quarter_weighting",
            pl.col("estimate"),
            pl.col("standard_error").alias("std_error"),
            "t_stat",
            "p_value",
            "stars",
            "n_obs",
            "n_quarters",
            "_scope_order",
            "_family_order",
            "_signal_order",
            "_spec_order",
            "_weight_order",
        )
    )


def _robustness_fit_comparisons_lf(
    lf: pl.LazyFrame,
    *,
    family_label: str,
) -> pl.LazyFrame:
    return (
        lf.filter(
            (pl.col("estimator_status") == pl.lit("estimated"))
            & (pl.col("variant_id") != pl.lit("top_5_sentences_neg_mean"))
        )
        .with_columns(
            pl.lit(family_label).alias("signal_family"),
            _robustness_signal_label_expr().alias("signal"),
            _scope_label_expr().alias("text_scope"),
            _comparison_label_expr().alias("comparison"),
            _stars_expr(pl.col("nw_p_value_delta_adj_r2")).alias("stars"),
            _scope_order_expr().alias("_scope_order"),
            _family_order_expr(family_label).alias("_family_order"),
            _robustness_signal_order_expr().alias("_signal_order"),
            _comparison_order_expr().alias("_comparison_order"),
        )
        .select(
            "text_scope",
            "signal_family",
            "signal",
            "comparison",
            pl.col("weighted_avg_delta_adj_r2").alias("delta_adj_r2_weighted"),
            pl.col("equal_quarter_avg_delta_adj_r2").alias("delta_adj_r2_equal_quarter"),
            pl.col("nw_t_stat_delta_adj_r2").alias("nw_t_stat"),
            pl.col("nw_p_value_delta_adj_r2").alias("nw_p_value"),
            "stars",
            "total_n_obs",
            "n_quarters",
            "_scope_order",
            "_family_order",
            "_signal_order",
            "_comparison_order",
        )
    )


def _robustness_signal_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("variant_id") == "baseline_neg_mean")
        .then(pl.lit("FinBERT negative mean"))
        .when(pl.col("variant_id") == "net_negative_mean")
        .then(pl.lit("FinBERT net negative mean"))
        .when(pl.col("variant_id") == "neg_dominant_share")
        .then(pl.lit("Negative-dominant share"))
        .when(pl.col("variant_id") == "neg_minus_pos_mean")
        .then(pl.lit("Negative minus positive mean"))
        .when(pl.col("variant_id") == "neg_mean_logit_clipped")
        .then(pl.lit("Negative mean logit"))
        .when(pl.col("variant_id") == "tail_exposure_tau_0_60")
        .then(pl.lit("Tail exposure tau=0.60"))
        .when(pl.col("variant_id") == "tail_exposure_tau_0_70")
        .then(pl.lit("Tail exposure tau=0.70"))
        .when(pl.col("variant_id") == "tail_exposure_tau_0_80")
        .then(pl.lit("Tail exposure tau=0.80"))
        .when(pl.col("variant_id") == "tail_share_tau_0_70")
        .then(pl.lit("Tail share tau=0.70"))
        .when(pl.col("variant_id") == "top_10pct_neg_mean")
        .then(pl.lit("Top 10% negative mean"))
        .when(pl.col("variant_id") == "top_20pct_neg_mean")
        .then(pl.lit("Top 20% negative mean"))
        .when(pl.col("variant_id") == "neg_prob_dispersion")
        .then(pl.lit("Negative-probability dispersion"))
        .when(pl.col("variant_id") == "baseline_neg_mean__q5_score")
        .then(pl.lit("Negative mean quintile score"))
        .when(pl.col("variant_id") == "baseline_neg_mean__q5_top_bottom")
        .then(pl.lit("Negative mean Q5-Q1"))
        .when(pl.col("variant_id") == "net_negative_mean__q5_score")
        .then(pl.lit("Net negative quintile score"))
        .when(pl.col("variant_id") == "net_negative_mean__q5_top_bottom")
        .then(pl.lit("Net negative Q5-Q1"))
        .when(pl.col("variant_id") == "top_20pct_neg_mean__q5_score")
        .then(pl.lit("Top 20% negative quintile score"))
        .when(pl.col("variant_id") == "top_20pct_neg_mean__q5_top_bottom")
        .then(pl.lit("Top 20% negative Q5-Q1"))
        .otherwise(pl.col("variant_id"))
    )


def _robustness_signal_order_expr() -> pl.Expr:
    order = {
        "baseline_neg_mean": 10,
        "net_negative_mean": 20,
        "neg_minus_pos_mean": 30,
        "neg_dominant_share": 40,
        "neg_mean_logit_clipped": 50,
        "tail_exposure_tau_0_60": 110,
        "tail_exposure_tau_0_70": 120,
        "tail_exposure_tau_0_80": 130,
        "tail_share_tau_0_70": 140,
        "top_10pct_neg_mean": 150,
        "top_20pct_neg_mean": 160,
        "neg_prob_dispersion": 170,
        "baseline_neg_mean__q5_score": 210,
        "baseline_neg_mean__q5_top_bottom": 220,
        "net_negative_mean__q5_score": 230,
        "net_negative_mean__q5_top_bottom": 240,
        "top_20pct_neg_mean__q5_score": 250,
        "top_20pct_neg_mean__q5_top_bottom": 260,
    }
    expr = pl.lit(999)
    for variant_id, order_value in reversed(tuple(order.items())):
        expr = pl.when(pl.col("variant_id") == variant_id).then(pl.lit(order_value)).otherwise(expr)
    return expr


def _scope_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("text_scope") == "item_7_mda")
        .then(pl.lit("Item 7 MD&A"))
        .when(pl.col("text_scope") == "item_1a_risk_factors")
        .then(pl.lit("Item 1A risk factors"))
        .otherwise(pl.col("text_scope"))
    )


def _scope_order_expr() -> pl.Expr:
    return (
        pl.when(pl.col("text_scope") == "item_7_mda")
        .then(pl.lit(1))
        .when(pl.col("text_scope") == "item_1a_risk_factors")
        .then(pl.lit(2))
        .otherwise(pl.lit(99))
    )


def _extension_c0_fit_filters() -> tuple[pl.Expr, ...]:
    return (
        pl.col("text_scope").is_in(TARGET_TEXT_SCOPES),
        pl.col("outcome_name") == pl.lit("filing_period_excess_return"),
        pl.col("control_set_id") == pl.lit("C0"),
        pl.col("dictionary_family_source").is_in(("replication", "extended")),
    )


def _ensure_dictionary_family_source(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    if "dictionary_family_source" in schema:
        return lf
    return lf.with_columns(pl.lit("replication").alias("dictionary_family_source"))


def _dictionary_family_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("dictionary_family_source") == "replication")
        .then(pl.lit("Replication dictionary"))
        .when(pl.col("dictionary_family_source") == "extended")
        .then(pl.lit("Extended dictionary"))
        .otherwise(pl.col("dictionary_family_source"))
    )


def _dictionary_family_order_expr() -> pl.Expr:
    return (
        pl.when(pl.col("dictionary_family_source") == "replication")
        .then(pl.lit(1))
        .when(pl.col("dictionary_family_source") == "extended")
        .then(pl.lit(2))
        .otherwise(pl.lit(99))
    )


def _specification_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("specification_name") == "dictionary_only")
        .then(pl.lit("Dictionary only"))
        .when(pl.col("specification_name") == "finbert_only")
        .then(pl.lit("FinBERT only"))
        .when(pl.col("specification_name") == "dictionary_finbert_joint")
        .then(pl.lit("Dictionary + FinBERT"))
        .otherwise(pl.col("specification_name"))
    )


def _specification_order_expr() -> pl.Expr:
    return (
        pl.when(pl.col("specification_name") == "dictionary_only")
        .then(pl.lit(1))
        .when(pl.col("specification_name") == "finbert_only")
        .then(pl.lit(2))
        .when(pl.col("specification_name") == "dictionary_finbert_joint")
        .then(pl.lit(3))
        .otherwise(pl.lit(99))
    )


def _quarter_weighting_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("weighting_rule") == "quarter_observation_count")
        .then(pl.lit("Observation-weighted quarters"))
        .when(pl.col("weighting_rule") == "equal_quarter")
        .then(pl.lit("Equal-weighted quarters"))
        .otherwise(pl.col("weighting_rule"))
    )


def _quarter_weighting_order_expr() -> pl.Expr:
    return (
        pl.when(pl.col("weighting_rule") == "quarter_observation_count")
        .then(pl.lit(1))
        .when(pl.col("weighting_rule") == "equal_quarter")
        .then(pl.lit(2))
        .otherwise(pl.lit(99))
    )


def _family_order_expr(family_label: str) -> pl.Expr:
    family_order = {
        "Existing scale": 1,
        "Tail signal": 2,
        "Quantile signal": 3,
    }.get(family_label, 99)
    return pl.lit(family_order)


def _comparison_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("comparison_name") == "finbert_minus_dictionary")
        .then(pl.lit("FinBERT - dictionary"))
        .when(pl.col("comparison_name") == "joint_minus_dictionary")
        .then(pl.lit("Joint - dictionary"))
        .when(pl.col("comparison_name") == "joint_minus_finbert")
        .then(pl.lit("Joint - FinBERT"))
        .otherwise(pl.col("comparison_name"))
    )


def _comparison_order_expr() -> pl.Expr:
    return (
        pl.when(pl.col("comparison_name") == "finbert_minus_dictionary")
        .then(pl.lit(1))
        .when(pl.col("comparison_name") == "joint_minus_dictionary")
        .then(pl.lit(2))
        .when(pl.col("comparison_name") == "joint_minus_finbert")
        .then(pl.lit(3))
        .otherwise(pl.lit(99))
    )


def _stars_expr(p_value: pl.Expr) -> pl.Expr:
    return (
        pl.when(p_value < 0.01)
        .then(pl.lit("***"))
        .when(p_value < 0.05)
        .then(pl.lit("**"))
        .when(p_value < 0.10)
        .then(pl.lit("*"))
        .otherwise(pl.lit(""))
    )


def _fit_signal_label_expr(column: str) -> pl.Expr:
    labels = {
        "lm_negative_tfidf": "LM negative tf-idf",
        "finbert_neg_prob_lenw_mean": "FinBERT negative probability",
        "lm_negative_tfidf,finbert_neg_prob_lenw_mean": (
            "LM negative tf-idf + FinBERT negative probability"
        ),
    }
    expr = pl.col(column)
    for raw_value, label in reversed(tuple(labels.items())):
        expr = pl.when(pl.col(column) == raw_value).then(pl.lit(label)).otherwise(expr)
    return expr


def _summary_statistics_table(
    lf: pl.LazyFrame,
    *,
    variables: tuple[tuple[str, str], ...],
) -> pl.DataFrame:
    schema = lf.collect_schema()
    rows: list[dict[str, object]] = []
    for column, label in variables:
        if column not in schema:
            continue
        stats = (
            lf.select(
                pl.col(column).cast(pl.Float64, strict=False).alias("_value"),
            )
            .select(
                pl.len().alias("rows"),
                pl.col("_value").drop_nulls().len().alias("non_null"),
                pl.col("_value").mean().alias("mean"),
                pl.col("_value").std().alias("sd"),
                pl.col("_value").quantile(0.25).alias("p25"),
                pl.col("_value").median().alias("median"),
                pl.col("_value").quantile(0.75).alias("p75"),
            )
            .collect()
            .to_dicts()[0]
        )
        rows.append({"variable": label, **stats})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _correlation_matrix_table(lf: pl.LazyFrame, *, columns: tuple[str, ...]) -> pl.DataFrame:
    aggregations = []
    pair_aliases: list[tuple[str, str, str]] = []
    for row_idx, left in enumerate(columns):
        for col_idx, right in enumerate(columns):
            alias = f"corr_{row_idx}_{col_idx}"
            aggregations.append(
                pl.corr(
                    pl.col(left).cast(pl.Float64, strict=False),
                    pl.col(right).cast(pl.Float64, strict=False),
                ).alias(alias)
            )
            pair_aliases.append((left, right, alias))

    corr_values = lf.select(aggregations).collect().row(0, named=True)
    return pl.DataFrame(
        [
            {
                "row_variable": _variable_label(left),
                "column_variable": _variable_label(right),
                "correlation": corr_values[alias],
            }
            for left, right, alias in pair_aliases
        ]
    )


def _score_summary_lf(
    lf: pl.LazyFrame,
    *,
    source_label: str,
    score_columns: tuple[str, ...],
) -> pl.LazyFrame:
    schema = lf.collect_schema()
    frames = []
    for column in score_columns:
        if column not in schema:
            continue
        frames.append(
            lf.select(
                pl.col("text_scope").cast(pl.Utf8, strict=False),
                pl.col(column).cast(pl.Float64, strict=False).alias("score"),
            )
            .drop_nulls(subset=["text_scope", "score"])
            .group_by("text_scope")
            .agg(
                pl.len().alias("non_null"),
                pl.col("score").mean().alias("mean"),
                pl.col("score").std().alias("sd"),
                pl.col("score").quantile(0.25).alias("p25"),
                pl.col("score").median().alias("median"),
                pl.col("score").quantile(0.75).alias("p75"),
            )
            .with_columns(
                pl.lit(source_label).alias("source"),
                _scope_label_expr().alias("scope"),
                pl.lit(_variable_label(column)).alias("score_name"),
            )
            .select("source", "scope", "text_scope", "score_name", "non_null", "mean", "sd", "p25", "median", "p75")
        )
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.LazyFrame()


def _normal_p_value_expr(t_stat: pl.Expr) -> pl.Expr:
    return t_stat.cast(pl.Float64, strict=False).map_elements(
        lambda value: None if value is None else math.erfc(abs(float(value)) / math.sqrt(2.0)),
        return_dtype=pl.Float64,
    )


def _signal_label_expr() -> pl.Expr:
    labels = {
        "h4n_inf_prop": "H4N-Inf proportion",
        "h4n_inf_tfidf": "H4N-Inf tf-idf",
        "lm_negative_prop": "LM negative proportion",
        "lm_negative_tfidf": "LM negative tf-idf",
        "finbert_neg_prob_lenw_mean": "FinBERT negative probability",
    }
    expr = pl.col("coefficient_name")
    for column, label in reversed(tuple(labels.items())):
        expr = pl.when(pl.col("coefficient_name") == column).then(pl.lit(label)).otherwise(expr)
    return expr


def _portfolio_signal_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("signal_name") == "fin_neg_prop")
        .then(pl.lit("LM negative proportion"))
        .when(pl.col("signal_name") == "fin_neg_tfidf")
        .then(pl.lit("LM negative tf-idf"))
        .when(pl.col("signal_name") == "h4n_inf_prop")
        .then(pl.lit("H4N-Inf proportion"))
        .when(pl.col("signal_name") == "h4n_inf_tfidf")
        .then(pl.lit("H4N-Inf tf-idf"))
        .otherwise(pl.col("signal_name"))
    )


def _portfolio_sort_signal_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("sort_signal_name") == "fin_neg_prop")
        .then(pl.lit("LM negative proportion"))
        .when(pl.col("sort_signal_name") == "fin_neg_tfidf")
        .then(pl.lit("LM negative tf-idf"))
        .when(pl.col("sort_signal_name") == "h4n_inf_prop")
        .then(pl.lit("H4N-Inf proportion"))
        .when(pl.col("sort_signal_name") == "h4n_inf_tfidf")
        .then(pl.lit("H4N-Inf tf-idf"))
        .otherwise(pl.col("sort_signal_name"))
    )


def _portfolio_metric_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("coefficient_name") == "mean_long_short_return")
        .then(pl.lit("Mean long-short return"))
        .when(pl.col("coefficient_name") == "alpha_ff3_mom")
        .then(pl.lit("FF3 + momentum alpha"))
        .when(pl.col("coefficient_name") == "r2")
        .then(pl.lit("Factor regression R2"))
        .otherwise(pl.col("coefficient_name"))
    )


def _variable_label(column: str) -> str:
    labels = {
        "filing_period_excess_return": "Filing-period excess return",
        "abnormal_volume": "Abnormal volume",
        "postevent_return_volatility": "Postevent return volatility",
        "lm_negative_prop": "LM negative proportion",
        "lm_negative_tfidf": "LM negative tf-idf",
        "h4n_inf_prop": "H4N-Inf proportion",
        "h4n_inf_tfidf": "H4N-Inf tf-idf",
        "finbert_neg_prob_lenw_mean": "FinBERT negative probability",
        "finbert_net_negative_lenw_mean": "FinBERT net negative",
        "log_size": "Log size",
        "log_book_to_market": "Log book-to-market",
        "log_share_turnover": "Log share turnover",
        "pre_ffalpha": "Pre-filing FF alpha",
        "nasdaq_dummy": "NASDAQ indicator",
    }
    return labels.get(column, column)


def _manifest_nested_value(manifest: dict[str, object], path: tuple[str, ...]) -> object | None:
    current: object = manifest
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _resolve_portfolio_table_artifact(
    context: BuildContext,
    fallback_artifact: ResolvedArtifact | None,
) -> tuple[ResolvedArtifact, str | None]:
    rerun_path = _latest_portfolio_rerun_path(context.repo_root, "lm2011_table_ia_ii_results.rerun.parquet")
    if rerun_path is not None:
        return _resolved_manual_lm2011_artifact(
            context,
            fallback_artifact,
            rerun_path,
            logical_name="table_ia_ii_results",
            artifact_key=ARTIFACT_KEY_TABLE_IA_II_RESULTS,
        ), f"Used newest local Table IA.II rerun artifact: {rerun_path}."
    if fallback_artifact is None:
        raise AssetBuildError("No portfolio Table IA.II artifact or local rerun artifact could be resolved.")
    return fallback_artifact, None


def _resolve_portfolio_monthly_artifact(
    context: BuildContext,
    fallback_artifact: ResolvedArtifact | None,
) -> tuple[ResolvedArtifact, str | None]:
    rerun_path = _latest_portfolio_rerun_path(context.repo_root, "lm2011_trading_strategy_monthly_returns.rerun.parquet")
    if rerun_path is not None:
        return _resolved_manual_lm2011_artifact(
            context,
            fallback_artifact,
            rerun_path,
            logical_name="trading_strategy_monthly_returns",
            artifact_key=ARTIFACT_KEY_TRADING_STRATEGY_MONTHLY_RETURNS,
        ), f"Used newest local monthly strategy rerun artifact: {rerun_path}."
    if fallback_artifact is None:
        raise AssetBuildError("No monthly strategy artifact or local rerun artifact could be resolved.")
    return fallback_artifact, None


def _latest_portfolio_rerun_path(repo_root: Path, filename: str) -> Path | None:
    candidates = [
        path.resolve()
        for path in (repo_root / "full_data_run").glob("lm2011_table_ia_ii_local_rerun_sample_*")
        if path.is_dir() and (path / filename).exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / filename).stat().st_mtime) / filename


def _resolved_manual_lm2011_artifact(
    context: BuildContext,
    fallback_artifact: ResolvedArtifact | None,
    path: Path,
    *,
    logical_name: str,
    artifact_key: str,
) -> ResolvedArtifact:
    run = fallback_artifact.run if fallback_artifact is not None else resolve_run(context, RUN_FAMILY_LM2011_POST_REFINITIV)
    requirement = (
        fallback_artifact.requirement
        if fallback_artifact is not None
        else ArtifactRequirement(
            logical_name=logical_name,
            run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
            artifact_key=artifact_key,
        )
    )
    return ResolvedArtifact(requirement=requirement, path=path.resolve(), run=run)


def _analyst_coverage_rows(context: BuildContext) -> pl.DataFrame | None:
    analyst_root = context.repo_root / "full_data_run" / "refinitiv_step1" / "analyst_common_stock"
    request_path = analyst_root / "refinitiv_analyst_request_universe_common_stock.parquet"
    normalized_path = analyst_root / "refinitiv_analyst_normalized_panel.parquet"
    frames: list[pl.DataFrame] = []
    if request_path.exists():
        frames.append(
            pl.scan_parquet(str(request_path))
            .with_columns(pl.col("actuals_request_start_date").dt.year().alias("calendar_year"))
            .group_by("calendar_year")
            .agg(
                pl.col("gvkey_int").n_unique().alias("unique_docs"),
                pl.len().alias("rows"),
                pl.col("retrieval_eligible").cast(pl.Int64).sum().alias("analyst_request_rows"),
            )
            .with_columns(
                pl.lit("refinitiv_analyst_request_universe").alias("coverage_source"),
                pl.lit(None).cast(pl.Utf8).alias("text_scope"),
                pl.lit("Analyst request universe").alias("scope"),
                pl.lit(None).cast(pl.Int64).alias("ownership_available_rows"),
                pl.lit(None).cast(pl.Float64).alias("ownership_available_rate"),
                pl.lit(None).cast(pl.Int64).alias("ownership_common_support_rows"),
                pl.lit(None).cast(pl.Float64).alias("ownership_common_support_rate"),
                pl.lit(None).cast(pl.Int64).alias("analyst_normalized_rows"),
            )
            .select(
                "coverage_source",
                "calendar_year",
                "text_scope",
                "scope",
                "unique_docs",
                "rows",
                "ownership_available_rows",
                "ownership_available_rate",
                "ownership_common_support_rows",
                "ownership_common_support_rate",
                "analyst_request_rows",
                "analyst_normalized_rows",
            )
            .collect()
        )
    if normalized_path.exists():
        frames.append(
            pl.scan_parquet(str(normalized_path))
            .with_columns(pl.col("announcement_date").dt.year().alias("calendar_year"))
            .group_by("calendar_year")
            .agg(
                pl.col("gvkey_int").n_unique().alias("unique_docs"),
                pl.len().alias("rows"),
            )
            .with_columns(
                pl.lit("refinitiv_analyst_normalized_panel").alias("coverage_source"),
                pl.lit(None).cast(pl.Utf8).alias("text_scope"),
                pl.lit("Analyst normalized panel").alias("scope"),
                pl.lit(None).cast(pl.Int64).alias("ownership_available_rows"),
                pl.lit(None).cast(pl.Float64).alias("ownership_available_rate"),
                pl.lit(None).cast(pl.Int64).alias("ownership_common_support_rows"),
                pl.lit(None).cast(pl.Float64).alias("ownership_common_support_rate"),
                pl.lit(None).cast(pl.Int64).alias("analyst_request_rows"),
                pl.col("rows").alias("analyst_normalized_rows"),
            )
            .select(
                "coverage_source",
                "calendar_year",
                "text_scope",
                "scope",
                "unique_docs",
                "rows",
                "ownership_available_rows",
                "ownership_available_rate",
                "ownership_common_support_rows",
                "ownership_common_support_rate",
                "analyst_request_rows",
                "analyst_normalized_rows",
            )
            .collect()
        )
    if not frames:
        return None
    return pl.concat(frames, how="vertical_relaxed")


def _resolve_finbert_preprocessing_path(
    context: BuildContext,
    finbert_artifact: ResolvedArtifact,
    filename: str,
) -> Path:
    run = finbert_artifact.run
    run_name = run.root.name
    if isinstance(run.manifest, dict) and isinstance(run.manifest.get("run_name"), str):
        run_name = str(run.manifest["run_name"])
    candidates = (
        run.root.parent / "_staged_intermediates" / f"{run_name}_sentence_preprocessing" / filename,
        run.root / "_staged_intermediates" / f"{run_name}_sentence_preprocessing" / filename,
        run.root / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise AssetBuildError(f"Could not resolve FinBERT preprocessing artifact {filename!r}. Checked: {checked}")


def _resolve_generated_dictionary_family_root(context: BuildContext) -> Path:
    candidates = (
        context.repo_root / "full_data_run" / "LM2011_additional_data" / "generated_dictionary_families",
        context.repo_root / "LM2011_additional_data" / "generated_dictionary_families",
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise AssetBuildError(f"Could not resolve generated dictionary families. Checked: {checked}")


def _count_nonblank_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def _skipped_quarter_summary_lf(lf: pl.LazyFrame, source: str) -> pl.LazyFrame:
    schema = lf.collect_schema()
    group_cols = ["skip_reason"]
    for optional in ("text_scope", "dependent_variable", "outcome_name", "feature_family", "variant_family"):
        if optional in schema:
            group_cols.append(optional)
    return (
        lf.with_columns(pl.lit(source).alias("source"))
        .group_by("source", *group_cols)
        .agg(
            pl.len().alias("skipped_quarters"),
            pl.col("n_obs").min().alias("min_n_obs") if "n_obs" in schema else pl.lit(None).alias("min_n_obs"),
            pl.col("n_obs").median().alias("median_n_obs") if "n_obs" in schema else pl.lit(None).alias("median_n_obs"),
            pl.col("n_obs").max().alias("max_n_obs") if "n_obs" in schema else pl.lit(None).alias("max_n_obs"),
        )
        .with_columns(
            pl.col("text_scope").cast(pl.Utf8, strict=False) if "text_scope" in group_cols else pl.lit(None).alias("text_scope"),
            pl.col("dependent_variable").cast(pl.Utf8, strict=False).alias("dependent_variable")
            if "dependent_variable" in group_cols
            else pl.col("outcome_name").cast(pl.Utf8, strict=False).alias("dependent_variable")
            if "outcome_name" in group_cols
            else pl.lit(None).alias("dependent_variable"),
            pl.col("feature_family").cast(pl.Utf8, strict=False).alias("feature_family")
            if "feature_family" in group_cols
            else pl.lit(None).alias("feature_family"),
            pl.col("variant_family").cast(pl.Utf8, strict=False).alias("variant_family")
            if "variant_family" in group_cols
            else pl.lit(None).alias("variant_family"),
        )
        .select(
            "source",
            "text_scope",
            "dependent_variable",
            "feature_family",
            "variant_family",
            "skip_reason",
            "skipped_quarters",
            "min_n_obs",
            "median_n_obs",
            "max_n_obs",
        )
    )


def _robustness_full_grid_lf(lf: pl.LazyFrame, signal_family: str) -> pl.LazyFrame:
    return (
        lf.filter(
            (pl.col("estimator_status") == pl.lit("estimated"))
            & (
                (pl.col("coefficient_name") == pl.col("signal_name"))
                | (pl.col("coefficient_name") == pl.col("variant_id"))
                | (pl.col("coefficient_name") == pl.lit("finbert_neg_prob_lenw_mean"))
            )
        )
        .with_columns(
            pl.lit(signal_family).alias("signal_family"),
            _scope_label_expr().alias("scope"),
            _specification_label_expr().alias("specification"),
            _stars_expr(pl.col("p_value")).alias("stars"),
        )
        .select(
            "signal_family",
            "scope",
            "text_scope",
            "variant_id",
            "variant_description",
            "specification",
            "coefficient_name",
            "estimate",
            pl.col("standard_error").alias("std_error"),
            "t_stat",
            "p_value",
            "stars",
            "n_obs",
            "n_quarters",
            "mean_quarter_n",
            "weighting_rule",
        )
    )


def _control_coefficients_lf(
    lf: pl.LazyFrame,
    source: str,
    control_columns: tuple[str, ...],
) -> pl.LazyFrame:
    schema = lf.collect_schema()
    select_cols = [
        pl.lit(source).alias("source"),
        pl.col("text_scope").cast(pl.Utf8, strict=False) if "text_scope" in schema else pl.lit(None).alias("text_scope"),
        pl.col("dependent_variable").cast(pl.Utf8, strict=False).alias("dependent_variable")
        if "dependent_variable" in schema
        else pl.col("outcome_name").cast(pl.Utf8, strict=False).alias("dependent_variable")
        if "outcome_name" in schema
        else pl.lit(None).alias("dependent_variable"),
        pl.col("control_set_id").cast(pl.Utf8, strict=False)
        if "control_set_id" in schema
        else pl.lit(None).alias("control_set_id"),
        pl.col("signal_name").cast(pl.Utf8, strict=False) if "signal_name" in schema else pl.lit(None).alias("signal_name"),
        pl.col("variant_id").cast(pl.Utf8, strict=False)
        if "variant_id" in schema
        else pl.lit(None).alias("variant_id"),
        pl.col("variant_description").cast(pl.Utf8, strict=False)
        if "variant_description" in schema
        else pl.lit(None).alias("variant_description"),
        pl.col("specification_name").cast(pl.Utf8, strict=False).alias("specification_name")
        if "specification_name" in schema
        else pl.col("specification_id").cast(pl.Utf8, strict=False).alias("specification_name")
        if "specification_id" in schema
        else pl.lit(None).alias("specification_name"),
        "coefficient_name",
        "estimate",
        pl.col("standard_error").alias("std_error") if "standard_error" in schema else pl.lit(None).alias("std_error"),
        "t_stat",
        pl.col("p_value") if "p_value" in schema else pl.lit(None).alias("p_value"),
        pl.col("n_obs") if "n_obs" in schema else pl.lit(None).alias("n_obs"),
        pl.col("n_quarters") if "n_quarters" in schema else pl.lit(None).alias("n_quarters"),
        pl.col("weighting_rule").cast(pl.Utf8, strict=False)
        if "weighting_rule" in schema
        else pl.lit(None).alias("weighting_rule"),
    ]
    return lf.filter(pl.col("coefficient_name").is_in(control_columns)).select(*select_cols)


def _write_table_outputs(
    context: BuildContext,
    spec: AssetSpec,
    df: pl.DataFrame,
) -> dict[str, str]:
    csv_path = context.output_dirs["csv"] / f"{spec.output_stem}.csv"
    tex_path = context.output_dirs["tex"] / f"{spec.output_stem}.tex"
    markdown_path = context.output_dirs["tables"] / f"{spec.output_stem}.md"
    write_csv_table(df, csv_path)
    write_latex_table(df, tex_path, caption=spec.caption_stub, notes=spec.notes_stub)
    write_markdown_table(df, markdown_path)
    return {
        "csv": str(csv_path),
        "tex": str(tex_path),
        "table_preview": str(markdown_path),
    }


def _write_figure_outputs(
    context: BuildContext,
    spec: AssetSpec,
    df: pl.DataFrame,
    figure,
) -> dict[str, str]:
    csv_path = context.output_dirs["csv"] / f"{spec.output_stem}.csv"
    write_csv_table(df, csv_path)
    figure_paths = write_figure_bundle(figure, context.output_dirs["figures"] / spec.output_stem)
    return {
        "csv": str(csv_path),
        "png": str(figure_paths["png"]),
        "pdf": str(figure_paths["pdf"]),
    }


def _document_score_long_df(
    lf: pl.LazyFrame,
    *,
    metric_specs: tuple[tuple[str, str, str], ...],
    filters: tuple[pl.Expr, ...],
) -> pl.DataFrame:
    filtered = lf
    for predicate in filters:
        filtered = filtered.filter(predicate)
    frames = [
        filtered.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("filing_date"),
            pl.col("text_scope").cast(pl.Utf8, strict=False),
            _scope_label_expr().alias("scope_label"),
            pl.lit(metric_id).alias("metric_id"),
            pl.lit(metric_label).alias("metric_label"),
            pl.concat_str([pl.lit(metric_label), pl.lit(" - "), _scope_label_expr()]).alias("series_label"),
            pl.col(score_column).cast(pl.Float64, strict=False).alias("score"),
        )
        for metric_id, metric_label, score_column in metric_specs
    ]
    return pl.concat(frames, how="vertical_relaxed").drop_nulls(subset=["score"]).collect()


def _exact_ecdf_frame(df: pl.DataFrame, *, score_col: str) -> pl.DataFrame:
    if df.is_empty():
        return _empty_exact_ecdf_frame()
    required = {"metric_id", "metric_label", "text_scope", "scope_label", "series_label", score_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise AssetBuildError(f"Cannot build ECDF; missing columns: {missing}")

    rows: list[dict[str, object]] = []
    groups = (
        df.select("metric_id", "metric_label", "text_scope", "scope_label", "series_label")
        .unique()
        .sort("metric_id", "text_scope")
        .to_dicts()
    )
    for group in groups:
        group_df = (
            df.filter(
                (pl.col("metric_id") == group["metric_id"])
                & (pl.col("text_scope") == group["text_scope"])
            )
            .select(pl.col(score_col).cast(pl.Float64, strict=False).alias("score"))
            .drop_nulls()
            .sort("score")
        )
        total = group_df.height
        if total == 0:
            continue
        for idx, value in enumerate(group_df.get_column("score").to_list(), start=1):
            rows.append(
                {
                    **group,
                    "score": float(value),
                    "rank": idx,
                    "total_count": total,
                    "ecdf": float(idx) / float(total),
                }
            )
    return pl.DataFrame(rows) if rows else _empty_exact_ecdf_frame()


def _empty_exact_ecdf_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "metric_id": pl.Utf8,
            "metric_label": pl.Utf8,
            "text_scope": pl.Utf8,
            "scope_label": pl.Utf8,
            "series_label": pl.Utf8,
            "score": pl.Float64,
            "rank": pl.Int64,
            "total_count": pl.Int64,
            "ecdf": pl.Float64,
        }
    )


def _cached_finbert_sentence_summary(
    context: BuildContext,
    artifacts: dict[str, ResolvedArtifact],
):
    analysis_artifact = artifacts["extension_analysis_panel"]
    sentence_artifact = artifacts["sentence_scores"]
    cache_key = f"finbert_sentence_summary::{analysis_artifact.path}::{sentence_artifact.path}"
    cached = context.asset_cache.get(cache_key)
    if cached is not None:
        return cached
    summary = build_finbert_sentence_summary(
        analysis_artifact=analysis_artifact,
        sentence_artifact=sentence_artifact,
        batch_size=sentence_batch_size_from_env(),
    )
    context.asset_cache[cache_key] = summary
    return summary


def _cached_lm_sentence_summary(
    context: BuildContext,
    artifacts: dict[str, ResolvedArtifact],
):
    analysis_artifact = artifacts["extension_analysis_panel"]
    sentence_artifact = artifacts["sentence_scores"]
    negative_word_path = _resolve_replication_negative_word_list(context, analysis_artifact)
    cache_key = f"lm_sentence_summary::{analysis_artifact.path}::{sentence_artifact.path}::{negative_word_path}"
    cached = context.asset_cache.get(cache_key)
    if cached is not None:
        return cached
    summary = build_lm_negative_sentence_summary(
        analysis_artifact=analysis_artifact,
        sentence_artifact=sentence_artifact,
        negative_words=load_lm2011_word_list(negative_word_path),
        batch_size=sentence_batch_size_from_env(),
    )
    context.asset_cache[cache_key] = summary
    return summary


def _resolve_replication_negative_word_list(
    context: BuildContext,
    extension_artifact: ResolvedArtifact,
) -> Path:
    candidates: list[Path] = []
    manifest = extension_artifact.run.manifest
    if isinstance(manifest, dict):
        _extend_negative_word_candidates_from_manifest(context, candidates, manifest)

    candidates.extend(
        [
            context.repo_root
            / "full_data_run"
            / "LM2011_additional_data"
            / "generated_dictionary_families"
            / "replication"
            / "Fin-Neg.txt",
            context.repo_root / "LM2011_additional_data" / "generated_dictionary_families" / "replication" / "Fin-Neg.txt",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_file() and resolved.stat().st_size > 0:
            return resolved
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise AssetBuildError(f"Could not resolve replication Fin-Neg.txt word list. Checked: {checked}")


def _extend_negative_word_candidates_from_manifest(
    context: BuildContext,
    candidates: list[Path],
    manifest: dict[str, object],
) -> None:
    dictionary_inputs = manifest.get("dictionary_inputs")
    if isinstance(dictionary_inputs, dict):
        resources = dictionary_inputs.get("resources")
        if isinstance(resources, list):
            for resource in resources:
                if not isinstance(resource, dict):
                    continue
                if resource.get("role") == "dictionary_list:negative":
                    raw_path = resource.get("path")
                    if isinstance(raw_path, str):
                        candidates.extend(_manifest_path_candidates(context, raw_path))

        generated = dictionary_inputs.get("generated_dictionary_families")
        if isinstance(generated, dict):
            replication = generated.get("replication")
            if isinstance(replication, dict):
                dictionary_lists = replication.get("dictionary_lists")
                if isinstance(dictionary_lists, dict):
                    negative = dictionary_lists.get("negative")
                    if isinstance(negative, dict):
                        raw_path = negative.get("path")
                        if isinstance(raw_path, str):
                            candidates.extend(_manifest_path_candidates(context, raw_path))

    family_runs = manifest.get("family_runs")
    if isinstance(family_runs, dict):
        replication_run = family_runs.get("replication")
        if isinstance(replication_run, dict):
            raw_dir = replication_run.get("dictionary_input_dir")
            if isinstance(raw_dir, str):
                for candidate in _manifest_path_candidates(context, raw_dir):
                    candidates.append(candidate / "Fin-Neg.txt")


def _manifest_path_candidates(context: BuildContext, raw_path: str) -> list[Path]:
    raw = Path(raw_path)
    candidates = [raw]
    parts = raw.parts
    if "LM2011_additional_data" in parts:
        index = parts.index("LM2011_additional_data")
        suffix_parts = parts[index + 1 :]
        if suffix_parts:
            suffix = Path(*suffix_parts)
            candidates.append(context.repo_root / "full_data_run" / "LM2011_additional_data" / suffix)
            candidates.append(context.repo_root / "LM2011_additional_data" / suffix)
    return candidates


def _sentence_figure_result(
    context: BuildContext,
    spec: AssetSpec,
    artifacts: dict[str, ResolvedArtifact],
    output_paths: dict[str, str],
    *,
    row_counts: dict[str, int],
) -> BuildResult:
    return BuildResult(
        asset_id=spec.asset_id,
        chapter=spec.chapter,
        asset_kind=spec.asset_kind,
        sample_contract_id=spec.sample_contract_id,
        status="completed",
        resolved_inputs={
            "extension_analysis_panel": str(artifacts["extension_analysis_panel"].path),
            "sentence_scores": str(artifacts["sentence_scores"].path),
        },
        output_paths=output_paths,
        row_counts=row_counts,
    )


def _require_contract(spec: AssetSpec, expected_contract: str) -> None:
    if spec.sample_contract_id != expected_contract:
        raise AssetBuildError(
            f"Asset {spec.asset_id!r} declares sample contract {spec.sample_contract_id!r}, "
            f"but builder expected {expected_contract!r}."
        )
