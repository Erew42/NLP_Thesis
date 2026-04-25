from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_assets.builders.artifacts import resolve_required_artifacts
from thesis_assets.builders.artifacts import scan_parquet_artifact
from thesis_assets.builders.sentence_summaries import build_finbert_sentence_summary
from thesis_assets.builders.sentence_summaries import build_lm_negative_sentence_summary
from thesis_assets.builders.sentence_summaries import sentence_batch_size_from_env
from thesis_assets.builders.sample_contracts import common_row_comparison
from thesis_assets.builders.sample_contracts import common_success_comparison
from thesis_assets.builders.sample_contracts import raw_available
from thesis_assets.config.constants import DEFAULT_COMMON_SUCCESS_POLICY
from thesis_assets.config.constants import DEFAULT_COMPARISON_JOIN_KEYS
from thesis_assets.errors import AssetBuildError
from thesis_assets.figures import build_concordance_figure
from thesis_assets.figures import build_concordance_by_scope_figure
from thesis_assets.figures import build_ecdf_lines_figure
from thesis_assets.figures import build_sample_attrition_figure
from thesis_assets.figures import build_sample_bridge_figure
from thesis_assets.figures import build_sample_funnel_figure
from thesis_assets.renderers import write_csv_table
from thesis_assets.renderers import write_figure_bundle
from thesis_assets.renderers import write_latex_table
from thesis_assets.renderers import write_markdown_table
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
        if spec.builder_id == "chapter5_fit_horserace":
            return _build_chapter5_fit_horserace(context, spec, artifact_map)
        if spec.builder_id == "chapter5_lm2011_table_vi_no_ownership":
            return _build_chapter5_lm2011_table_vi_no_ownership(context, spec, artifact_map)
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
        if spec.builder_id == "chapter5_finbert_robustness_coefficients":
            return _build_chapter5_finbert_robustness_coefficients(context, spec, artifact_map)
        if spec.builder_id == "chapter5_finbert_robustness_fit_comparisons":
            return _build_chapter5_finbert_robustness_fit_comparisons(context, spec, artifact_map)
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

    figure = build_ecdf_lines_figure(ecdf_df, x_col="score", x_label="Document score")
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


def _specification_label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("specification_name") == "finbert_only")
        .then(pl.lit("FinBERT only"))
        .when(pl.col("specification_name") == "dictionary_finbert_joint")
        .then(pl.lit("Dictionary + FinBERT"))
        .otherwise(pl.col("specification_name"))
    )


def _specification_order_expr() -> pl.Expr:
    return (
        pl.when(pl.col("specification_name") == "finbert_only")
        .then(pl.lit(1))
        .when(pl.col("specification_name") == "dictionary_finbert_joint")
        .then(pl.lit(2))
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
