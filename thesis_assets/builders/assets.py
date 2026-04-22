from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_assets.builders.artifacts import resolve_required_artifacts
from thesis_assets.builders.artifacts import scan_parquet_artifact
from thesis_assets.builders.sample_contracts import common_row_comparison
from thesis_assets.builders.sample_contracts import common_success_comparison
from thesis_assets.builders.sample_contracts import raw_available
from thesis_assets.config.constants import DEFAULT_COMMON_SUCCESS_POLICY
from thesis_assets.config.constants import DEFAULT_COMPARISON_JOIN_KEYS
from thesis_assets.errors import AssetBuildError
from thesis_assets.figures import build_concordance_figure
from thesis_assets.renderers import write_csv_table
from thesis_assets.renderers import write_figure_bundle
from thesis_assets.renderers import write_latex_table
from thesis_assets.renderers import write_markdown_table
from thesis_assets.specs import AssetSpec
from thesis_assets.specs import BuildContext
from thesis_assets.specs import BuildResult
from thesis_assets.specs import ResolvedArtifact


def build_asset(context: BuildContext, spec: AssetSpec) -> BuildResult:
    context.logger.info("Building asset %s", spec.asset_id)
    try:
        artifact_map = resolve_required_artifacts(context, spec)
        if spec.builder_id == "chapter4_sample_attrition":
            return _build_chapter4_sample_attrition(context, spec, artifact_map)
        if spec.builder_id == "chapter5_fit_horserace":
            return _build_chapter5_fit_horserace(context, spec, artifact_map)
        if spec.builder_id == "chapter5_concordance":
            return _build_chapter5_concordance(context, spec, artifact_map)
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


def _require_contract(spec: AssetSpec, expected_contract: str) -> None:
    if spec.sample_contract_id != expected_contract:
        raise AssetBuildError(
            f"Asset {spec.asset_id!r} declares sample contract {spec.sample_contract_id!r}, "
            f"but builder expected {expected_contract!r}."
        )
