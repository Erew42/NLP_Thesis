from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import FinbertAnalysisRunArtifacts
from thesis_pkg.benchmarking.contracts import FinbertAnalysisRunConfig
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import FinbertSentenceParquetInferenceRunConfig
from thesis_pkg.benchmarking.contracts import FinbertSentencePreprocessingRunConfig
from thesis_pkg.benchmarking.run_logging import utc_timestamp
from thesis_pkg.benchmarking.finbert_sentence_preprocessing import run_finbert_sentence_preprocessing


ITEM_FEATURE_METRIC_COLUMNS: tuple[str, ...] = (
    "sentence_count",
    "negative_prob_mean",
    "neutral_prob_mean",
    "positive_prob_mean",
    "argmax_share_negative",
    "argmax_share_neutral",
    "argmax_share_positive",
    "sentiment_balance_mean",
)
TARGET_SENTIMENT_LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
RUNNER_NAME = "finbert_item_analysis"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _empty_item_features_long_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "benchmark_row_id": pl.Utf8,
            "doc_id": pl.Utf8,
            "cik_10": pl.Utf8,
            "accession_nodash": pl.Utf8,
            "filing_date": pl.Date,
            "filing_year": pl.Int32,
            "source_year_file": pl.Int32,
            "document_type": pl.Utf8,
            "document_type_raw": pl.Utf8,
            "document_type_normalized": pl.Utf8,
            "benchmark_item_code": pl.Utf8,
            "benchmark_item_label": pl.Utf8,
            "sentence_count": pl.Int32,
            "negative_prob_mean": pl.Float64,
            "neutral_prob_mean": pl.Float64,
            "positive_prob_mean": pl.Float64,
            "argmax_share_negative": pl.Float64,
            "argmax_share_neutral": pl.Float64,
            "argmax_share_positive": pl.Float64,
            "sentiment_balance_mean": pl.Float64,
        }
    )


def _empty_doc_features_wide_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "doc_id": pl.Utf8,
            "cik_10": pl.Utf8,
            "accession_nodash": pl.Utf8,
            "filing_date": pl.Date,
            "filing_year": pl.Int32,
            "source_year_file": pl.Int32,
        }
    )


def _empty_coverage_report_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "doc_id": pl.Utf8,
            "has_finbert_features": pl.Boolean,
            "has_item_1": pl.Boolean,
            "has_item_1a": pl.Boolean,
            "has_item_7": pl.Boolean,
        }
    )


def _annotate_analysis_sections(sections_df: pl.DataFrame) -> pl.DataFrame:
    if sections_df.is_empty():
        return sections_df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("benchmark_row_id"))
    return sections_df.with_columns(
        pl.concat_str([pl.col("doc_id"), pl.lit(":"), pl.col("benchmark_item_code")]).alias("benchmark_row_id")
    )


def aggregate_sentence_scores_to_item_features(
    sentence_scores_df: pl.DataFrame,
    sections_df: pl.DataFrame,
) -> pl.DataFrame:
    metadata = sections_df.select(
        [
            "benchmark_row_id",
            "doc_id",
            "cik_10",
            "accession_nodash",
            "filing_date",
            "filing_year",
            "source_year_file",
            "document_type",
            "document_type_raw",
            "document_type_normalized",
            "benchmark_item_code",
            "benchmark_item_label",
        ]
    ).sort(["filing_year", "doc_id", "benchmark_item_code"])
    if metadata.is_empty():
        return _empty_item_features_long_frame()

    if sentence_scores_df.is_empty():
        return (
            metadata.with_columns(
                [
                    pl.lit(0, dtype=pl.Int32).alias("sentence_count"),
                    pl.lit(None, dtype=pl.Float64).alias("negative_prob_mean"),
                    pl.lit(None, dtype=pl.Float64).alias("neutral_prob_mean"),
                    pl.lit(None, dtype=pl.Float64).alias("positive_prob_mean"),
                    pl.lit(None, dtype=pl.Float64).alias("argmax_share_negative"),
                    pl.lit(None, dtype=pl.Float64).alias("argmax_share_neutral"),
                    pl.lit(None, dtype=pl.Float64).alias("argmax_share_positive"),
                    pl.lit(None, dtype=pl.Float64).alias("sentiment_balance_mean"),
                ]
            )
            .select(_empty_item_features_long_frame().columns)
        )

    aggregated = (
        sentence_scores_df.group_by("benchmark_row_id")
        .agg(
            [
                pl.len().cast(pl.Int32).alias("sentence_count"),
                pl.col("negative_prob").mean().alias("negative_prob_mean"),
                pl.col("neutral_prob").mean().alias("neutral_prob_mean"),
                pl.col("positive_prob").mean().alias("positive_prob_mean"),
                (pl.col("predicted_label") == "negative").mean().alias("argmax_share_negative"),
                (pl.col("predicted_label") == "neutral").mean().alias("argmax_share_neutral"),
                (pl.col("predicted_label") == "positive").mean().alias("argmax_share_positive"),
            ]
        )
        .with_columns(
            (pl.col("positive_prob_mean") - pl.col("negative_prob_mean")).alias("sentiment_balance_mean")
        )
    )

    return (
        metadata.join(aggregated, on="benchmark_row_id", how="left")
        .with_columns(pl.col("sentence_count").fill_null(0).cast(pl.Int32))
        .select(_empty_item_features_long_frame().columns)
        .sort(["filing_year", "doc_id", "benchmark_item_code"])
    )


def pivot_item_features_to_doc_wide(item_features_df: pl.DataFrame) -> pl.DataFrame:
    if item_features_df.is_empty():
        return _empty_doc_features_wide_frame()

    metadata_columns = [
        "cik_10",
        "accession_nodash",
        "filing_date",
        "filing_year",
        "source_year_file",
    ]
    doc_wide = (
        item_features_df.group_by("doc_id")
        .agg([pl.col(column).drop_nulls().first().alias(column) for column in metadata_columns])
        .sort(["filing_year", "doc_id"])
    )
    for metric in ITEM_FEATURE_METRIC_COLUMNS:
        metric_wide = item_features_df.select(["doc_id", "benchmark_item_code", metric]).pivot(
            values=metric,
            index="doc_id",
            on="benchmark_item_code",
            aggregate_function="first",
        )
        rename_map = {
            column: f"{column}_{metric}"
            for column in metric_wide.columns
            if column != "doc_id"
        }
        doc_wide = doc_wide.join(metric_wide.rename(rename_map), on="doc_id", how="left")
    return doc_wide


def build_coverage_report(
    item_features_df: pl.DataFrame,
    backbone_df: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    if "doc_id" not in backbone_df.columns:
        raise ValueError("Backbone parquet must contain a doc_id column for coverage reporting.")

    backbone_docs = (
        backbone_df.select(pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"))
        .filter(pl.col("doc_id").is_not_null())
        .unique(maintain_order=True)
        .sort("doc_id")
    )
    if backbone_docs.is_empty():
        return _empty_coverage_report_frame(), {
            "backbone_doc_count": 0,
            "covered_doc_count": 0,
            "covered_item_1_doc_count": 0,
            "covered_item_1a_doc_count": 0,
            "covered_item_7_doc_count": 0,
        }

    if item_features_df.is_empty():
        coverage = backbone_docs.with_columns(
            [
                pl.lit(False).alias("has_finbert_features"),
                pl.lit(False).alias("has_item_1"),
                pl.lit(False).alias("has_item_1a"),
                pl.lit(False).alias("has_item_7"),
            ]
        )
        return coverage, {
            "backbone_doc_count": int(coverage.height),
            "covered_doc_count": 0,
            "covered_item_1_doc_count": 0,
            "covered_item_1a_doc_count": 0,
            "covered_item_7_doc_count": 0,
        }

    coverage_flags = (
        item_features_df.group_by("doc_id")
        .agg(
            [
                pl.len().gt(0).alias("has_finbert_features"),
                (pl.col("benchmark_item_code") == "item_1").any().alias("has_item_1"),
                (pl.col("benchmark_item_code") == "item_1a").any().alias("has_item_1a"),
                (pl.col("benchmark_item_code") == "item_7").any().alias("has_item_7"),
            ]
        )
        .with_columns(pl.col("doc_id").cast(pl.Utf8, strict=False))
    )
    coverage = (
        backbone_docs.join(coverage_flags, on="doc_id", how="left")
        .with_columns(
            [
                pl.col("has_finbert_features").fill_null(False),
                pl.col("has_item_1").fill_null(False),
                pl.col("has_item_1a").fill_null(False),
                pl.col("has_item_7").fill_null(False),
            ]
        )
        .select(_empty_coverage_report_frame().columns)
    )
    summary = {
        "backbone_doc_count": int(coverage.height),
        "covered_doc_count": int(coverage["has_finbert_features"].sum()),
        "covered_item_1_doc_count": int(coverage["has_item_1"].sum()),
        "covered_item_1a_doc_count": int(coverage["has_item_1a"].sum()),
        "covered_item_7_doc_count": int(coverage["has_item_7"].sum()),
    }
    return coverage, summary


def _align_frame_to_schema(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    return df.select(
        [
            (
                pl.col(name).cast(dtype, strict=False).alias(name)
                if name in df.columns
                else pl.lit(None, dtype=dtype).alias(name)
            )
            for name, dtype in schema.items()
        ]
    )


def _concat_aligned_frames(frames: list[pl.DataFrame], *, empty_schema: pl.DataFrame) -> pl.DataFrame:
    if not frames:
        return empty_schema
    schema = frames[0].schema
    return pl.concat([_align_frame_to_schema(frame, schema) for frame in frames], how="vertical")


def run_finbert_item_analysis(
    run_cfg: FinbertAnalysisRunConfig,
    *,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
) -> FinbertAnalysisRunArtifacts:
    run_name = run_cfg.run_name or f"{RUNNER_NAME}_{utc_timestamp().replace(':', '')}"
    from thesis_pkg.benchmarking.finbert_staged_inference import (
        run_finbert_sentence_parquet_inference,
    )

    preprocessing_artifacts = run_finbert_sentence_preprocessing(
        FinbertSentencePreprocessingRunConfig(
            source_items_dir=run_cfg.source_items_dir,
            out_root=run_cfg.out_root / "_staged_intermediates",
            section_universe=run_cfg.section_universe,
            sentence_dataset=run_cfg.sentence_dataset,
            target_doc_universe_path=run_cfg.backbone_path,
            year_filter=run_cfg.year_filter,
            overwrite=run_cfg.overwrite,
            run_name=f"{run_name}_sentence_preprocessing",
            note=run_cfg.note,
        ),
        authority=authority,
    )
    inference_artifacts = run_finbert_sentence_parquet_inference(
        FinbertSentenceParquetInferenceRunConfig(
            sentence_dataset_dir=preprocessing_artifacts.sentence_dataset_dir,
            out_root=run_cfg.out_root,
            batch_config=run_cfg.batch_config,
            runtime=run_cfg.runtime,
            bucket_lengths=run_cfg.bucket_lengths,
            backbone_path=run_cfg.backbone_path,
            year_filter=run_cfg.year_filter,
            write_sentence_scores=run_cfg.write_sentence_scores,
            overwrite=run_cfg.overwrite,
            run_name=run_name,
            note=run_cfg.note,
        ),
        authority=authority,
    )
    return FinbertAnalysisRunArtifacts(
        run_dir=inference_artifacts.run_dir,
        run_manifest_path=inference_artifacts.run_manifest_path,
        item_features_long_path=inference_artifacts.item_features_long_path,
        doc_features_wide_path=inference_artifacts.doc_features_wide_path,
        coverage_report_path=inference_artifacts.coverage_report_path,
        sentence_scores_dir=inference_artifacts.sentence_scores_dir,
    )
