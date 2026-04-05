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
from thesis_pkg.benchmarking.finbert_benchmark import _empty_sentence_score_frame
from thesis_pkg.benchmarking.finbert_benchmark import _resolve_device
from thesis_pkg.benchmarking.finbert_benchmark import _runtime_environment
from thesis_pkg.benchmarking.finbert_benchmark import load_finbert_model
from thesis_pkg.benchmarking.finbert_benchmark import resolve_finbert_label_mapping
from thesis_pkg.benchmarking.finbert_benchmark import score_sentence_frame
from thesis_pkg.benchmarking.finbert_dataset import _resolve_year_paths
from thesis_pkg.benchmarking.finbert_dataset import load_eligible_section_universe
from thesis_pkg.benchmarking.run_logging import utc_timestamp
from thesis_pkg.benchmarking.sentences import derive_sentence_frame
from thesis_pkg.benchmarking.token_lengths import load_finbert_tokenizer


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


def _resolve_analysis_year_paths(run_cfg: FinbertAnalysisRunConfig) -> list[Path]:
    year_paths = _resolve_year_paths(run_cfg.source_items_dir)
    if run_cfg.year_filter is None:
        return year_paths

    target_years = set(run_cfg.year_filter)
    selected = [path for path in year_paths if int(path.stem) in target_years]
    missing_years = sorted(target_years - {int(path.stem) for path in selected})
    if missing_years:
        raise FileNotFoundError(
            f"Requested filing years were not found in {run_cfg.source_items_dir}: {missing_years}"
        )
    return selected


def _load_backbone_doc_ids(path: Path) -> pl.DataFrame:
    return pl.scan_parquet(path).select(pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id")).collect()


def run_finbert_item_analysis(
    run_cfg: FinbertAnalysisRunConfig,
    *,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
) -> FinbertAnalysisRunArtifacts:
    year_paths = _resolve_analysis_year_paths(run_cfg)
    run_name = run_cfg.run_name or f"{RUNNER_NAME}_{utc_timestamp().replace(':', '')}"
    run_dir = run_cfg.out_root / run_name
    item_features_by_year_dir = run_dir / "item_features" / "by_year"
    sentence_scores_dir = run_dir / "sentence_scores" / "by_year"
    item_features_long_path = run_dir / "item_features_long.parquet"
    doc_features_wide_path = run_dir / "doc_features_wide.parquet"
    coverage_report_path = run_dir / "coverage_report.parquet" if run_cfg.backbone_path is not None else None
    run_manifest_path = run_dir / "run_manifest.json"

    runtime_environment = _runtime_environment(
        run_cfg.runtime,
        _resolve_device(run_cfg.runtime),
    )
    tokenizer = None
    model = None
    label_mapping: dict[int, str] | None = None
    warnings: list[str] = []
    year_results: list[dict[str, Any]] = []
    item_feature_frames: list[pl.DataFrame] = []

    for year_path in year_paths:
        filing_year = int(year_path.stem)
        year_item_features_path = item_features_by_year_dir / f"{filing_year}.parquet"
        year_sentence_scores_path = sentence_scores_dir / f"{filing_year}.parquet"
        can_reuse = (
            year_item_features_path.exists()
            and not run_cfg.overwrite
            and (not run_cfg.write_sentence_scores or year_sentence_scores_path.exists())
        )
        if can_reuse:
            item_features_df = pl.read_parquet(year_item_features_path)
            item_feature_frames.append(item_features_df)
            year_results.append(
                {
                    "filing_year": filing_year,
                    "status": "reused_existing",
                    "source_path": str(year_path.resolve()),
                    "item_features_path": str(year_item_features_path.resolve()),
                    "sentence_scores_path": (
                        str(year_sentence_scores_path.resolve())
                        if run_cfg.write_sentence_scores and year_sentence_scores_path.exists()
                        else None
                    ),
                    "section_rows": None,
                    "sentence_rows": None,
                    "item_feature_rows": int(item_features_df.height),
                    "doc_count": int(item_features_df["doc_id"].n_unique()) if item_features_df.height else 0,
                }
            )
            continue

        if tokenizer is None:
            tokenizer = load_finbert_tokenizer(authority)
        if model is None:
            model = load_finbert_model(authority, run_cfg.runtime)
            label_mapping = resolve_finbert_label_mapping(model)

        sections_df = (
            load_eligible_section_universe(run_cfg.section_universe, year_paths=[year_path]).collect()
        )
        sections_df = _annotate_analysis_sections(sections_df)
        if sections_df.is_empty():
            item_features_df = _empty_item_features_long_frame()
            year_item_features_path.parent.mkdir(parents=True, exist_ok=True)
            item_features_df.write_parquet(year_item_features_path, compression="zstd")
            if run_cfg.write_sentence_scores:
                year_sentence_scores_path.parent.mkdir(parents=True, exist_ok=True)
                _empty_sentence_score_frame().write_parquet(year_sentence_scores_path, compression="zstd")
            item_feature_frames.append(item_features_df)
            year_results.append(
                {
                    "filing_year": filing_year,
                    "status": "processed_empty",
                    "source_path": str(year_path.resolve()),
                    "item_features_path": str(year_item_features_path.resolve()),
                    "sentence_scores_path": (
                        str(year_sentence_scores_path.resolve()) if run_cfg.write_sentence_scores else None
                    ),
                    "section_rows": 0,
                    "sentence_rows": 0,
                    "item_feature_rows": 0,
                    "doc_count": 0,
                }
            )
            continue

        sentence_scores = score_sentence_frame(
            derive_sentence_frame(sections_df, run_cfg.sentence_dataset, authority=authority),
            tokenizer,
            model,
            run_cfg.runtime,
            batch_config=run_cfg.batch_config,
            bucket_lengths=run_cfg.bucket_lengths,
        )
        item_features_df = aggregate_sentence_scores_to_item_features(sentence_scores, sections_df)

        year_item_features_path.parent.mkdir(parents=True, exist_ok=True)
        item_features_df.write_parquet(year_item_features_path, compression="zstd")
        if run_cfg.write_sentence_scores:
            year_sentence_scores_path.parent.mkdir(parents=True, exist_ok=True)
            sentence_scores.write_parquet(year_sentence_scores_path, compression="zstd")

        item_feature_frames.append(item_features_df)
        year_results.append(
            {
                "filing_year": filing_year,
                "status": "processed",
                "source_path": str(year_path.resolve()),
                "item_features_path": str(year_item_features_path.resolve()),
                "sentence_scores_path": (
                    str(year_sentence_scores_path.resolve()) if run_cfg.write_sentence_scores else None
                ),
                "section_rows": int(sections_df.height),
                "sentence_rows": int(sentence_scores.height),
                "item_feature_rows": int(item_features_df.height),
                "doc_count": int(item_features_df["doc_id"].n_unique()) if item_features_df.height else 0,
            }
        )

    item_features_long = _concat_aligned_frames(
        item_feature_frames,
        empty_schema=_empty_item_features_long_frame(),
    ).sort(["filing_year", "doc_id", "benchmark_item_code"])
    run_dir.mkdir(parents=True, exist_ok=True)
    item_features_long.write_parquet(item_features_long_path, compression="zstd")

    doc_features_wide = pivot_item_features_to_doc_wide(item_features_long)
    doc_features_wide.write_parquet(doc_features_wide_path, compression="zstd")

    coverage_summary: dict[str, int] | None = None
    if coverage_report_path is not None and run_cfg.backbone_path is not None:
        coverage_report, coverage_summary = build_coverage_report(
            item_features_long,
            _load_backbone_doc_ids(run_cfg.backbone_path),
        )
        coverage_report.write_parquet(coverage_report_path, compression="zstd")

    if run_cfg.backbone_path is None:
        warnings.append("backbone_path_not_provided")
    if label_mapping is None:
        warnings.append("model_not_loaded_all_years_reused_existing_artifacts")

    manifest = {
        "runner_name": RUNNER_NAME,
        "run_name": run_name,
        "created_at_utc": utc_timestamp(),
        "authority": asdict(authority),
        "runtime": asdict(run_cfg.runtime),
        "runtime_environment": runtime_environment,
        "batch_config": asdict(run_cfg.batch_config),
        "bucket_lengths": asdict(run_cfg.bucket_lengths),
        "sentence_dataset": asdict(run_cfg.sentence_dataset),
        "section_universe": {
            "source_items_dir": str(run_cfg.section_universe.source_items_dir.resolve()),
            "form_types": list(run_cfg.section_universe.form_types),
            "target_items": [asdict(item) for item in run_cfg.section_universe.target_items],
            "require_active_items": run_cfg.section_universe.require_active_items,
            "require_exists_by_regime": run_cfg.section_universe.require_exists_by_regime,
            "min_char_count": run_cfg.section_universe.min_char_count,
        },
        "backbone_path": str(run_cfg.backbone_path.resolve()) if run_cfg.backbone_path is not None else None,
        "year_filter": list(run_cfg.year_filter) if run_cfg.year_filter is not None else None,
        "write_sentence_scores": run_cfg.write_sentence_scores,
        "overwrite": run_cfg.overwrite,
        "note": run_cfg.note,
        "label_mapping": {str(key): value for key, value in (label_mapping or {}).items()},
        "counts": {
            "requested_year_count": len(year_paths),
            "processed_year_count": sum(1 for row in year_results if row["status"] != "reused_existing"),
            "reused_year_count": sum(1 for row in year_results if row["status"] == "reused_existing"),
            "item_feature_rows": int(item_features_long.height),
            "doc_feature_rows": int(doc_features_wide.height),
        },
        "coverage_summary": coverage_summary,
        "year_results": year_results,
        "warnings": warnings,
        "artifacts": {
            "run_dir": str(run_dir.resolve()),
            "item_features_long_path": str(item_features_long_path.resolve()),
            "doc_features_wide_path": str(doc_features_wide_path.resolve()),
            "coverage_report_path": str(coverage_report_path.resolve()) if coverage_report_path is not None else None,
            "item_features_by_year_dir": str(item_features_by_year_dir.resolve()),
            "sentence_scores_dir": (
                str(sentence_scores_dir.resolve()) if run_cfg.write_sentence_scores else None
            ),
        },
    }
    _write_json(run_manifest_path, manifest)

    return FinbertAnalysisRunArtifacts(
        run_dir=run_dir,
        run_manifest_path=run_manifest_path,
        item_features_long_path=item_features_long_path,
        doc_features_wide_path=doc_features_wide_path,
        coverage_report_path=coverage_report_path,
        sentence_scores_dir=sentence_scores_dir if run_cfg.write_sentence_scores else None,
    )
