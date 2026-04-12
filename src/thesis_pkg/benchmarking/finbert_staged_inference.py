from __future__ import annotations

from dataclasses import asdict
from importlib import metadata
import json
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import FinbertSentenceParquetInferenceRunArtifacts
from thesis_pkg.benchmarking.contracts import FinbertSentenceParquetInferenceRunConfig
from thesis_pkg.benchmarking.contracts import FinbertTokenizerProfileRunArtifacts
from thesis_pkg.benchmarking.contracts import FinbertTokenizerProfileRunConfig
from thesis_pkg.benchmarking.finbert_analysis import _empty_item_features_long_frame
from thesis_pkg.benchmarking.finbert_analysis import FINBERT_SEGMENT_POLICY_ID
from thesis_pkg.benchmarking.finbert_analysis import aggregate_sentence_scores_to_item_features
from thesis_pkg.benchmarking.finbert_analysis import build_coverage_report
from thesis_pkg.benchmarking.finbert_analysis import finbert_item_feature_contract_payload
from thesis_pkg.benchmarking.finbert_analysis import pivot_item_features_to_doc_wide
from thesis_pkg.benchmarking.finbert_benchmark import _bucket_frame
from thesis_pkg.benchmarking.finbert_benchmark import _bucket_max_length
from thesis_pkg.benchmarking.finbert_benchmark import _empty_sentence_score_frame
from thesis_pkg.benchmarking.finbert_benchmark import _median
from thesis_pkg.benchmarking.finbert_benchmark import _resolve_device
from thesis_pkg.benchmarking.finbert_benchmark import _runtime_environment
from thesis_pkg.benchmarking.finbert_benchmark import _tokenize_text_batches
from thesis_pkg.benchmarking.finbert_benchmark import load_finbert_model
from thesis_pkg.benchmarking.finbert_benchmark import resolve_finbert_label_mapping
from thesis_pkg.benchmarking.finbert_benchmark import score_sentence_frame
from thesis_pkg.benchmarking.run_logging import append_jsonl_record
from thesis_pkg.benchmarking.run_logging import utc_timestamp
from thesis_pkg.benchmarking.run_logging import write_json
from thesis_pkg.benchmarking.token_lengths import FINBERT_TOKEN_BUCKET_COLUMN
from thesis_pkg.benchmarking.token_lengths import FINBERT_TOKEN_COUNT_COLUMN
from thesis_pkg.benchmarking.token_lengths import load_finbert_tokenizer


TOKENIZER_PROFILE_RUNS = 3
TOKENIZER_PROFILE_RUNNER_NAME = "finbert_tokenizer_profile"
MODEL_INFERENCE_RUNNER_NAME = "finbert_sentence_parquet_inference"
_BUCKETS: tuple[str, ...] = ("short", "medium", "long")
_SENTENCE_METADATA_COLUMNS: tuple[str, ...] = (
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
)
_OPTIONAL_SENTENCE_METADATA_COLUMNS: tuple[str, ...] = (
    "text_scope",
    "cleaning_policy_id",
    "segment_policy_id",
)


def _empty_tokenizer_bucket_summary_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "filing_year": pl.Int32,
            "bucket": pl.Utf8,
            "sentence_rows": pl.Int64,
            "token_count_mean": pl.Float64,
            "token_count_median": pl.Float64,
            "sentence_dataset_path": pl.Utf8,
        }
    )


def _empty_tokenizer_timing_summary_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "filing_year": pl.Int32,
            "bucket": pl.Utf8,
            "sampled_rows": pl.Int64,
            "batch_size": pl.Int64,
            "max_length": pl.Int64,
            "median_seconds": pl.Float64,
            "rows_per_second": pl.Float64,
            "sentence_dataset_path": pl.Utf8,
        }
    )


def _empty_model_yearly_summary_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "filing_year": pl.Int32,
            "status": pl.Utf8,
            "sentence_dataset_path": pl.Utf8,
            "sentence_rows": pl.Int64,
            "item_feature_rows": pl.Int64,
            "doc_rows": pl.Int64,
            "item_features_path": pl.Utf8,
            "sentence_scores_path": pl.Utf8,
        }
    )


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


def _concat_aligned_frames(
    frames: list[pl.DataFrame],
    *,
    empty_schema: pl.DataFrame,
) -> pl.DataFrame:
    if not frames:
        return empty_schema
    schema = empty_schema.schema
    return pl.concat(
        [_align_frame_to_schema(frame, schema) for frame in frames],
        how="vertical_relaxed",
    )


def _resolve_sentence_year_paths(sentence_dataset_dir: Path) -> list[Path]:
    year_paths = sorted(
        path
        for path in Path(sentence_dataset_dir).glob("*.parquet")
        if path.stem.isdigit() and len(path.stem) == 4
    )
    if not year_paths:
        raise FileNotFoundError(f"No year parquet files found in {sentence_dataset_dir}")
    return year_paths


def _select_sentence_year_paths(
    sentence_dataset_dir: Path,
    year_filter: tuple[int, ...] | None,
) -> list[Path]:
    year_paths = _resolve_sentence_year_paths(sentence_dataset_dir)
    if year_filter is None:
        return year_paths

    target_years = set(year_filter)
    selected = [path for path in year_paths if int(path.stem) in target_years]
    missing_years = sorted(target_years - {int(path.stem) for path in selected})
    if missing_years:
        raise FileNotFoundError(
            f"Requested filing years were not found in {sentence_dataset_dir}: {missing_years}"
        )
    return selected


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _sentence_dataset_manifest_payload(sentence_dataset_dir: Path) -> dict[str, Any] | None:
    manifest_path = sentence_dataset_dir.parent.parent / "run_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {
        "manifest_path": str(manifest_path.resolve()),
        "cleaning_policy_id": manifest.get("cleaning_policy_id"),
        "cleaning": manifest.get("cleaning"),
        "segment_policy_id": manifest.get("segment_policy_id"),
        "authority": manifest.get("authority"),
        "sentence_dataset": manifest.get("sentence_dataset"),
    }
    return json.loads(json.dumps(payload))


def _semantic_inference_payload(
    run_cfg: FinbertSentenceParquetInferenceRunConfig,
    authority: FinbertAuthoritySpec,
) -> dict[str, Any]:
    payload = {
        "authority": asdict(authority),
        "runtime": asdict(run_cfg.runtime),
        "batch_config": asdict(run_cfg.batch_config),
        "bucket_lengths": asdict(run_cfg.bucket_lengths),
        "sentence_dataset_dir": str(run_cfg.sentence_dataset_dir.resolve()),
        "backbone_path": str(run_cfg.backbone_path.resolve()) if run_cfg.backbone_path is not None else None,
        "year_filter": list(run_cfg.year_filter) if run_cfg.year_filter is not None else None,
        "sentence_slice_rows": run_cfg.sentence_slice_rows,
        "write_sentence_scores": run_cfg.write_sentence_scores,
        "source_sentence_dataset_manifest": _sentence_dataset_manifest_payload(run_cfg.sentence_dataset_dir),
    }
    return json.loads(json.dumps(payload))


def _assert_existing_inference_run_compatible(
    run_manifest_path: Path,
    run_cfg: FinbertSentenceParquetInferenceRunConfig,
    authority: FinbertAuthoritySpec,
) -> None:
    if run_cfg.overwrite or not run_manifest_path.exists():
        return
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected = _semantic_inference_payload(run_cfg, authority)
    existing = manifest.get("semantic_reuse_guard", {})
    mismatched = [
        key
        for key, expected_value in expected.items()
        if existing.get(key) != expected_value
    ]
    if mismatched:
        raise ValueError(
            "Existing FinBERT inference artifacts were created with incompatible semantic settings "
            f"for {mismatched}. Use a new run_name or set overwrite=True."
        )


def _validate_sentence_frame_columns(
    sentence_df: pl.DataFrame,
    *,
    required_columns: tuple[str, ...],
    source_path: Path,
) -> None:
    missing = sorted(set(required_columns) - set(sentence_df.columns))
    if missing:
        raise ValueError(
            f"Sentence dataset {source_path} is missing required columns: {missing}"
        )


def _sample_bucket_frame(
    bucket_df: pl.DataFrame,
    *,
    row_cap: int,
    seed: int,
) -> pl.DataFrame:
    if bucket_df.height <= row_cap:
        return bucket_df
    return bucket_df.sample(n=row_cap, seed=seed, shuffle=False)


def _build_bucket_summary(
    sentence_df: pl.DataFrame,
    *,
    filing_year: int,
    source_path: Path,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket in _BUCKETS:
        bucket_df = _bucket_frame(sentence_df, bucket)
        token_mean = (
            float(bucket_df[FINBERT_TOKEN_COUNT_COLUMN].mean())
            if bucket_df.height
            else None
        )
        token_median = (
            float(bucket_df[FINBERT_TOKEN_COUNT_COLUMN].median())
            if bucket_df.height
            else None
        )
        rows.append(
            {
                "filing_year": filing_year,
                "bucket": bucket,
                "sentence_rows": int(bucket_df.height),
                "token_count_mean": token_mean,
                "token_count_median": token_median,
                "sentence_dataset_path": str(source_path.resolve()),
            }
        )
    return pl.DataFrame(rows).select(_empty_tokenizer_bucket_summary_frame().columns)


def _profile_tokenizer_runtime_for_year(
    sentence_df: pl.DataFrame,
    *,
    filing_year: int,
    source_path: Path,
    run_cfg: FinbertTokenizerProfileRunConfig,
    authority: FinbertAuthoritySpec,
) -> pl.DataFrame:
    tokenizer = load_finbert_tokenizer(authority)
    rows: list[dict[str, Any]] = []
    for bucket in _BUCKETS:
        bucket_df = _bucket_frame(sentence_df, bucket)
        sampled_df = _sample_bucket_frame(
            bucket_df,
            row_cap=run_cfg.profile_row_cap_per_bucket,
            seed=run_cfg.sample_seed + filing_year,
        )
        batch_size = run_cfg.batch_config.batch_size_for_bucket(bucket)
        max_length = _bucket_max_length(bucket, run_cfg.bucket_lengths)
        texts = sampled_df["sentence_text"].to_list() if sampled_df.height else []
        if not texts:
            rows.append(
                {
                    "filing_year": filing_year,
                    "bucket": bucket,
                    "sampled_rows": 0,
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "median_seconds": 0.0,
                    "rows_per_second": None,
                    "sentence_dataset_path": str(source_path.resolve()),
                }
            )
            continue

        tokenizer(
            texts[: min(len(texts), batch_size)],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        timings: list[float] = []
        import time

        for _ in range(TOKENIZER_PROFILE_RUNS):
            start = time.perf_counter()
            _tokenize_text_batches(
                texts,
                tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                return_tensors="pt",
            )
            timings.append(time.perf_counter() - start)

        median_seconds = _median(timings)
        rows.append(
            {
                "filing_year": filing_year,
                "bucket": bucket,
                "sampled_rows": int(sampled_df.height),
                "batch_size": batch_size,
                "max_length": max_length,
                "median_seconds": median_seconds,
                "rows_per_second": (
                    float(sampled_df.height / median_seconds)
                    if median_seconds
                    else None
                ),
                "sentence_dataset_path": str(source_path.resolve()),
            }
        )
    return pl.DataFrame(rows).select(_empty_tokenizer_timing_summary_frame().columns)


def _load_backbone_doc_ids(path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select(pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"))
        .collect()
    )


def _section_metadata_from_sentence_frame(sentence_df: pl.DataFrame) -> pl.DataFrame:
    if sentence_df.is_empty():
        return _empty_item_features_long_frame().select(_SENTENCE_METADATA_COLUMNS)

    selected_columns = [
        *_SENTENCE_METADATA_COLUMNS,
        *[column for column in _OPTIONAL_SENTENCE_METADATA_COLUMNS if column in sentence_df.columns],
    ]
    return (
        sentence_df.select(selected_columns)
        .unique(subset=["benchmark_row_id"], maintain_order=True)
        .sort(["filing_year", "doc_id", "benchmark_item_code"])
    )


def _score_sentence_frame_in_slices(
    sentence_df: pl.DataFrame,
    tokenizer,
    model,
    run_cfg: FinbertSentenceParquetInferenceRunConfig,
) -> pl.DataFrame:
    if sentence_df.is_empty():
        return _empty_sentence_score_frame()

    slice_size = max(run_cfg.sentence_slice_rows, 1)
    chunks: list[pl.DataFrame] = []
    for bucket in _BUCKETS:
        bucket_df = _bucket_frame(sentence_df, bucket)
        if bucket_df.is_empty():
            continue
        for slice_df in bucket_df.iter_slices(n_rows=slice_size):
            if slice_df.is_empty():
                continue
            chunks.append(
                score_sentence_frame(
                    slice_df,
                    tokenizer,
                    model,
                    run_cfg.runtime,
                    batch_config=run_cfg.batch_config,
                    bucket_lengths=run_cfg.bucket_lengths,
                )
            )

    return _concat_aligned_frames(
        chunks,
        empty_schema=_empty_sentence_score_frame(),
    ).sort(["filing_year", "benchmark_row_id", "sentence_index"])


def run_finbert_tokenizer_profile(
    run_cfg: FinbertTokenizerProfileRunConfig,
    *,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
) -> FinbertTokenizerProfileRunArtifacts:
    year_paths = _select_sentence_year_paths(run_cfg.sentence_dataset_dir, run_cfg.year_filter)
    run_name = run_cfg.run_name or f"{TOKENIZER_PROFILE_RUNNER_NAME}_{utc_timestamp().replace(':', '')}"
    run_dir = run_cfg.out_root / run_name
    bucket_summary_by_year_dir = run_dir / "tokenizer_profile" / "by_year"
    records_path = run_dir / "tokenizer_profile_records.jsonl"
    bucket_summary_path = run_dir / "tokenizer_profile_bucket_summary.parquet"
    bucket_summary_csv_path = run_dir / "tokenizer_profile_bucket_summary.csv"
    timing_summary_path = run_dir / "tokenizer_profile_timing_summary.parquet"
    timing_summary_csv_path = run_dir / "tokenizer_profile_timing_summary.csv"
    run_manifest_path = run_dir / "run_manifest.json"

    bucket_frames: list[pl.DataFrame] = []
    timing_frames: list[pl.DataFrame] = []
    year_results: list[dict[str, Any]] = []
    run_dir.mkdir(parents=True, exist_ok=True)

    for year_path in year_paths:
        filing_year = int(year_path.stem)
        year_bucket_summary_path = bucket_summary_by_year_dir / f"{filing_year}_bucket_summary.parquet"
        year_timing_summary_path = bucket_summary_by_year_dir / f"{filing_year}_timing_summary.parquet"
        if (
            year_bucket_summary_path.exists()
            and year_timing_summary_path.exists()
            and not run_cfg.overwrite
        ):
            bucket_df = pl.read_parquet(year_bucket_summary_path)
            timing_df = pl.read_parquet(year_timing_summary_path)
            bucket_frames.append(bucket_df)
            timing_frames.append(timing_df)
            year_results.append(
                {
                    "filing_year": filing_year,
                    "status": "reused_existing",
                    "sentence_dataset_path": str(year_path.resolve()),
                    "bucket_summary_path": str(year_bucket_summary_path.resolve()),
                    "timing_summary_path": str(year_timing_summary_path.resolve()),
                }
            )
            continue

        sentence_df = pl.read_parquet(year_path)
        _validate_sentence_frame_columns(
            sentence_df,
            required_columns=(
                "sentence_text",
                FINBERT_TOKEN_COUNT_COLUMN,
                FINBERT_TOKEN_BUCKET_COLUMN,
                "filing_year",
            ),
            source_path=year_path,
        )
        bucket_df = _build_bucket_summary(
            sentence_df,
            filing_year=filing_year,
            source_path=year_path,
        )
        timing_df = _profile_tokenizer_runtime_for_year(
            sentence_df,
            filing_year=filing_year,
            source_path=year_path,
            run_cfg=run_cfg,
            authority=authority,
        )
        year_bucket_summary_path.parent.mkdir(parents=True, exist_ok=True)
        bucket_df.write_parquet(year_bucket_summary_path, compression="zstd")
        timing_df.write_parquet(year_timing_summary_path, compression="zstd")
        bucket_frames.append(bucket_df)
        timing_frames.append(timing_df)
        year_results.append(
            {
                "filing_year": filing_year,
                "status": "processed",
                "sentence_dataset_path": str(year_path.resolve()),
                "bucket_summary_path": str(year_bucket_summary_path.resolve()),
                "timing_summary_path": str(year_timing_summary_path.resolve()),
            }
        )

        for row in bucket_df.to_dicts():
            append_jsonl_record(
                records_path,
                {
                    "created_at_utc": utc_timestamp(),
                    "record_type": "bucket_summary",
                    **row,
                },
            )
        for row in timing_df.to_dicts():
            append_jsonl_record(
                records_path,
                {
                    "created_at_utc": utc_timestamp(),
                    "record_type": "timing_summary",
                    **row,
                },
            )

    bucket_summary_df = _concat_aligned_frames(
        bucket_frames,
        empty_schema=_empty_tokenizer_bucket_summary_frame(),
    )
    timing_summary_df = _concat_aligned_frames(
        timing_frames,
        empty_schema=_empty_tokenizer_timing_summary_frame(),
    )
    bucket_summary_df.write_parquet(bucket_summary_path, compression="zstd")
    bucket_summary_df.write_csv(bucket_summary_csv_path)
    timing_summary_df.write_parquet(timing_summary_path, compression="zstd")
    timing_summary_df.write_csv(timing_summary_csv_path)

    write_json(
        run_manifest_path,
        {
            "runner_name": TOKENIZER_PROFILE_RUNNER_NAME,
            "run_name": run_name,
            "created_at_utc": utc_timestamp(),
            "authority": asdict(authority),
            "sentence_dataset_dir": str(run_cfg.sentence_dataset_dir.resolve()),
            "batch_config": asdict(run_cfg.batch_config),
            "bucket_lengths": asdict(run_cfg.bucket_lengths),
            "year_filter": list(run_cfg.year_filter) if run_cfg.year_filter is not None else None,
            "profile_row_cap_per_bucket": run_cfg.profile_row_cap_per_bucket,
            "sample_seed": run_cfg.sample_seed,
            "tokenizer_profile_runs": TOKENIZER_PROFILE_RUNS,
            "overwrite": run_cfg.overwrite,
            "note": run_cfg.note,
            "counts": {
                "requested_year_count": len(year_paths),
                "processed_year_count": sum(
                    1 for row in year_results if row["status"] == "processed"
                ),
                "reused_year_count": sum(
                    1 for row in year_results if row["status"] == "reused_existing"
                ),
                "sentence_rows": int(bucket_summary_df["sentence_rows"].sum()) if bucket_summary_df.height else 0,
                "profiled_rows": int(timing_summary_df["sampled_rows"].sum()) if timing_summary_df.height else 0,
            },
            "year_results": year_results,
            "artifacts": {
                "run_dir": str(run_dir.resolve()),
                "records_path": str(records_path.resolve()),
                "bucket_summary_path": str(bucket_summary_path.resolve()),
                "bucket_summary_csv_path": str(bucket_summary_csv_path.resolve()),
                "timing_summary_path": str(timing_summary_path.resolve()),
                "timing_summary_csv_path": str(timing_summary_csv_path.resolve()),
                "by_year_dir": str(bucket_summary_by_year_dir.resolve()),
            },
        },
    )

    return FinbertTokenizerProfileRunArtifacts(
        run_dir=run_dir,
        run_manifest_path=run_manifest_path,
        bucket_summary_path=bucket_summary_path,
        timing_summary_path=timing_summary_path,
    )


def run_finbert_sentence_parquet_inference(
    run_cfg: FinbertSentenceParquetInferenceRunConfig,
    *,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
) -> FinbertSentenceParquetInferenceRunArtifacts:
    year_paths = _select_sentence_year_paths(run_cfg.sentence_dataset_dir, run_cfg.year_filter)
    run_name = run_cfg.run_name or f"{MODEL_INFERENCE_RUNNER_NAME}_{utc_timestamp().replace(':', '')}"
    run_dir = run_cfg.out_root / run_name
    item_features_by_year_dir = run_dir / "item_features" / "by_year"
    sentence_scores_dir = run_dir / "sentence_scores" / "by_year"
    item_features_long_path = run_dir / "item_features_long.parquet"
    doc_features_wide_path = run_dir / "doc_features_wide.parquet"
    coverage_report_path = run_dir / "coverage_report.parquet" if run_cfg.backbone_path is not None else None
    yearly_summary_path = run_dir / "model_inference_yearly_summary.parquet"
    yearly_summary_csv_path = run_dir / "model_inference_yearly_summary.csv"
    run_manifest_path = run_dir / "run_manifest.json"
    source_sentence_manifest = _sentence_dataset_manifest_payload(run_cfg.sentence_dataset_dir)
    source_segment_policy_id = (
        source_sentence_manifest.get("segment_policy_id")
        if source_sentence_manifest is not None
        else None
    )
    _assert_existing_inference_run_compatible(run_manifest_path, run_cfg, authority)

    tokenizer = None
    model = None
    label_mapping: dict[int, str] | None = None
    warnings: list[str] = []
    year_results: list[dict[str, Any]] = []
    item_feature_frames: list[pl.DataFrame] = []

    resolved_device = _resolve_device(run_cfg.runtime)
    runtime_environment = _runtime_environment(run_cfg.runtime, resolved_device)

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
            sentence_rows = int(
                pl.scan_parquet(year_path).select(pl.len()).collect().item()
            )
            year_results.append(
                {
                    "filing_year": filing_year,
                    "status": "reused_existing",
                    "sentence_dataset_path": str(year_path.resolve()),
                    "sentence_rows": sentence_rows,
                    "item_feature_rows": int(item_features_df.height),
                    "doc_rows": int(item_features_df["doc_id"].n_unique()) if item_features_df.height else 0,
                    "item_features_path": str(year_item_features_path.resolve()),
                    "sentence_scores_path": (
                        str(year_sentence_scores_path.resolve())
                        if run_cfg.write_sentence_scores and year_sentence_scores_path.exists()
                        else None
                    ),
                }
            )
            continue

        if tokenizer is None:
            tokenizer = load_finbert_tokenizer(authority)
        if model is None:
            model = load_finbert_model(authority, run_cfg.runtime)
            label_mapping = resolve_finbert_label_mapping(model)

        sentence_df = pl.read_parquet(year_path)
        _validate_sentence_frame_columns(
            sentence_df,
            required_columns=(
                "benchmark_sentence_id",
                "benchmark_row_id",
                "doc_id",
                "cik_10",
                "accession_nodash",
                "filing_date",
                "filing_year",
                "benchmark_item_code",
                "benchmark_item_label",
                "source_year_file",
                "document_type",
                "document_type_raw",
                "document_type_normalized",
                "sentence_index",
                "sentence_text",
                "sentence_char_count",
                "sentencizer_backend",
                "sentencizer_version",
                FINBERT_TOKEN_COUNT_COLUMN,
                FINBERT_TOKEN_BUCKET_COLUMN,
            ),
            source_path=year_path,
        )

        if sentence_df.is_empty():
            sentence_scores_df = _empty_sentence_score_frame()
            item_features_df = _empty_item_features_long_frame()
        else:
            sentence_scores_df = _score_sentence_frame_in_slices(
                sentence_df,
                tokenizer,
                model,
                run_cfg,
            )
            item_features_df = aggregate_sentence_scores_to_item_features(
                sentence_scores_df,
                _section_metadata_from_sentence_frame(sentence_df),
            ).with_columns(
                [
                    pl.lit(authority.model_name, dtype=pl.Utf8).alias("model_name"),
                    pl.lit(authority.model_revision, dtype=pl.Utf8).alias("model_version"),
                    pl.coalesce(
                        [
                            pl.col("segment_policy_id").cast(pl.Utf8, strict=False),
                            pl.lit(FINBERT_SEGMENT_POLICY_ID, dtype=pl.Utf8),
                        ]
                    ).alias("segment_policy_id"),
                ]
            )

        year_item_features_path.parent.mkdir(parents=True, exist_ok=True)
        item_features_df.write_parquet(year_item_features_path, compression="zstd")
        if run_cfg.write_sentence_scores:
            year_sentence_scores_path.parent.mkdir(parents=True, exist_ok=True)
            sentence_scores_df.write_parquet(year_sentence_scores_path, compression="zstd")

        item_feature_frames.append(item_features_df)
        year_results.append(
            {
                "filing_year": filing_year,
                "status": "processed" if sentence_df.height else "processed_empty",
                "sentence_dataset_path": str(year_path.resolve()),
                "sentence_rows": int(sentence_df.height),
                "item_feature_rows": int(item_features_df.height),
                "doc_rows": int(item_features_df["doc_id"].n_unique()) if item_features_df.height else 0,
                "item_features_path": str(year_item_features_path.resolve()),
                "sentence_scores_path": (
                    str(year_sentence_scores_path.resolve()) if run_cfg.write_sentence_scores else None
                ),
            }
        )

    yearly_summary_df = _concat_aligned_frames(
        [pl.DataFrame([row]) for row in year_results],
        empty_schema=_empty_model_yearly_summary_frame(),
    ).sort("filing_year")

    run_dir.mkdir(parents=True, exist_ok=True)
    yearly_summary_df.write_parquet(yearly_summary_path, compression="zstd")
    yearly_summary_df.write_csv(yearly_summary_csv_path)

    item_features_long = _concat_aligned_frames(
        item_feature_frames,
        empty_schema=_empty_item_features_long_frame(),
    ).sort(["filing_year", "doc_id", "benchmark_item_code"])
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
    elif coverage_report_path is None:
        warnings.append("backbone_path_not_provided")

    if label_mapping is None:
        warnings.append("model_not_loaded_all_years_reused_existing_artifacts")

    write_json(
        run_manifest_path,
        {
            "runner_name": MODEL_INFERENCE_RUNNER_NAME,
            "run_name": run_name,
            "created_at_utc": utc_timestamp(),
            "authority": asdict(authority),
            "semantic_reuse_guard": _semantic_inference_payload(run_cfg, authority),
            "item_feature_contract": finbert_item_feature_contract_payload(
                authority,
                segment_policy_id=source_segment_policy_id or FINBERT_SEGMENT_POLICY_ID,
            ),
            "runtime": asdict(run_cfg.runtime),
            "runtime_environment": runtime_environment,
            "transformers_version": _package_version("transformers"),
            "torch_version": runtime_environment.get("torch_version"),
            "source_sentence_dataset_manifest": source_sentence_manifest,
            "batch_config": asdict(run_cfg.batch_config),
            "bucket_lengths": asdict(run_cfg.bucket_lengths),
            "sentence_dataset_dir": str(run_cfg.sentence_dataset_dir.resolve()),
            "backbone_path": str(run_cfg.backbone_path.resolve()) if run_cfg.backbone_path is not None else None,
            "year_filter": list(run_cfg.year_filter) if run_cfg.year_filter is not None else None,
            "sentence_slice_rows": run_cfg.sentence_slice_rows,
            "write_sentence_scores": run_cfg.write_sentence_scores,
            "overwrite": run_cfg.overwrite,
            "note": run_cfg.note,
            "label_mapping": {str(key): value for key, value in (label_mapping or {}).items()},
            "counts": {
                "requested_year_count": len(year_paths),
                "processed_year_count": sum(
                    1 for row in year_results if row["status"] != "reused_existing"
                ),
                "reused_year_count": sum(
                    1 for row in year_results if row["status"] == "reused_existing"
                ),
                "item_feature_rows": int(item_features_long.height),
                "doc_feature_rows": int(doc_features_wide.height),
            },
            "coverage_summary": coverage_summary,
            "year_results": year_results,
            "warnings": warnings,
            "artifacts": {
                "run_dir": str(run_dir.resolve()),
                "item_features_by_year_dir": str(item_features_by_year_dir.resolve()),
                "item_features_long_path": str(item_features_long_path.resolve()),
                "doc_features_wide_path": str(doc_features_wide_path.resolve()),
                "coverage_report_path": str(coverage_report_path.resolve()) if coverage_report_path is not None else None,
                "sentence_scores_dir": (
                    str(sentence_scores_dir.resolve()) if run_cfg.write_sentence_scores else None
                ),
                "yearly_summary_path": str(yearly_summary_path.resolve()),
                "yearly_summary_csv_path": str(yearly_summary_csv_path.resolve()),
            },
        },
    )

    return FinbertSentenceParquetInferenceRunArtifacts(
        run_dir=run_dir,
        run_manifest_path=run_manifest_path,
        item_features_long_path=item_features_long_path,
        doc_features_wide_path=doc_features_wide_path,
        coverage_report_path=coverage_report_path,
        sentence_scores_dir=sentence_scores_dir if run_cfg.write_sentence_scores else None,
        yearly_summary_path=yearly_summary_path,
    )
