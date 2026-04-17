from __future__ import annotations

from dataclasses import asdict
from importlib import metadata
import json
from pathlib import Path
import shutil
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
from thesis_pkg.benchmarking.manifest_contracts import json_sha256
from thesis_pkg.benchmarking.manifest_contracts import MANIFEST_PATH_SEMANTICS_RELATIVE
from thesis_pkg.benchmarking.manifest_contracts import make_semantic_reuse_guard
from thesis_pkg.benchmarking.manifest_contracts import normalize_contract_path
from thesis_pkg.benchmarking.manifest_contracts import relative_artifact_path
from thesis_pkg.benchmarking.manifest_contracts import semantic_file_fingerprint
from thesis_pkg.benchmarking.manifest_contracts import semantic_guard_mismatches
from thesis_pkg.benchmarking.manifest_contracts import stable_string_fingerprint
from thesis_pkg.benchmarking.manifest_contracts import write_manifest_path_value
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
_STREAMING_PARQUET_COMPRESSION = "zstd"


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


def _empty_frame(schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _scan_aligned_frame(path: Path, schema: pl.Schema) -> pl.LazyFrame:
    lf = pl.scan_parquet(path)
    schema_names = set(lf.collect_schema().names())
    return lf.select(
        [
            (
                pl.col(name).cast(dtype, strict=False).alias(name)
                if name in schema_names
                else pl.lit(None, dtype=dtype).alias(name)
            )
            for name, dtype in schema.items()
        ]
    )


def _sink_parquet_from_paths(
    paths: list[Path],
    *,
    output_path: Path,
    schema: pl.Schema,
    compression: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    if not paths:
        _empty_frame(schema).write_parquet(output_path, compression=compression)
        return
    if len(paths) == 1:
        _align_frame_to_schema(pl.read_parquet(paths[0]), schema).write_parquet(
            output_path,
            compression=compression,
        )
        return
    pl.concat(
        [_scan_aligned_frame(path, schema) for path in paths],
        how="vertical_relaxed",
    ).sink_parquet(output_path, compression=compression)


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
        "cleaning_policy_id": manifest.get("cleaning_policy_id"),
        "cleaning": manifest.get("cleaning"),
        "segment_policy_id": manifest.get("segment_policy_id"),
        "authority": manifest.get("authority"),
        "sentence_dataset": manifest.get("sentence_dataset"),
        "accepted_universe_contract": manifest.get("accepted_universe_contract"),
        "year_filter": manifest.get("year_filter"),
    }
    return json.loads(json.dumps(payload))


def _sentence_dataset_manifest_path(sentence_dataset_dir: Path) -> Path:
    return sentence_dataset_dir.parent.parent / "run_manifest.json"


def _sentence_year_fingerprints(year_paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        path.stem: semantic_file_fingerprint(path)
        for path in year_paths
    }


def _semantic_inference_payload(
    run_cfg: FinbertSentenceParquetInferenceRunConfig,
    authority: FinbertAuthoritySpec,
) -> dict[str, Any]:
    payload = {
        "authority": asdict(authority),
        "runtime": asdict(run_cfg.runtime),
        "batch_config": asdict(run_cfg.batch_config),
        "bucket_lengths": asdict(run_cfg.bucket_lengths),
        "year_filter": list(run_cfg.year_filter) if run_cfg.year_filter is not None else None,
        "sentence_slice_rows": run_cfg.sentence_slice_rows,
        "write_sentence_scores": run_cfg.write_sentence_scores,
        "source_sentence_dataset_manifest": _sentence_dataset_manifest_payload(run_cfg.sentence_dataset_dir),
    }
    return json.loads(json.dumps(payload))


def _semantic_inference_guard(
    run_cfg: FinbertSentenceParquetInferenceRunConfig,
    authority: FinbertAuthoritySpec,
    *,
    year_paths: list[Path],
) -> dict[str, Any]:
    source_manifest_payload = _sentence_dataset_manifest_payload(run_cfg.sentence_dataset_dir)
    return make_semantic_reuse_guard(
        version="sentence_parquet_inference_v3",
        payload=_semantic_inference_payload(run_cfg, authority),
        fingerprints={
            "source_sentence_dataset_manifest_fingerprint": (
                json_sha256(source_manifest_payload)
                if source_manifest_payload is not None
                else None
            ),
            "sentence_dataset_by_year": _sentence_year_fingerprints(year_paths),
        },
    )


def _assert_existing_inference_run_compatible(
    run_manifest_path: Path,
    run_cfg: FinbertSentenceParquetInferenceRunConfig,
    authority: FinbertAuthoritySpec,
    *,
    year_paths: list[Path],
) -> None:
    if run_cfg.overwrite or not run_manifest_path.exists():
        return
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected = _semantic_inference_guard(run_cfg, authority, year_paths=year_paths)
    existing = manifest.get("semantic_reuse_guard", {})
    mismatched = semantic_guard_mismatches(existing, expected)
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


def _validate_sentence_frame_schema(
    sentence_schema: pl.Schema,
    *,
    required_columns: tuple[str, ...],
    source_path: Path,
) -> None:
    missing = sorted(set(required_columns) - set(sentence_schema.names()))
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
    del source_path
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
                "sentence_dataset_path": None,
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
    del source_path
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
                    "sentence_dataset_path": None,
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
                "sentence_dataset_path": None,
            }
        )
    return pl.DataFrame(rows).select(_empty_tokenizer_timing_summary_frame().columns)


def _load_backbone_doc_ids(path: Path) -> pl.DataFrame:
    return _load_backbone_doc_ids_for_years(path, year_filter=None)


def _load_backbone_doc_ids_for_years(
    path: Path,
    *,
    year_filter: tuple[int, ...] | None,
) -> pl.DataFrame:
    lf = pl.scan_parquet(path).select(pl.all())
    schema = lf.collect_schema()
    if "doc_id" not in schema:
        raise ValueError("Backbone parquet must contain a doc_id column for coverage reporting.")
    if year_filter is not None:
        if "filing_date" in schema:
            lf = lf.filter(pl.col("filing_date").cast(pl.Date, strict=False).dt.year().is_in(year_filter))
        elif "filing_year" in schema:
            lf = lf.filter(pl.col("filing_year").cast(pl.Int32, strict=False).is_in(year_filter))
        else:
            raise ValueError(
                "Backbone parquet must contain filing_date or filing_year when year_filter is provided."
            )
    return lf.select(pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id")).collect()


def _effective_year_filter(
    run_year_filter: tuple[int, ...] | None,
    source_sentence_manifest: dict[str, Any] | None,
) -> tuple[int, ...] | None:
    if run_year_filter is not None:
        return run_year_filter
    source_years = source_sentence_manifest.get("year_filter") if source_sentence_manifest else None
    if source_years is None:
        return None
    return tuple(int(year) for year in source_years)


def _build_backbone_contract(
    *,
    backbone_path: Path | None,
    effective_year_filter: tuple[int, ...] | None,
    source_sentence_manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if backbone_path is None:
        return None
    backbone_df = _load_backbone_doc_ids_for_years(backbone_path, year_filter=effective_year_filter)
    accepted_universe_contract = (
        source_sentence_manifest.get("accepted_universe_contract")
        if source_sentence_manifest is not None
        else None
    )
    return {
        "contract_version": "backbone_contract_v2",
        "normalized_relative_backbone_path": normalize_contract_path(backbone_path, base_path=Path.cwd()),
        "effective_year_filter": list(effective_year_filter) if effective_year_filter is not None else None,
        "filtered_backbone_doc_count": int(backbone_df["doc_id"].n_unique()) if backbone_df.height else 0,
        "filtered_backbone_doc_universe_fingerprint": stable_string_fingerprint(
            backbone_df["doc_id"].to_list() if backbone_df.height else []
        ),
        "accepted_universe_contract_fingerprint": (
            json_sha256(accepted_universe_contract)
            if accepted_universe_contract is not None
            else None
        ),
    }


def _semantic_tokenizer_payload(
    run_cfg: FinbertTokenizerProfileRunConfig,
    authority: FinbertAuthoritySpec,
) -> dict[str, Any]:
    payload = {
        "authority": asdict(authority),
        "batch_config": asdict(run_cfg.batch_config),
        "bucket_lengths": asdict(run_cfg.bucket_lengths),
        "profile_row_cap_per_bucket": run_cfg.profile_row_cap_per_bucket,
        "sample_seed": run_cfg.sample_seed,
        "year_filter": list(run_cfg.year_filter) if run_cfg.year_filter is not None else None,
    }
    return json.loads(json.dumps(payload))


def _semantic_tokenizer_guard(
    run_cfg: FinbertTokenizerProfileRunConfig,
    authority: FinbertAuthoritySpec,
    *,
    year_paths: list[Path],
) -> dict[str, Any]:
    source_manifest_payload = _sentence_dataset_manifest_payload(run_cfg.sentence_dataset_dir)
    return make_semantic_reuse_guard(
        version="tokenizer_profile_v3",
        payload=_semantic_tokenizer_payload(run_cfg, authority),
        fingerprints={
            "source_sentence_dataset_manifest_fingerprint": (
                json_sha256(source_manifest_payload)
                if source_manifest_payload is not None
                else None
            ),
            "sentence_dataset_by_year": _sentence_year_fingerprints(year_paths),
        },
    )


def _assert_existing_tokenizer_run_compatible(
    run_manifest_path: Path,
    run_cfg: FinbertTokenizerProfileRunConfig,
    authority: FinbertAuthoritySpec,
    *,
    year_paths: list[Path],
) -> None:
    if run_cfg.overwrite or not run_manifest_path.exists():
        return
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected = _semantic_tokenizer_guard(run_cfg, authority, year_paths=year_paths)
    existing = manifest.get("semantic_reuse_guard", {})
    mismatched = semantic_guard_mismatches(existing, expected)
    if mismatched:
        raise ValueError(
            "Existing FinBERT tokenizer-profile artifacts were created with incompatible semantic settings "
            f"for {mismatched}. Use a new run_name or set overwrite=True."
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


def _lazy_section_metadata_from_sentence_dataset(sentence_path: Path) -> pl.LazyFrame:
    sentence_lf = pl.scan_parquet(sentence_path)
    schema_names = set(sentence_lf.collect_schema().names())
    selected_columns = [
        pl.col("benchmark_row_id").cast(pl.Utf8, strict=False).alias("benchmark_row_id"),
        pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
        pl.col("cik_10").cast(pl.Utf8, strict=False).alias("cik_10"),
        pl.col("accession_nodash").cast(pl.Utf8, strict=False).alias("accession_nodash"),
        pl.col("filing_date").cast(pl.Date, strict=False).alias("filing_date"),
        pl.col("filing_year").cast(pl.Int32, strict=False).alias("filing_year"),
        pl.col("source_year_file").cast(pl.Int32, strict=False).alias("source_year_file"),
        pl.col("document_type").cast(pl.Utf8, strict=False).alias("document_type"),
        pl.col("document_type_raw").cast(pl.Utf8, strict=False).alias("document_type_raw"),
        pl.col("document_type_normalized").cast(pl.Utf8, strict=False).alias("document_type_normalized"),
        pl.col("benchmark_item_code").cast(pl.Utf8, strict=False).alias("benchmark_item_code"),
        pl.col("benchmark_item_label").cast(pl.Utf8, strict=False).alias("benchmark_item_label"),
        (
            pl.col("text_scope").cast(pl.Utf8, strict=False)
            if "text_scope" in schema_names
            else pl.col("benchmark_item_code").cast(pl.Utf8, strict=False)
        ).alias("text_scope"),
        (
            pl.col("cleaning_policy_id").cast(pl.Utf8, strict=False)
            if "cleaning_policy_id" in schema_names
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("cleaning_policy_id"),
        (
            pl.col("segment_policy_id").cast(pl.Utf8, strict=False)
            if "segment_policy_id" in schema_names
            else pl.lit(FINBERT_SEGMENT_POLICY_ID, dtype=pl.Utf8)
        ).alias("segment_policy_id"),
    ]
    return (
        sentence_lf.select(selected_columns)
        .unique(subset=["benchmark_row_id"], keep="first")
        .sort(["filing_year", "doc_id", "benchmark_item_code"])
    )


def _lazy_item_features_from_sentence_score_paths(
    score_paths: list[Path],
    *,
    sentence_path: Path,
    authority: FinbertAuthoritySpec,
) -> pl.LazyFrame:
    metadata_lf = _lazy_section_metadata_from_sentence_dataset(sentence_path)
    if not score_paths:
        return (
            metadata_lf.with_columns(
                [
                    pl.lit(0, dtype=pl.Int32).alias("sentence_count"),
                    pl.lit(None, dtype=pl.Float64).alias("negative_prob_mean"),
                    pl.lit(None, dtype=pl.Float64).alias("neutral_prob_mean"),
                    pl.lit(None, dtype=pl.Float64).alias("positive_prob_mean"),
                    pl.lit(None, dtype=pl.Float64).alias("argmax_share_negative"),
                    pl.lit(None, dtype=pl.Float64).alias("argmax_share_neutral"),
                    pl.lit(None, dtype=pl.Float64).alias("argmax_share_positive"),
                    pl.lit(None, dtype=pl.Float64).alias("sentiment_balance_mean"),
                    pl.lit(0, dtype=pl.Int32).alias("finbert_segment_count"),
                    pl.lit(0, dtype=pl.Int64).alias("finbert_token_count_512_sum"),
                    pl.lit(None, dtype=pl.Float64).alias("finbert_neg_prob_lenw_mean"),
                    pl.lit(None, dtype=pl.Float64).alias("finbert_pos_prob_lenw_mean"),
                    pl.lit(None, dtype=pl.Float64).alias("finbert_neu_prob_lenw_mean"),
                    pl.lit(None, dtype=pl.Float64).alias("finbert_net_negative_lenw_mean"),
                    pl.lit(None, dtype=pl.Float64).alias("finbert_neg_dominant_share"),
                    pl.lit(authority.model_name, dtype=pl.Utf8).alias("model_name"),
                    pl.lit(authority.model_revision, dtype=pl.Utf8).alias("model_version"),
                ]
            )
            .select(_empty_item_features_long_frame().columns)
        )

    score_lf = pl.concat(
        [_scan_aligned_frame(path, _empty_sentence_score_frame().schema) for path in score_paths],
        how="vertical_relaxed",
    )
    token_weight = (
        pl.when(pl.col(FINBERT_TOKEN_COUNT_COLUMN).cast(pl.Float64, strict=False) > 0.0)
        .then(pl.col(FINBERT_TOKEN_COUNT_COLUMN).cast(pl.Float64, strict=False))
        .otherwise(pl.lit(0.0))
    )

    def _length_weighted_mean(value: pl.Expr, alias: str) -> pl.Expr:
        denominator = token_weight.sum()
        return (
            pl.when(denominator > 0.0)
            .then((value * token_weight).sum() / denominator)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(alias)
        )

    aggregated_lf = (
        score_lf.group_by("benchmark_row_id")
        .agg(
            [
                pl.len().cast(pl.Int32).alias("sentence_count"),
                pl.col("negative_prob").mean().alias("negative_prob_mean"),
                pl.col("neutral_prob").mean().alias("neutral_prob_mean"),
                pl.col("positive_prob").mean().alias("positive_prob_mean"),
                (pl.col("predicted_label") == "negative").mean().alias("argmax_share_negative"),
                (pl.col("predicted_label") == "neutral").mean().alias("argmax_share_neutral"),
                (pl.col("predicted_label") == "positive").mean().alias("argmax_share_positive"),
                pl.len().cast(pl.Int32).alias("finbert_segment_count"),
                token_weight.sum().round(0).cast(pl.Int64).alias("finbert_token_count_512_sum"),
                _length_weighted_mean(pl.col("negative_prob"), "finbert_neg_prob_lenw_mean"),
                _length_weighted_mean(pl.col("positive_prob"), "finbert_pos_prob_lenw_mean"),
                _length_weighted_mean(pl.col("neutral_prob"), "finbert_neu_prob_lenw_mean"),
                _length_weighted_mean(
                    pl.col("negative_prob") - pl.col("positive_prob"),
                    "finbert_net_negative_lenw_mean",
                ),
                (pl.col("predicted_label") == "negative").mean().alias("finbert_neg_dominant_share"),
            ]
        )
        .with_columns(
            (pl.col("positive_prob_mean") - pl.col("negative_prob_mean")).alias("sentiment_balance_mean")
        )
    )
    return (
        metadata_lf.join(aggregated_lf, on="benchmark_row_id", how="left")
        .with_columns(
            [
                pl.col("sentence_count").fill_null(0).cast(pl.Int32),
                pl.coalesce(
                    [
                        pl.col("segment_policy_id").cast(pl.Utf8, strict=False),
                        pl.lit(FINBERT_SEGMENT_POLICY_ID, dtype=pl.Utf8),
                    ]
                ).alias("segment_policy_id"),
                pl.lit(authority.model_name, dtype=pl.Utf8).alias("model_name"),
                pl.lit(authority.model_revision, dtype=pl.Utf8).alias("model_version"),
            ]
        )
        .select(_empty_item_features_long_frame().columns)
        .sort(["filing_year", "doc_id", "benchmark_item_code"])
    )


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
    _assert_existing_tokenizer_run_compatible(
        run_manifest_path,
        run_cfg,
        authority,
        year_paths=year_paths,
    )

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
                    "sentence_dataset_path": None,
                    "bucket_summary_path": relative_artifact_path(year_bucket_summary_path, base_path=run_dir),
                    "timing_summary_path": relative_artifact_path(year_timing_summary_path, base_path=run_dir),
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
                "sentence_dataset_path": None,
                "bucket_summary_path": relative_artifact_path(year_bucket_summary_path, base_path=run_dir),
                "timing_summary_path": relative_artifact_path(year_timing_summary_path, base_path=run_dir),
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
            "path_semantics": MANIFEST_PATH_SEMANTICS_RELATIVE,
            "runner_name": TOKENIZER_PROFILE_RUNNER_NAME,
            "run_name": run_name,
            "created_at_utc": utc_timestamp(),
            "authority": asdict(authority),
            "semantic_reuse_guard": _semantic_tokenizer_guard(
                run_cfg,
                authority,
                year_paths=year_paths,
            ),
            "batch_config": asdict(run_cfg.batch_config),
            "bucket_lengths": asdict(run_cfg.bucket_lengths),
            "year_filter": list(run_cfg.year_filter) if run_cfg.year_filter is not None else None,
            "profile_row_cap_per_bucket": run_cfg.profile_row_cap_per_bucket,
            "sample_seed": run_cfg.sample_seed,
            "tokenizer_profile_runs": TOKENIZER_PROFILE_RUNS,
            "overwrite": run_cfg.overwrite,
            "note": run_cfg.note,
            "nonportable_diagnostics": {
                "sentence_dataset_dir": str(run_cfg.sentence_dataset_dir.resolve()),
            },
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
                "run_dir": write_manifest_path_value(
                    run_dir,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "records_path": write_manifest_path_value(
                    records_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "bucket_summary_path": write_manifest_path_value(
                    bucket_summary_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "bucket_summary_csv_path": write_manifest_path_value(
                    bucket_summary_csv_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "timing_summary_path": write_manifest_path_value(
                    timing_summary_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "timing_summary_csv_path": write_manifest_path_value(
                    timing_summary_csv_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "by_year_dir": write_manifest_path_value(
                    bucket_summary_by_year_dir,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
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
    _assert_existing_inference_run_compatible(
        run_manifest_path,
        run_cfg,
        authority,
        year_paths=year_paths,
    )

    tokenizer = None
    model = None
    label_mapping: dict[int, str] | None = None
    warnings: list[str] = []
    year_results: list[dict[str, Any]] = []

    resolved_device = _resolve_device(run_cfg.runtime)
    runtime_environment = _runtime_environment(run_cfg.runtime, resolved_device)
    effective_year_filter = _effective_year_filter(run_cfg.year_filter, source_sentence_manifest)
    backbone_contract = _build_backbone_contract(
        backbone_path=run_cfg.backbone_path,
        effective_year_filter=effective_year_filter,
        source_sentence_manifest=source_sentence_manifest,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    sentence_score_schema = _empty_sentence_score_frame().schema
    item_feature_schema = _empty_item_features_long_frame().schema

    for year_path in year_paths:
        filing_year = int(year_path.stem)
        year_item_features_path = item_features_by_year_dir / f"{filing_year}.parquet"
        year_sentence_scores_path = sentence_scores_dir / f"{filing_year}.parquet"
        sentence_schema = pl.scan_parquet(year_path).collect_schema()
        _validate_sentence_frame_schema(
            sentence_schema,
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
        sentence_rows = int(pl.scan_parquet(year_path).select(pl.len()).collect().item())
        can_reuse = (
            year_item_features_path.exists()
            and not run_cfg.overwrite
            and (not run_cfg.write_sentence_scores or year_sentence_scores_path.exists())
        )
        if can_reuse:
            item_feature_rows = int(pl.scan_parquet(year_item_features_path).select(pl.len()).collect().item())
            doc_rows = int(
                pl.scan_parquet(year_item_features_path)
                .select(pl.col("doc_id").n_unique())
                .collect()
                .item()
            )
            year_results.append(
                {
                    "filing_year": filing_year,
                    "status": "reused_existing",
                    "sentence_dataset_path": None,
                    "sentence_rows": sentence_rows,
                    "item_feature_rows": item_feature_rows,
                    "doc_rows": doc_rows,
                    "item_features_path": relative_artifact_path(year_item_features_path, base_path=run_dir),
                    "sentence_scores_path": (
                        relative_artifact_path(year_sentence_scores_path, base_path=run_dir)
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

        score_shard_dir = run_dir / "_temp_sentence_score_shards" / f"{filing_year}"
        if score_shard_dir.exists():
            shutil.rmtree(score_shard_dir)

        score_shard_paths: list[Path] = []
        if sentence_rows == 0:
            _empty_item_features_long_frame().write_parquet(
                year_item_features_path,
                compression=_STREAMING_PARQUET_COMPRESSION,
            )
            if run_cfg.write_sentence_scores:
                year_sentence_scores_path.parent.mkdir(parents=True, exist_ok=True)
                _empty_sentence_score_frame().write_parquet(
                    year_sentence_scores_path,
                    compression=_STREAMING_PARQUET_COMPRESSION,
                )
        else:
            batch_iter = pl.scan_parquet(year_path).collect_batches(
                chunk_size=max(run_cfg.sentence_slice_rows, 1)
            )
            for batch_index, batch in enumerate(batch_iter, start=1):
                sentence_batch = batch if isinstance(batch, pl.DataFrame) else pl.DataFrame(batch)
                if sentence_batch.is_empty():
                    continue
                sentence_scores_df = _score_sentence_frame_in_slices(
                    sentence_batch,
                    tokenizer,
                    model,
                    run_cfg,
                )
                if sentence_scores_df.is_empty():
                    continue
                shard_path = score_shard_dir / f"{batch_index:06d}.parquet"
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                sentence_scores_df.write_parquet(
                    shard_path,
                    compression=_STREAMING_PARQUET_COMPRESSION,
                )
                score_shard_paths.append(shard_path)

            year_item_features_path.parent.mkdir(parents=True, exist_ok=True)
            if year_item_features_path.exists():
                year_item_features_path.unlink()
            _lazy_item_features_from_sentence_score_paths(
                score_shard_paths,
                sentence_path=year_path,
                authority=authority,
            ).sink_parquet(
                year_item_features_path,
                compression=_STREAMING_PARQUET_COMPRESSION,
            )
            if run_cfg.write_sentence_scores:
                _sink_parquet_from_paths(
                    score_shard_paths,
                    output_path=year_sentence_scores_path,
                    schema=sentence_score_schema,
                    compression=_STREAMING_PARQUET_COMPRESSION,
                )
        if score_shard_dir.exists():
            shutil.rmtree(score_shard_dir)

        item_feature_rows = int(pl.scan_parquet(year_item_features_path).select(pl.len()).collect().item())
        doc_rows = int(
            pl.scan_parquet(year_item_features_path)
            .select(pl.col("doc_id").n_unique())
            .collect()
            .item()
        )
        year_results.append(
            {
                "filing_year": filing_year,
                "status": "processed" if sentence_rows else "processed_empty",
                "sentence_dataset_path": None,
                "sentence_rows": sentence_rows,
                "item_feature_rows": item_feature_rows,
                "doc_rows": doc_rows,
                "item_features_path": relative_artifact_path(year_item_features_path, base_path=run_dir),
                "sentence_scores_path": (
                    relative_artifact_path(year_sentence_scores_path, base_path=run_dir)
                    if run_cfg.write_sentence_scores
                    else None
                ),
            }
        )

    yearly_summary_df = _concat_aligned_frames(
        [pl.DataFrame([row]) for row in year_results],
        empty_schema=_empty_model_yearly_summary_frame(),
    ).sort("filing_year")

    yearly_summary_df.write_parquet(yearly_summary_path, compression="zstd")
    yearly_summary_df.write_csv(yearly_summary_csv_path)

    year_item_feature_paths = [
        item_features_by_year_dir / f"{int(path.stem)}.parquet"
        for path in year_paths
        if (item_features_by_year_dir / f"{int(path.stem)}.parquet").exists()
    ]
    _sink_parquet_from_paths(
        year_item_feature_paths,
        output_path=item_features_long_path,
        schema=item_feature_schema,
        compression=_STREAMING_PARQUET_COMPRESSION,
    )
    item_features_long = pl.read_parquet(item_features_long_path).sort(
        ["filing_year", "doc_id", "benchmark_item_code"]
    )
    item_features_long.write_parquet(item_features_long_path, compression=_STREAMING_PARQUET_COMPRESSION)

    doc_features_wide = pivot_item_features_to_doc_wide(item_features_long)
    doc_features_wide.write_parquet(doc_features_wide_path, compression=_STREAMING_PARQUET_COMPRESSION)

    coverage_summary: dict[str, int] | None = None
    if coverage_report_path is not None and run_cfg.backbone_path is not None:
        coverage_report, coverage_summary = build_coverage_report(
            item_features_long,
            _load_backbone_doc_ids_for_years(run_cfg.backbone_path, year_filter=effective_year_filter),
        )
        coverage_report.write_parquet(coverage_report_path, compression=_STREAMING_PARQUET_COMPRESSION)
    elif coverage_report_path is None:
        warnings.append("backbone_path_not_provided")

    if label_mapping is None:
        warnings.append("model_not_loaded_all_years_reused_existing_artifacts")

    write_json(
        run_manifest_path,
        {
            "path_semantics": MANIFEST_PATH_SEMANTICS_RELATIVE,
            "runner_name": MODEL_INFERENCE_RUNNER_NAME,
            "run_name": run_name,
            "created_at_utc": utc_timestamp(),
            "authority": asdict(authority),
            "semantic_reuse_guard": _semantic_inference_guard(
                run_cfg,
                authority,
                year_paths=year_paths,
            ),
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
            "year_filter": list(run_cfg.year_filter) if run_cfg.year_filter is not None else None,
            "backbone_contract": backbone_contract,
            "sentence_slice_rows": run_cfg.sentence_slice_rows,
            "write_sentence_scores": run_cfg.write_sentence_scores,
            "overwrite": run_cfg.overwrite,
            "note": run_cfg.note,
            "nonportable_diagnostics": {
                "sentence_dataset_dir": str(run_cfg.sentence_dataset_dir.resolve()),
                "backbone_path": (
                    str(run_cfg.backbone_path.resolve())
                    if run_cfg.backbone_path is not None
                    else None
                ),
            },
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
                "run_dir": write_manifest_path_value(
                    run_dir,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "item_features_by_year_dir": write_manifest_path_value(
                    item_features_by_year_dir,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "item_features_long_path": write_manifest_path_value(
                    item_features_long_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "doc_features_wide_path": write_manifest_path_value(
                    doc_features_wide_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "coverage_report_path": (
                    write_manifest_path_value(
                        coverage_report_path,
                        manifest_path=run_manifest_path,
                        path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                    )
                    if coverage_report_path is not None
                    else None
                ),
                "sentence_scores_dir": (
                    write_manifest_path_value(
                        sentence_scores_dir,
                        manifest_path=run_manifest_path,
                        path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                    )
                    if run_cfg.write_sentence_scores
                    else None
                ),
                "yearly_summary_path": write_manifest_path_value(
                    yearly_summary_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "yearly_summary_csv_path": write_manifest_path_value(
                    yearly_summary_csv_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
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
