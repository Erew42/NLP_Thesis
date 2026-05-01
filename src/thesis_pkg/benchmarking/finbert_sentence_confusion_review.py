from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import polars as pl


THESIS_REVIEW_TEXT_SCOPES: tuple[str, ...] = ("item_1a_risk_factors", "item_7_mda")
STRATUM_COLUMNS: tuple[str, ...] = (
    "text_scope",
    "predicted_label",
    "probability_majority_bucket",
)
MAJORITY_BUCKET_VALUES: tuple[str, ...] = (
    "negative_majority",
    "neutral_majority",
    "positive_majority",
    "no_majority",
)
ALLOCATION_MODE_VALUES: tuple[str, ...] = ("balanced", "proportional")
LABEL_VALUES: tuple[str, ...] = ("yes", "no", "uncertain")
CONFUSION_VALUES: tuple[str, ...] = ("TP", "FP", "FN", "TN", "uncertain")

REQUIRED_SENTENCE_SCORE_COLUMNS: tuple[str, ...] = (
    "benchmark_sentence_id",
    "benchmark_row_id",
    "doc_id",
    "filing_year",
    "benchmark_item_code",
    "text_scope",
    "sentence_index",
    "sentence_text",
    "finbert_token_count_512",
    "finbert_token_bucket_512",
    "negative_prob",
    "neutral_prob",
    "positive_prob",
    "predicted_label",
)
OPTIONAL_SENTENCE_SCORE_COLUMNS: tuple[str, ...] = (
    "cik_10",
    "accession_nodash",
    "filing_date",
    "benchmark_item_label",
    "source_year_file",
    "document_type",
    "document_type_raw",
    "document_type_normalized",
    "canonical_item",
    "cleaning_policy_id",
    "segment_policy_id",
    "sentence_char_count",
    "sentencizer_backend",
    "sentencizer_version",
)
SAMPLE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "review_case_id",
    "sample_order",
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
    "canonical_item",
    "text_scope",
    "cleaning_policy_id",
    "segment_policy_id",
    "sentence_index",
    "sentence_text",
    "prev_text",
    "next_text",
    "sentence_char_count",
    "sentencizer_backend",
    "sentencizer_version",
    "finbert_token_count_512",
    "finbert_token_bucket_512",
    "negative_prob",
    "neutral_prob",
    "positive_prob",
    "predicted_label",
    "probability_majority_bucket",
    "finbert_predicted_negative",
    "stratum_id",
    "population_count",
    "sample_count",
    "sample_weight",
    "sample_key",
)


@dataclass(frozen=True)
class FinbertSentenceConfusionReviewConfig:
    sample_size: int = 1000
    seed: int = 42
    text_scopes: tuple[str, ...] = THESIS_REVIEW_TEXT_SCOPES
    chunk_count: int = 10
    include_no_majority: bool = False
    allocation_mode: str = "balanced"
    stream_batch_size: int = 25_000
    initial_oversampling_factor: float = 8.0
    max_oversampling_factor: float = 256.0


@dataclass(frozen=True)
class FinbertSentenceConfusionReviewArtifacts:
    output_dir: Path
    manifest_path: Path
    population_counts_by_majority_bucket_path: Path
    population_counts_by_stratum_path: Path
    population_counts_summary_path: Path
    sample_path: Path | None
    sample_csv_path: Path | None
    review_html_path: Path | None
    labeling_prompt_path: Path | None
    chunk_dir: Path | None
    sample_row_count: int
    counts_only: bool
    oversampling_factor: float | None


@dataclass(frozen=True)
class FinbertSentenceConfusionSummaryArtifacts:
    output_dir: Path
    reviewed_cases_path: Path
    reviewed_cases_csv_path: Path
    confusion_matrix_path: Path
    metrics_json_path: Path
    metrics_markdown_path: Path
    majority_bucket_metrics_path: Path
    examples_by_cell_path: Path


def probability_majority_bucket_expr() -> pl.Expr:
    return (
        pl.when(pl.col("negative_prob") > 0.5)
        .then(pl.lit("negative_majority"))
        .when(pl.col("neutral_prob") > 0.5)
        .then(pl.lit("neutral_majority"))
        .when(pl.col("positive_prob") > 0.5)
        .then(pl.lit("positive_majority"))
        .otherwise(pl.lit("no_majority"))
        .alias("probability_majority_bucket")
    )


def add_probability_majority_bucket(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(probability_majority_bucket_expr())


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _resolve_sentence_scores_by_year_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_file():
        raise ValueError(f"Expected a sentence_scores directory or by_year directory, got file: {resolved}")
    if (resolved / "by_year").is_dir():
        return (resolved / "by_year").resolve()
    if any(resolved.glob("*.parquet")):
        return resolved
    raise FileNotFoundError(
        f"Could not find parquet shards under {resolved} or {resolved / 'by_year'}."
    )


def _sentence_score_parquet_glob(path: Path) -> str:
    return str(_resolve_sentence_scores_by_year_dir(path) / "*.parquet")


def _sentence_score_parquet_paths(path: Path) -> tuple[Path, ...]:
    by_year_dir = _resolve_sentence_scores_by_year_dir(path)
    parquet_paths = tuple(sorted(by_year_dir.glob("*.parquet")))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet shards found under {by_year_dir}.")
    return parquet_paths


def _scan_sentence_score_source(source: Path | str) -> pl.LazyFrame:
    scan = pl.scan_parquet(str(source))
    schema = scan.collect_schema()
    missing = [column for column in REQUIRED_SENTENCE_SCORE_COLUMNS if column not in schema]
    if missing:
        raise ValueError(f"sentence_scores is missing required columns: {missing}")

    select_exprs: list[pl.Expr] = [pl.col(column) for column in REQUIRED_SENTENCE_SCORE_COLUMNS]
    for column in OPTIONAL_SENTENCE_SCORE_COLUMNS:
        if column in schema:
            select_exprs.append(pl.col(column))
        else:
            select_exprs.append(pl.lit(None).alias(column))
    return scan.select(select_exprs)


def _scan_sentence_scores(path: Path) -> pl.LazyFrame:
    return _scan_sentence_score_source(_sentence_score_parquet_glob(path))


def _prepared_sentence_lf(sentence_scores_dir: Path, text_scopes: tuple[str, ...]) -> pl.LazyFrame:
    return (
        _scan_sentence_scores(sentence_scores_dir)
        .filter(pl.col("text_scope").is_in(list(text_scopes)))
        .with_columns(
            probability_majority_bucket_expr(),
            (pl.col("predicted_label") == "negative").alias("finbert_predicted_negative"),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("text_scope"),
                    pl.lit("||"),
                    pl.col("predicted_label"),
                    pl.lit("||"),
                    pl.col("probability_majority_bucket"),
                ]
            ).alias("stratum_id")
        )
    )


def _prepared_sentence_shard_lf(parquet_path: Path, text_scopes: tuple[str, ...]) -> pl.LazyFrame:
    return (
        _scan_sentence_score_source(parquet_path)
        .filter(pl.col("text_scope").is_in(list(text_scopes)))
        .with_columns(
            probability_majority_bucket_expr(),
            (pl.col("predicted_label") == "negative").alias("finbert_predicted_negative"),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("text_scope"),
                    pl.lit("||"),
                    pl.col("predicted_label"),
                    pl.lit("||"),
                    pl.col("probability_majority_bucket"),
                ]
            ).alias("stratum_id")
        )
    )


def _prepare_sentence_batch_df(df: pl.DataFrame, text_scopes: tuple[str, ...]) -> pl.DataFrame:
    return (
        df.filter(pl.col("text_scope").is_in(list(text_scopes)))
        .with_columns(
            probability_majority_bucket_expr(),
            (pl.col("predicted_label") == "negative").alias("finbert_predicted_negative"),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("text_scope"),
                    pl.lit("||"),
                    pl.col("predicted_label"),
                    pl.lit("||"),
                    pl.col("probability_majority_bucket"),
                ]
            ).alias("stratum_id")
        )
    )


def _iter_sentence_score_batches(
    parquet_paths: tuple[Path, ...],
    *,
    batch_size: int,
) -> Any:
    if batch_size <= 0:
        raise ValueError("stream_batch_size must be positive.")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - pyarrow is a project dependency.
        raise RuntimeError("pyarrow is required for memory-safe streaming sample builds.") from exc

    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        available_columns = set(parquet_file.schema_arrow.names)
        missing = [column for column in REQUIRED_SENTENCE_SCORE_COLUMNS if column not in available_columns]
        if missing:
            raise ValueError(f"Parquet shard {parquet_path} is missing required columns: {missing}")
        read_columns = [
            column
            for column in (*REQUIRED_SENTENCE_SCORE_COLUMNS, *OPTIONAL_SENTENCE_SCORE_COLUMNS)
            if column in available_columns
        ]
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
            df = pl.from_arrow(batch)
            for column in OPTIONAL_SENTENCE_SCORE_COLUMNS:
                if column not in df.columns:
                    df = df.with_columns(pl.lit(None).alias(column))
            yield df.select([*REQUIRED_SENTENCE_SCORE_COLUMNS, *OPTIONAL_SENTENCE_SCORE_COLUMNS])


def _counts_preflight(
    sentence_scores_dir: Path,
    output_dir: Path,
    *,
    text_scopes: tuple[str, ...],
    sample_size: int,
    seed: int,
    include_no_majority: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    lf = _prepared_sentence_lf(sentence_scores_dir, text_scopes)
    output_dir.mkdir(parents=True, exist_ok=True)

    stratum_counts_raw = (
        lf.group_by(list(STRATUM_COLUMNS))
        .len()
        .rename({"len": "population_count"})
        .with_columns(
            pl.concat_str(
                [
                    pl.col("text_scope"),
                    pl.lit("||"),
                    pl.col("predicted_label"),
                    pl.lit("||"),
                    pl.col("probability_majority_bucket"),
                ]
            ).alias("stratum_id")
        )
        .select(["stratum_id", *STRATUM_COLUMNS, "population_count"])
        .sort(["text_scope", "predicted_label", "probability_majority_bucket"])
        .collect()
    )
    stratum_counts = stratum_counts_raw.with_columns(
        (
            pl.lit(include_no_majority)
            | (pl.col("probability_majority_bucket") != "no_majority")
        ).alias("sample_eligible")
    ).with_columns(
        pl.when(pl.col("sample_eligible"))
        .then(pl.lit(None))
        .otherwise(pl.lit("excluded_no_majority"))
        .alias("exclusion_reason")
    )

    bucket_counts = (
        stratum_counts.group_by("probability_majority_bucket")
        .agg(
            pl.col("population_count").sum().alias("population_count"),
            pl.when(pl.col("sample_eligible"))
            .then(pl.col("population_count"))
            .otherwise(0)
            .sum()
            .alias("sample_eligible_population_count"),
        )
        .sort("probability_majority_bucket")
    )
    label_counts = (
        stratum_counts.group_by("predicted_label")
        .agg(
            pl.col("population_count").sum().alias("population_count"),
            pl.when(pl.col("sample_eligible"))
            .then(pl.col("population_count"))
            .otherwise(0)
            .sum()
            .alias("sample_eligible_population_count"),
        )
        .sort("predicted_label")
    )
    scope_counts = (
        stratum_counts.group_by("text_scope")
        .agg(
            pl.col("population_count").sum().alias("population_count"),
            pl.when(pl.col("sample_eligible"))
            .then(pl.col("population_count"))
            .otherwise(0)
            .sum()
            .alias("sample_eligible_population_count"),
        )
        .sort("text_scope")
    )

    total_rows = int(stratum_counts["population_count"].sum()) if stratum_counts.height else 0
    eligible_rows = int(
        stratum_counts.filter(pl.col("sample_eligible"))["population_count"].sum()
    ) if stratum_counts.height else 0
    summary = {
        "generated_at": _utc_now_iso(),
        "sentence_scores_dir": str(_resolve_sentence_scores_by_year_dir(sentence_scores_dir)),
        "text_scopes": list(text_scopes),
        "requested_sample_size": sample_size,
        "seed": seed,
        "include_no_majority": include_no_majority,
        "total_population_rows": total_rows,
        "sample_eligible_population_rows": eligible_rows,
        "non_empty_stratum_count": int(stratum_counts.height),
        "sample_eligible_stratum_count": int(stratum_counts.filter(pl.col("sample_eligible")).height),
        "majority_bucket_values": list(MAJORITY_BUCKET_VALUES),
        "stratum_columns": list(STRATUM_COLUMNS),
        "counts_by_majority_bucket": bucket_counts.to_dicts(),
        "counts_by_predicted_label": label_counts.to_dicts(),
        "counts_by_text_scope": scope_counts.to_dicts(),
    }

    bucket_counts.write_csv(output_dir / "population_counts_by_majority_bucket.csv")
    stratum_counts.write_csv(output_dir / "population_counts_by_stratum.csv")
    _write_json(output_dir / "population_counts_summary.json", summary)
    return bucket_counts, stratum_counts, summary


def _finalize_allocation_rows(
    rows: list[dict[str, Any]],
    *,
    total_population: int,
    allocation_mode: str,
) -> pl.DataFrame:
    allocated_rows = []
    for row in rows:
        sample_count = int(row["sample_count"])
        population_count = int(row["population_count"])
        allocated_rows.append(
            {
                **row,
                "sample_count": sample_count,
                "sample_weight": (population_count / sample_count) if sample_count else None,
                "population_fraction": population_count / total_population,
                "allocation_mode": allocation_mode,
            }
        )
    return pl.DataFrame(allocated_rows).sort(
        ["text_scope", "predicted_label", "probability_majority_bucket"]
    )


def _allocate_balanced_sample_counts(
    rows: list[dict[str, Any]],
    *,
    target_sample_size: int,
    total_population: int,
) -> pl.DataFrame:
    ordered_rows = sorted(rows, key=lambda row: str(row["stratum_id"]))
    active_rows = [row for row in ordered_rows if int(row["population_count"]) > 0]
    if not active_rows:
        raise ValueError("No eligible rows found for the requested review universe.")

    base = target_sample_size // len(active_rows)
    remainder = target_sample_size % len(active_rows)
    allocation_rows = []
    for index, row in enumerate(active_rows):
        allocation_rows.append(
            {
                **row,
                "sample_count": min(base + (1 if index < remainder else 0), int(row["population_count"])),
            }
        )

    allocated = sum(int(row["sample_count"]) for row in allocation_rows)
    while allocated < target_sample_size:
        progressed = False
        for row in allocation_rows:
            if allocated >= target_sample_size:
                break
            if int(row["sample_count"]) < int(row["population_count"]):
                row["sample_count"] = int(row["sample_count"]) + 1
                allocated += 1
                progressed = True
        if not progressed:
            break

    if allocated != target_sample_size:
        raise RuntimeError(
            f"Balanced allocation produced {allocated} rows, expected {target_sample_size}."
        )
    return _finalize_allocation_rows(
        allocation_rows,
        total_population=total_population,
        allocation_mode="balanced",
    )


def _allocate_proportional_sample_counts(
    rows: list[dict[str, Any]],
    *,
    target_sample_size: int,
    total_population: int,
) -> pl.DataFrame:
    if target_sample_size == total_population:
        allocated = [
            {
                **row,
                "sample_count": int(row["population_count"]),
            }
            for row in rows
        ]
        return _finalize_allocation_rows(
            allocated,
            total_population=total_population,
            allocation_mode="proportional",
        )

    quota_rows: list[dict[str, Any]] = []
    running_floor = 0
    for row in rows:
        population_count = int(row["population_count"])
        exact = target_sample_size * population_count / total_population
        floor_count = int(math.floor(exact))
        quota_rows.append(
            {
                **row,
                "_exact_quota": exact,
                "_remainder": exact - floor_count,
                "sample_count": min(floor_count, population_count),
            }
        )
        running_floor += min(floor_count, population_count)

    remaining = target_sample_size - running_floor
    for row in sorted(quota_rows, key=lambda item: (-float(item["_remainder"]), str(item["stratum_id"]))):
        if remaining <= 0:
            break
        if int(row["sample_count"]) < int(row["population_count"]):
            row["sample_count"] = int(row["sample_count"]) + 1
            remaining -= 1

    if target_sample_size >= len(quota_rows):
        zero_rows = [row for row in quota_rows if int(row["sample_count"]) == 0 and int(row["population_count"]) > 0]
        for zero_row in zero_rows:
            donors = [
                row
                for row in quota_rows
                if int(row["sample_count"]) > 1
                and float(row["_exact_quota"]) < float(row["sample_count"])
            ]
            if not donors:
                donors = [row for row in quota_rows if int(row["sample_count"]) > 1]
            if not donors:
                break
            donor = sorted(
                donors,
                key=lambda item: (
                    float(item["sample_count"]) - float(item["_exact_quota"]),
                    int(item["sample_count"]),
                ),
                reverse=True,
            )[0]
            donor["sample_count"] = int(donor["sample_count"]) - 1
            zero_row["sample_count"] = 1

    actual_total = sum(int(row["sample_count"]) for row in quota_rows)
    if actual_total != target_sample_size:
        raise RuntimeError(
            f"Internal allocation error: allocated {actual_total}, expected {target_sample_size}."
        )

    allocated_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"_exact_quota", "_remainder"}
        }
        for row in quota_rows
    ]
    return _finalize_allocation_rows(
        allocated_rows,
        total_population=total_population,
        allocation_mode="proportional",
    )


def _allocate_sample_counts(
    stratum_counts: pl.DataFrame,
    sample_size: int,
    *,
    allocation_mode: str,
) -> pl.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if allocation_mode not in ALLOCATION_MODE_VALUES:
        raise ValueError(f"allocation_mode must be one of {ALLOCATION_MODE_VALUES}, got {allocation_mode!r}.")
    if stratum_counts.is_empty():
        raise ValueError("No eligible rows found for the requested review universe.")

    rows = stratum_counts.to_dicts()
    total_population = sum(int(row["population_count"]) for row in rows)
    if total_population <= 0:
        raise ValueError("No eligible rows found for the requested review universe.")
    target_sample_size = min(sample_size, total_population)
    if allocation_mode == "balanced":
        return _allocate_balanced_sample_counts(
            rows,
            target_sample_size=target_sample_size,
            total_population=total_population,
        )
    return _allocate_proportional_sample_counts(
        rows,
        target_sample_size=target_sample_size,
        total_population=total_population,
    )


def _sample_key_expr(seed: int) -> pl.Expr:
    max_uint64 = float(2**64 - 1)
    return (
        pl.concat_str([pl.lit(str(seed)), pl.lit("::"), pl.col("benchmark_sentence_id")])
        .hash(seed=seed)
        .cast(pl.Float64)
        .truediv(max_uint64)
        .alias("sample_key")
    )


def _stable_random(seed: int, value: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}::{value}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big", signed=False))


def _target_positions_for_allocation(
    allocation_df: pl.DataFrame,
    *,
    seed: int,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    target_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for allocation_row in allocation_df.to_dicts():
        stratum_id = str(allocation_row["stratum_id"])
        population_count = int(allocation_row["population_count"])
        sample_count = int(allocation_row["sample_count"])
        if sample_count <= 0:
            continue
        if sample_count > population_count:
            raise ValueError(
                f"Cannot sample {sample_count} rows from stratum {stratum_id} "
                f"with population {population_count}."
            )
        rng = _stable_random(seed, stratum_id)
        positions = sorted(rng.sample(range(population_count), sample_count))
        target_rows.extend(
            {
                "stratum_id": stratum_id,
                "_target_ordinal": int(position),
                "_target_rank": rank,
            }
            for rank, position in enumerate(positions)
        )
        summary_rows.append(
            {
                "stratum_id": stratum_id,
                "population_count": population_count,
                "sample_count": sample_count,
                "first_target_ordinal": positions[0] if positions else None,
                "last_target_ordinal": positions[-1] if positions else None,
            }
        )
    schema = {
        "stratum_id": pl.Utf8,
        "_target_ordinal": pl.Int64,
        "_target_rank": pl.Int64,
    }
    return pl.DataFrame(target_rows, schema=schema), summary_rows


def _collect_exact_sample_streaming(
    parquet_paths: tuple[Path, ...],
    allocation_df: pl.DataFrame,
    *,
    text_scopes: tuple[str, ...],
    seed: int,
    batch_size: int,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    target_df, target_summary = _target_positions_for_allocation(allocation_df, seed=seed)
    if target_df.is_empty():
        raise ValueError("Allocation produced no target positions.")
    target_strata = set(target_df["stratum_id"].to_list())
    offsets: dict[str, int] = {stratum_id: 0 for stratum_id in target_strata}
    allocation_metadata = allocation_df.select(
        ["stratum_id", "population_count", "sample_count", "sample_weight"]
    )
    selected_frames: list[pl.DataFrame] = []
    scanned_batches = 0
    retained_rows = 0

    for batch_df in _iter_sentence_score_batches(parquet_paths, batch_size=batch_size):
        scanned_batches += 1
        prepared = _prepare_sentence_batch_df(batch_df, text_scopes).filter(
            pl.col("stratum_id").is_in(target_strata)
        )
        if prepared.is_empty():
            continue
        present_counts = prepared.group_by("stratum_id").len().rename({"len": "_batch_count"})
        offset_df = pl.DataFrame(
            [
                {"stratum_id": row["stratum_id"], "_stratum_seen_offset": offsets[str(row["stratum_id"])]}
                for row in present_counts.to_dicts()
            ],
            schema={"stratum_id": pl.Utf8, "_stratum_seen_offset": pl.Int64},
        )
        with_ordinals = (
            prepared.join(offset_df, on="stratum_id", how="left")
            .with_columns(
                (
                    pl.col("_stratum_seen_offset")
                    + pl.int_range(0, pl.len()).over("stratum_id")
                ).alias("_target_ordinal")
            )
            .drop("_stratum_seen_offset")
        )
        hits = (
            with_ordinals.join(target_df, on=["stratum_id", "_target_ordinal"], how="inner")
            .join(allocation_metadata, on="stratum_id", how="left")
            .select(
                [
                    *REQUIRED_SENTENCE_SCORE_COLUMNS,
                    *OPTIONAL_SENTENCE_SCORE_COLUMNS,
                    "probability_majority_bucket",
                    "finbert_predicted_negative",
                    "stratum_id",
                    "population_count",
                    "sample_count",
                    "sample_weight",
                    "_target_ordinal",
                    "_target_rank",
                ]
            )
        )
        if not hits.is_empty():
            retained_rows += int(hits.height)
            selected_frames.append(hits)
        for row in present_counts.to_dicts():
            stratum_id = str(row["stratum_id"])
            offsets[stratum_id] += int(row["_batch_count"])

    selected = (
        pl.concat(selected_frames, how="vertical_relaxed")
        if selected_frames
        else pl.DataFrame()
    )
    underfilled = _underfilled_strata(selected, allocation_df)
    if not underfilled.is_empty():
        raise RuntimeError(
            "Streaming sample did not find every target row. "
            f"Underfilled strata: {underfilled.to_dicts()}"
        )
    selected = selected.with_columns(_sample_key_expr(seed))
    sampling_summary = {
        "sampling_method": "stream_target_ordinals",
        "batch_size": batch_size,
        "scanned_parquet_shards": len(parquet_paths),
        "scanned_batches": scanned_batches,
        "retained_sample_rows": retained_rows,
        "target_positions_by_stratum": target_summary,
    }
    return selected, [sampling_summary]


def _candidate_threshold(population_count: int, sample_count: int, oversampling_factor: float) -> float:
    if population_count <= 0 or sample_count <= 0:
        return 0.0
    return min(float(oversampling_factor) * sample_count / population_count, 1.0)


def _select_exact_allocated_candidates(candidates: pl.DataFrame) -> pl.DataFrame:
    if candidates.is_empty():
        return candidates
    return (
        candidates.sort(["stratum_id", "sample_key"])
        .with_columns(pl.int_range(0, pl.len()).over("stratum_id").alias("_stratum_rank"))
        .filter(pl.col("_stratum_rank") < pl.col("sample_count"))
        .drop("_stratum_rank")
    )


def _underfilled_strata(selected: pl.DataFrame, allocation_df: pl.DataFrame) -> pl.DataFrame:
    expected = allocation_df.filter(pl.col("sample_count") > 0).select(["stratum_id", "sample_count"])
    if expected.is_empty():
        return expected.with_columns(pl.lit(0).alias("selected_count"))
    if selected.is_empty():
        actual = pl.DataFrame({"stratum_id": [], "selected_count": []}, schema={"stratum_id": pl.Utf8, "selected_count": pl.Int64})
    else:
        actual = selected.group_by("stratum_id").len().rename({"len": "selected_count"})
    return (
        expected.join(actual, on="stratum_id", how="left")
        .with_columns(pl.col("selected_count").fill_null(0))
        .filter(pl.col("selected_count") < pl.col("sample_count"))
        .sort("stratum_id")
    )


def _collect_exact_sample_from_shards(
    parquet_paths: tuple[Path, ...],
    allocation_df: pl.DataFrame,
    *,
    text_scopes: tuple[str, ...],
    seed: int,
    initial_oversampling_factor: float,
    max_oversampling_factor: float,
) -> tuple[pl.DataFrame, float, list[dict[str, Any]]]:
    if allocation_df.filter(pl.col("sample_count") > 0).is_empty():
        raise ValueError("Allocation has no rows with sample_count > 0.")

    oversampling_factor = float(initial_oversampling_factor)
    attempts: list[dict[str, Any]] = []
    selected = pl.DataFrame()
    allocation_rows = [
        row
        for row in allocation_df.to_dicts()
        if int(row.get("sample_count") or 0) > 0
    ]
    while oversampling_factor <= max_oversampling_factor + 1e-9:
        candidate_frames: list[pl.DataFrame] = []
        candidate_row_count = 0
        for allocation_row in allocation_rows:
            threshold = _candidate_threshold(
                int(allocation_row["population_count"]),
                int(allocation_row["sample_count"]),
                oversampling_factor,
            )
            if threshold <= 0:
                continue
            for parquet_path in parquet_paths:
                candidate_lf = (
                    _prepared_sentence_shard_lf(parquet_path, text_scopes)
                    .filter(
                        (pl.col("text_scope") == pl.lit(str(allocation_row["text_scope"])))
                        & (pl.col("predicted_label") == pl.lit(str(allocation_row["predicted_label"])))
                        & (
                            pl.col("probability_majority_bucket")
                            == pl.lit(str(allocation_row["probability_majority_bucket"]))
                        )
                    )
                    .with_columns(_sample_key_expr(seed))
                    .filter(pl.col("sample_key") <= pl.lit(threshold))
                    .with_columns(
                        pl.lit(int(allocation_row["population_count"])).alias("population_count"),
                        pl.lit(int(allocation_row["sample_count"])).alias("sample_count"),
                        pl.lit(float(allocation_row["sample_weight"])).alias("sample_weight"),
                    )
                    .select(
                        [
                            *REQUIRED_SENTENCE_SCORE_COLUMNS,
                            *OPTIONAL_SENTENCE_SCORE_COLUMNS,
                            "probability_majority_bucket",
                            "finbert_predicted_negative",
                            "stratum_id",
                            "population_count",
                            "sample_count",
                            "sample_weight",
                            "sample_key",
                        ]
                    )
                )
                candidate_frame = candidate_lf.collect()
                candidate_row_count += int(candidate_frame.height)
                if not candidate_frame.is_empty():
                    candidate_frames.append(candidate_frame)
        candidates = (
            pl.concat(candidate_frames, how="vertical_relaxed")
            if candidate_frames
            else pl.DataFrame()
        )
        selected = _select_exact_allocated_candidates(candidates)
        underfilled = _underfilled_strata(selected, allocation_df)
        attempts.append(
            {
                "oversampling_factor": oversampling_factor,
                "candidate_row_count": candidate_row_count,
                "selected_row_count": int(selected.height),
                "underfilled_stratum_count": int(underfilled.height),
                "underfilled_strata": underfilled.to_dicts(),
            }
        )
        if underfilled.is_empty():
            return selected, oversampling_factor, attempts
        oversampling_factor *= 2.0

    raise RuntimeError(
        "Could not collect enough candidates for every allocated stratum. "
        f"Last underfilled strata: {attempts[-1]['underfilled_strata']}"
    )


def _attach_neighbor_context(
    sample_df: pl.DataFrame,
    parquet_paths: tuple[Path, ...],
    text_scopes: tuple[str, ...],
    batch_size: int,
) -> pl.DataFrame:
    if sample_df.is_empty():
        return sample_df.with_columns(pl.lit(None).alias("prev_text"), pl.lit(None).alias("next_text"))

    target_rows: list[dict[str, object]] = []
    for row in sample_df.select(["review_case_id", "benchmark_row_id", "sentence_index"]).to_dicts():
        sentence_index = row["sentence_index"]
        if sentence_index is None:
            continue
        try:
            current_index = int(sentence_index)
        except (TypeError, ValueError):
            continue
        target_rows.append(
            {
                "review_case_id": row["review_case_id"],
                "neighbor_kind": "prev_text",
                "benchmark_row_id": row["benchmark_row_id"],
                "sentence_index": current_index - 1,
            }
        )
        target_rows.append(
            {
                "review_case_id": row["review_case_id"],
                "neighbor_kind": "next_text",
                "benchmark_row_id": row["benchmark_row_id"],
                "sentence_index": current_index + 1,
            }
        )

    sample_rows = sample_df.to_dicts()
    if not target_rows:
        for row in sample_rows:
            row["prev_text"] = None
            row["next_text"] = None
        return pl.DataFrame(sample_rows)

    target_df = pl.DataFrame(target_rows).filter(pl.col("sentence_index") >= 0)
    if target_df.is_empty():
        for row in sample_rows:
            row["prev_text"] = None
            row["next_text"] = None
        return pl.DataFrame(sample_rows)

    neighbor_frames: list[pl.DataFrame] = []
    for batch_df in _iter_sentence_score_batches(parquet_paths, batch_size=batch_size):
        neighbor_frame = (
            _prepare_sentence_batch_df(batch_df, text_scopes)
            .select(["benchmark_row_id", "sentence_index", "sentence_text"])
            .join(
                target_df,
                on=["benchmark_row_id", "sentence_index"],
                how="inner",
            )
        )
        if not neighbor_frame.is_empty():
            neighbor_frames.append(neighbor_frame)
    neighbors = (
        pl.concat(neighbor_frames, how="vertical_relaxed")
        if neighbor_frames
        else pl.DataFrame(
            schema={
                "benchmark_row_id": pl.Utf8,
                "sentence_index": pl.Int64,
                "sentence_text": pl.Utf8,
                "review_case_id": pl.Utf8,
                "neighbor_kind": pl.Utf8,
            }
        )
    )
    lookup = {
        (row["review_case_id"], row["neighbor_kind"]): row.get("sentence_text")
        for row in neighbors.to_dicts()
    }
    for row in sample_rows:
        row["prev_text"] = lookup.get((row["review_case_id"], "prev_text"))
        row["next_text"] = lookup.get((row["review_case_id"], "next_text"))
    return pl.DataFrame(sample_rows)


def _finalize_sample_frame(
    sample_df: pl.DataFrame,
    parquet_paths: tuple[Path, ...],
    text_scopes: tuple[str, ...],
    batch_size: int,
) -> pl.DataFrame:
    ordered = sample_df.sort("sample_key")
    rows = []
    for index, row in enumerate(ordered.to_dicts(), start=1):
        rows.append(
            {
                **row,
                "sample_order": index,
                "review_case_id": f"finbert_review_{index:06d}",
            }
        )
    with_ids = pl.DataFrame(rows)
    with_context = _attach_neighbor_context(with_ids, parquet_paths, text_scopes, batch_size)
    for column in SAMPLE_OUTPUT_COLUMNS:
        if column not in with_context.columns:
            with_context = with_context.with_columns(pl.lit(None).alias(column))
    return with_context.select(list(SAMPLE_OUTPUT_COLUMNS)).sort("sample_order")


def _chunk_rows(rows: list[dict[str, Any]], chunk_count: int) -> list[list[dict[str, Any]]]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive.")
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(chunk_count)]
    for index, row in enumerate(rows):
        chunks[index % chunk_count].append(row)
    return chunks


def _json_ready_record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_default(value) if hasattr(value, "isoformat") else value for key, value in row.items()}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(_json_ready_record(record), sort_keys=True, default=_json_default) + "\n")


def _labeling_input_record(row: dict[str, Any], *, pass_id: str | None = None) -> dict[str, Any]:
    payload = {
        "review_case_id": row["review_case_id"],
        "benchmark_sentence_id": row["benchmark_sentence_id"],
        "doc_id": row["doc_id"],
        "filing_year": row["filing_year"],
        "text_scope": row["text_scope"],
        "benchmark_item_code": row["benchmark_item_code"],
        "sentence_index": row["sentence_index"],
        "prev_text": row.get("prev_text"),
        "sentence_text": row["sentence_text"],
        "next_text": row.get("next_text"),
        "predicted_label": row["predicted_label"],
        "negative_prob": row["negative_prob"],
        "neutral_prob": row["neutral_prob"],
        "positive_prob": row["positive_prob"],
        "probability_majority_bucket": row["probability_majority_bucket"],
        "finbert_token_bucket_512": row["finbert_token_bucket_512"],
    }
    if pass_id is not None:
        payload["labeling_pass_id"] = pass_id
    return payload


def _write_labeling_chunks(sample_df: pl.DataFrame, output_dir: Path, chunk_count: int) -> Path:
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    rows = sample_df.to_dicts()
    chunks = _chunk_rows(rows, chunk_count)
    for index, chunk in enumerate(chunks, start=1):
        _write_jsonl(
            chunk_dir / f"chunk_{index:02d}.jsonl",
            [_labeling_input_record(row) for row in chunk],
        )

    pass_specs = (
        ("pass_a_low", "gpt-5.5", "low"),
        ("pass_b_medium", "gpt-5.5", "medium"),
    )
    pass_root = output_dir / "llm_pass_chunks"
    for pass_id, model, effort in pass_specs:
        pass_dir = pass_root / pass_id
        pass_dir.mkdir(parents=True, exist_ok=True)
        for index, chunk in enumerate(chunks, start=1):
            records = []
            for row in chunk:
                record = _labeling_input_record(row, pass_id=pass_id)
                record["recommended_model"] = model
                record["recommended_reasoning_effort"] = effort
                records.append(record)
            _write_jsonl(pass_dir / f"chunk_{index:02d}.jsonl", records)
    return chunk_dir


def _write_labeling_prompt(path: Path) -> None:
    text = """# FinBERT Negative/Adverse Sentence Labeling Rubric

You are labeling SEC 10-K sentences for a binary validation of FinBERT's negative/adverse class.

Return one JSON object per input row with these fields:

- `review_case_id`
- `gold_negative`: one of `yes`, `no`, `uncertain`
- `gold_sentiment`: one of `negative`, `neutral`, `positive`, `mixed`, `uncertain`
- `confidence`: one of `high`, `medium`, `low`
- `issue_flags`: an array using values such as `table_fragment`, `heading`, `sentence_fragment`, `boilerplate`, `needs_context`, or `none`
- `evidence`: a short phrase explaining the judgment

Use `gold_negative=yes` when the sentence itself communicates adverse business conditions, risks, losses, uncertainty, deterioration, constraints, litigation/regulatory problems, liquidity pressure, impairment, default, or similarly negative financial/business content.

Use `gold_negative=no` for neutral factual descriptions, positive/improving performance, generic company background, headings, and table fragments that do not themselves express adverse content.

Use `uncertain` only when the sentence cannot be judged from the sentence plus immediate context.
"""
    path.write_text(text, encoding="utf-8")


def _html_json_payload(sample_df: pl.DataFrame) -> str:
    payload = json.dumps(
        [_json_ready_record(row) for row in sample_df.to_dicts()],
        default=_json_default,
        ensure_ascii=True,
    )
    return payload.replace("</", "<\\/")


def _write_review_html(path: Path, sample_df: pl.DataFrame) -> None:
    data_json = _html_json_payload(sample_df)
    text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FinBERT Negative/Adverse Review</title>
<style>
:root {{
  --bg: #f7f8fa;
  --panel: #ffffff;
  --ink: #1f2933;
  --muted: #64748b;
  --line: #d8dee9;
  --accent: #0f766e;
  --warn: #b45309;
  --bad: #b91c1c;
  --good: #15803d;
}}
body {{ margin: 0; font-family: Arial, sans-serif; background: var(--bg); color: var(--ink); }}
header {{ position: sticky; top: 0; z-index: 2; background: var(--panel); border-bottom: 1px solid var(--line); padding: 12px 16px; }}
h1 {{ margin: 0 0 10px 0; font-size: 20px; }}
.controls {{ display: grid; grid-template-columns: repeat(7, minmax(120px, 1fr)); gap: 8px; align-items: end; }}
label {{ display: grid; gap: 3px; font-size: 12px; color: var(--muted); }}
select, input[type="search"], input[type="file"], textarea {{ font: inherit; border: 1px solid var(--line); border-radius: 6px; padding: 6px; background: white; }}
button {{ border: 1px solid var(--accent); background: var(--accent); color: white; border-radius: 6px; padding: 7px 10px; cursor: pointer; }}
button.secondary {{ background: white; color: var(--accent); }}
main {{ padding: 12px 16px 40px; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
.stat {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; min-width: 110px; }}
.stat strong {{ display: block; font-size: 18px; }}
.card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin-bottom: 10px; padding: 12px; }}
.card.reviewed {{ border-left: 5px solid var(--good); }}
.card.uncertain {{ border-left: 5px solid var(--warn); }}
.card.disagree {{ border-left: 5px solid var(--bad); }}
.meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
.pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; font-size: 12px; color: var(--muted); background: #fbfcfd; }}
.sentence {{ font-size: 15px; line-height: 1.45; white-space: pre-wrap; margin: 8px 0; }}
.context {{ color: var(--muted); font-size: 13px; white-space: pre-wrap; margin: 5px 0; }}
.review-grid {{ display: grid; grid-template-columns: 130px 130px 130px 1fr auto; gap: 8px; align-items: start; margin-top: 10px; }}
.llm {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
textarea {{ width: 100%; min-height: 44px; resize: vertical; box-sizing: border-box; }}
.hidden {{ display: none; }}
@media (max-width: 980px) {{
  .controls {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
  .review-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <h1>FinBERT Negative/Adverse Sentence Review</h1>
  <div class="controls">
    <label>Search<input id="search" type="search" placeholder="sentence, doc, id"></label>
    <label>Scope<select id="scopeFilter"><option value="">All</option></select></label>
    <label>Predicted<select id="predFilter"><option value="">All</option></select></label>
    <label>Majority<select id="bucketFilter"><option value="">All</option></select></label>
    <label>Agreement<select id="agreementFilter"><option value="">All</option><option value="disagree">Disagree</option><option value="agree">Agree</option><option value="missing">Missing LLM</option></select></label>
    <label>Confusion<select id="confusionFilter"><option value="">All</option><option>TP</option><option>FP</option><option>FN</option><option>TN</option><option>uncertain</option><option>unreviewed</option></select></label>
    <label>Import labels/review<input id="importFile" type="file" accept=".json,.jsonl"></label>
  </div>
  <div class="controls" style="margin-top:8px">
    <button id="exportJson">Export JSON</button>
    <button id="exportCsv" class="secondary">Export CSV</button>
    <button id="clearLocal" class="secondary">Clear Local Review</button>
  </div>
</header>
<main>
  <div id="stats" class="stats"></div>
  <div id="rows"></div>
</main>
<script>
const SAMPLE_ROWS = {data_json};
const STORAGE_KEY = "finbert_negative_review_v1";
const LLM_STORAGE_KEY = "finbert_negative_review_llm_labels_v1";
let reviews = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
let llmLabels = JSON.parse(localStorage.getItem(LLM_STORAGE_KEY) || "{{}}");

function norm(v) {{
  if (v === undefined || v === null) return "";
  return String(v).trim().toLowerCase();
}}
function labelFor(row) {{
  const review = reviews[row.review_case_id] || {{}};
  const human = norm(review.human_gold_negative || review.final_gold_negative || review.gold_negative);
  if (human) return human;
  const a = norm((llmLabels[row.review_case_id] || {{}}).llm_a_gold_negative);
  const b = norm((llmLabels[row.review_case_id] || {{}}).llm_b_gold_negative);
  if (a && b && a === b && a !== "uncertain") return a;
  return "";
}}
function llmAgreement(row) {{
  const labels = llmLabels[row.review_case_id] || {{}};
  const a = norm(labels.llm_a_gold_negative);
  const b = norm(labels.llm_b_gold_negative);
  if (!a || !b) return "missing";
  return a === b ? "agree" : "disagree";
}}
function confusion(row) {{
  const label = labelFor(row);
  if (!label) return "unreviewed";
  if (label === "uncertain") return "uncertain";
  const pred = row.predicted_label === "negative";
  const gold = label === "yes";
  if (pred && gold) return "TP";
  if (pred && !gold) return "FP";
  if (!pred && gold) return "FN";
  return "TN";
}}
function priority(row) {{
  const agreement = llmAgreement(row);
  const bucket = row.probability_majority_bucket;
  if (agreement === "disagree") return 0;
  if (labelFor(row) === "uncertain") return 1;
  if (bucket === "no_majority") return 2;
  if (row.predicted_label === "negative") return 3;
  return 4;
}}
function optionFill(id, values) {{
  const el = document.getElementById(id);
  [...new Set(values)].sort().forEach(v => {{
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    el.appendChild(opt);
  }});
}}
optionFill("scopeFilter", SAMPLE_ROWS.map(r => r.text_scope));
optionFill("predFilter", SAMPLE_ROWS.map(r => r.predicted_label));
optionFill("bucketFilter", SAMPLE_ROWS.map(r => r.probability_majority_bucket));

function saveReview(id, patch) {{
  reviews[id] = Object.assign({{}}, reviews[id] || {{}}, patch);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(reviews));
  render();
}}
function rowMatches(row) {{
  const q = norm(document.getElementById("search").value);
  if (q) {{
    const hay = norm([row.review_case_id, row.benchmark_sentence_id, row.doc_id, row.sentence_text].join(" "));
    if (!hay.includes(q)) return false;
  }}
  if (document.getElementById("scopeFilter").value && row.text_scope !== document.getElementById("scopeFilter").value) return false;
  if (document.getElementById("predFilter").value && row.predicted_label !== document.getElementById("predFilter").value) return false;
  if (document.getElementById("bucketFilter").value && row.probability_majority_bucket !== document.getElementById("bucketFilter").value) return false;
  if (document.getElementById("agreementFilter").value && llmAgreement(row) !== document.getElementById("agreementFilter").value) return false;
  if (document.getElementById("confusionFilter").value && confusion(row) !== document.getElementById("confusionFilter").value) return false;
  return true;
}}
function escapeHtml(s) {{
  return String(s ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
}}
function renderStats(filtered) {{
  const reviewed = SAMPLE_ROWS.filter(r => labelFor(r)).length;
  const cells = {{}};
  SAMPLE_ROWS.forEach(r => cells[confusion(r)] = (cells[confusion(r)] || 0) + 1);
  const stats = [
    ["Rows", SAMPLE_ROWS.length],
    ["Visible", filtered.length],
    ["Reviewed", reviewed],
    ["TP", cells.TP || 0],
    ["FP", cells.FP || 0],
    ["FN", cells.FN || 0],
    ["TN", cells.TN || 0],
    ["Uncertain", cells.uncertain || 0],
  ];
  document.getElementById("stats").innerHTML = stats.map(([k,v]) => `<div class="stat"><strong>${{v}}</strong>${{k}}</div>`).join("");
}}
function render() {{
  const rows = SAMPLE_ROWS.filter(rowMatches).sort((a,b) => priority(a) - priority(b) || a.sample_order - b.sample_order);
  renderStats(rows);
  document.getElementById("rows").innerHTML = rows.map(row => {{
    const review = reviews[row.review_case_id] || {{}};
    const labels = llmLabels[row.review_case_id] || {{}};
    const cell = confusion(row);
    const agree = llmAgreement(row);
    const cls = agree === "disagree" ? "disagree" : (cell === "uncertain" ? "uncertain" : (cell !== "unreviewed" ? "reviewed" : ""));
    return `<section class="card ${{cls}}">
      <div class="meta">
        <span class="pill">${{escapeHtml(row.review_case_id)}}</span>
        <span class="pill">${{escapeHtml(row.text_scope)}}</span>
        <span class="pill">pred=${{escapeHtml(row.predicted_label)}}</span>
        <span class="pill">${{escapeHtml(row.probability_majority_bucket)}}</span>
        <span class="pill">neg=${{Number(row.negative_prob).toFixed(3)}}</span>
        <span class="pill">neu=${{Number(row.neutral_prob).toFixed(3)}}</span>
        <span class="pill">pos=${{Number(row.positive_prob).toFixed(3)}}</span>
        <span class="pill">cell=${{cell}}</span>
      </div>
      <div class="context">${{escapeHtml(row.prev_text || "")}}</div>
      <div class="sentence">${{escapeHtml(row.sentence_text)}}</div>
      <div class="context">${{escapeHtml(row.next_text || "")}}</div>
      <div class="llm">LLM A: ${{escapeHtml(labels.llm_a_gold_negative || "")}} ${{escapeHtml(labels.llm_a_confidence || "")}} | LLM B: ${{escapeHtml(labels.llm_b_gold_negative || "")}} ${{escapeHtml(labels.llm_b_confidence || "")}}</div>
      <div class="review-grid">
        <label>Gold negative
          <select onchange="saveReview('${{row.review_case_id}}', {{human_gold_negative:this.value}})">
            <option value=""></option><option value="yes" ${{norm(review.human_gold_negative)==="yes"?"selected":""}}>yes</option><option value="no" ${{norm(review.human_gold_negative)==="no"?"selected":""}}>no</option><option value="uncertain" ${{norm(review.human_gold_negative)==="uncertain"?"selected":""}}>uncertain</option>
          </select>
        </label>
        <label>Gold sentiment
          <select onchange="saveReview('${{row.review_case_id}}', {{human_gold_sentiment:this.value}})">
            <option value=""></option><option value="negative" ${{norm(review.human_gold_sentiment)==="negative"?"selected":""}}>negative</option><option value="neutral" ${{norm(review.human_gold_sentiment)==="neutral"?"selected":""}}>neutral</option><option value="positive" ${{norm(review.human_gold_sentiment)==="positive"?"selected":""}}>positive</option><option value="mixed" ${{norm(review.human_gold_sentiment)==="mixed"?"selected":""}}>mixed</option><option value="uncertain" ${{norm(review.human_gold_sentiment)==="uncertain"?"selected":""}}>uncertain</option>
          </select>
        </label>
        <label>Confidence
          <select onchange="saveReview('${{row.review_case_id}}', {{human_confidence:this.value}})">
            <option value=""></option><option value="high" ${{norm(review.human_confidence)==="high"?"selected":""}}>high</option><option value="medium" ${{norm(review.human_confidence)==="medium"?"selected":""}}>medium</option><option value="low" ${{norm(review.human_confidence)==="low"?"selected":""}}>low</option>
          </select>
        </label>
        <label>Notes<textarea onchange="saveReview('${{row.review_case_id}}', {{human_notes:this.value}})">${{escapeHtml(review.human_notes || "")}}</textarea></label>
        <button class="secondary" onclick="acceptConsensus('${{row.review_case_id}}')">Accept consensus</button>
      </div>
    </section>`;
  }}).join("");
}}
function acceptConsensus(id) {{
  const labels = llmLabels[id] || {{}};
  const a = norm(labels.llm_a_gold_negative);
  const b = norm(labels.llm_b_gold_negative);
  if (a && b && a === b) saveReview(id, {{human_gold_negative:a, human_confidence:"medium"}});
}}
function exportPayload() {{
  return {{
    exported_at: new Date().toISOString(),
    rows: SAMPLE_ROWS.map(row => Object.assign({{review_case_id: row.review_case_id, benchmark_sentence_id: row.benchmark_sentence_id}}, llmLabels[row.review_case_id] || {{}}, reviews[row.review_case_id] || {{}}))
  }};
}}
function download(name, text, type) {{
  const blob = new Blob([text], {{type}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}}
document.getElementById("exportJson").onclick = () => download("human_review.json", JSON.stringify(exportPayload(), null, 2), "application/json");
document.getElementById("exportCsv").onclick = () => {{
  const rows = exportPayload().rows;
  const cols = ["review_case_id","benchmark_sentence_id","human_gold_negative","human_gold_sentiment","human_confidence","human_notes","llm_a_gold_negative","llm_b_gold_negative"];
  const csv = [cols.join(",")].concat(rows.map(r => cols.map(c => JSON.stringify(r[c] ?? "")).join(","))).join("\\n");
  download("human_review.csv", csv, "text/csv");
}};
document.getElementById("clearLocal").onclick = () => {{ if (confirm("Clear local review and imported LLM labels?")) {{ reviews = {{}}; llmLabels = {{}}; localStorage.removeItem(STORAGE_KEY); localStorage.removeItem(LLM_STORAGE_KEY); render(); }} }};
document.getElementById("importFile").onchange = async (event) => {{
  const file = event.target.files[0];
  if (!file) return;
  const text = await file.text();
  let records = [];
  if (file.name.endsWith(".jsonl")) records = text.trim().split(/\\n+/).map(line => JSON.parse(line));
  else {{
    const parsed = JSON.parse(text);
    records = Array.isArray(parsed) ? parsed : (parsed.rows || parsed.labels || []);
  }}
  records.forEach(record => {{
    const id = record.review_case_id;
    if (!id) return;
    if (record.labeling_pass_id === "pass_a_low" || record.pass_id === "pass_a_low") {{
      llmLabels[id] = Object.assign({{}}, llmLabels[id] || {{}}, {{llm_a_gold_negative: record.gold_negative, llm_a_confidence: record.confidence, llm_a_evidence: record.evidence}});
    }} else if (record.labeling_pass_id === "pass_b_medium" || record.pass_id === "pass_b_medium") {{
      llmLabels[id] = Object.assign({{}}, llmLabels[id] || {{}}, {{llm_b_gold_negative: record.gold_negative, llm_b_confidence: record.confidence, llm_b_evidence: record.evidence}});
    }} else {{
      reviews[id] = Object.assign({{}}, reviews[id] || {{}}, record);
    }}
  }});
  localStorage.setItem(STORAGE_KEY, JSON.stringify(reviews));
  localStorage.setItem(LLM_STORAGE_KEY, JSON.stringify(llmLabels));
  render();
}};
["search","scopeFilter","predFilter","bucketFilter","agreementFilter","confusionFilter"].forEach(id => document.getElementById(id).addEventListener("input", render));
render();
</script>
</body>
</html>
"""
    path.write_text(text, encoding="utf-8")


def build_finbert_sentence_confusion_review_pack(
    sentence_scores_dir: Path,
    *,
    output_dir: Path,
    cfg: FinbertSentenceConfusionReviewConfig | None = None,
    counts_only: bool = False,
) -> FinbertSentenceConfusionReviewArtifacts:
    cfg = cfg or FinbertSentenceConfusionReviewConfig()
    if cfg.sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if cfg.chunk_count <= 0:
        raise ValueError("chunk_count must be positive.")

    resolved_output_dir = output_dir.resolve()
    resolved_sentence_scores_dir = _resolve_sentence_scores_by_year_dir(sentence_scores_dir)
    _, stratum_counts, count_summary = _counts_preflight(
        resolved_sentence_scores_dir,
        resolved_output_dir,
        text_scopes=cfg.text_scopes,
        sample_size=cfg.sample_size,
        seed=cfg.seed,
        include_no_majority=cfg.include_no_majority,
    )
    allocation_source_df = stratum_counts.filter(pl.col("sample_eligible")).drop(
        ["sample_eligible", "exclusion_reason"]
    )
    allocation_df = _allocate_sample_counts(
        allocation_source_df,
        cfg.sample_size,
        allocation_mode=cfg.allocation_mode,
    )
    allocation_df.write_csv(resolved_output_dir / "sample_allocation_by_stratum.csv")

    sample_path: Path | None = None
    sample_csv_path: Path | None = None
    review_html_path: Path | None = None
    labeling_prompt_path: Path | None = None
    chunk_dir: Path | None = None
    sample_row_count = 0
    oversampling_factor: float | None = None
    sampling_attempts: list[dict[str, Any]] = []

    if not counts_only:
        parquet_paths = _sentence_score_parquet_paths(resolved_sentence_scores_dir)
        selected, sampling_attempts = _collect_exact_sample_streaming(
            parquet_paths,
            allocation_df,
            text_scopes=cfg.text_scopes,
            seed=cfg.seed,
            batch_size=cfg.stream_batch_size,
        )
        sample_df = _finalize_sample_frame(
            selected,
            parquet_paths,
            cfg.text_scopes,
            cfg.stream_batch_size,
        )
        sample_row_count = int(sample_df.height)
        sample_path = resolved_output_dir / "sample.parquet"
        sample_csv_path = resolved_output_dir / "sample.csv"
        review_html_path = resolved_output_dir / "review.html"
        labeling_prompt_path = resolved_output_dir / "labeling_prompt.md"
        sample_df.write_parquet(sample_path, compression="zstd")
        sample_df.write_csv(sample_csv_path)
        chunk_dir = _write_labeling_chunks(sample_df, resolved_output_dir, cfg.chunk_count)
        _write_labeling_prompt(labeling_prompt_path)
        _write_review_html(review_html_path, sample_df)

    manifest = {
        "generated_at": _utc_now_iso(),
        "counts_only": counts_only,
        "config": asdict(cfg),
        "sentence_scores_by_year_dir": str(resolved_sentence_scores_dir),
        "output_dir": str(resolved_output_dir),
        "required_columns": list(REQUIRED_SENTENCE_SCORE_COLUMNS),
        "optional_columns": list(OPTIONAL_SENTENCE_SCORE_COLUMNS),
        "majority_bucket_rule": {
            "negative_majority": "negative_prob > 0.5",
            "neutral_majority": "neutral_prob > 0.5",
            "positive_majority": "positive_prob > 0.5",
            "no_majority": "all class probabilities <= 0.5",
        },
        "stratum_columns": list(STRATUM_COLUMNS),
        "population_summary": count_summary,
        "allocation": allocation_df.to_dicts(),
        "sampling_attempts": sampling_attempts,
        "oversampling_factor": oversampling_factor,
        "sample_row_count": sample_row_count,
        "artifacts": {
            "population_counts_by_majority_bucket": "population_counts_by_majority_bucket.csv",
            "population_counts_by_stratum": "population_counts_by_stratum.csv",
            "population_counts_summary": "population_counts_summary.json",
            "sample_allocation_by_stratum": "sample_allocation_by_stratum.csv",
            "sample": "sample.parquet" if sample_path is not None else None,
            "sample_csv": "sample.csv" if sample_csv_path is not None else None,
            "chunks": "chunks" if chunk_dir is not None else None,
            "llm_pass_chunks": "llm_pass_chunks" if chunk_dir is not None else None,
            "labeling_prompt": "labeling_prompt.md" if labeling_prompt_path is not None else None,
            "review_html": "review.html" if review_html_path is not None else None,
        },
    }
    manifest_path = resolved_output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    return FinbertSentenceConfusionReviewArtifacts(
        output_dir=resolved_output_dir,
        manifest_path=manifest_path,
        population_counts_by_majority_bucket_path=resolved_output_dir / "population_counts_by_majority_bucket.csv",
        population_counts_by_stratum_path=resolved_output_dir / "population_counts_by_stratum.csv",
        population_counts_summary_path=resolved_output_dir / "population_counts_summary.json",
        sample_path=sample_path,
        sample_csv_path=sample_csv_path,
        review_html_path=review_html_path,
        labeling_prompt_path=labeling_prompt_path,
        chunk_dir=chunk_dir,
        sample_row_count=sample_row_count,
        counts_only=counts_only,
        oversampling_factor=oversampling_factor,
    )


def _load_review_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return payload["rows"]
        if isinstance(payload.get("labels"), list):
            return payload["labels"]
        if isinstance(payload.get("reviews"), dict):
            return [
                {"review_case_id": review_case_id, **record}
                for review_case_id, record in payload["reviews"].items()
            ]
    raise ValueError(f"Unsupported review JSON shape: {path}")


def _normalize_binary_label(value: Any) -> str:
    raw = "" if value is None else str(value).strip().lower()
    if raw in {"yes", "y", "true", "1", "negative", "adverse"}:
        return "yes"
    if raw in {"no", "n", "false", "0", "not_negative", "non_negative", "neutral", "positive"}:
        return "no"
    if raw in {"uncertain", "unknown", "unsure", "maybe", "ambiguous"}:
        return "uncertain"
    return ""


def _final_gold_negative(record: dict[str, Any]) -> tuple[str, str]:
    for key in ("human_gold_negative", "final_gold_negative", "gold_negative"):
        label = _normalize_binary_label(record.get(key))
        if label:
            return label, key
    a_label = _normalize_binary_label(record.get("llm_a_gold_negative"))
    b_label = _normalize_binary_label(record.get("llm_b_gold_negative"))
    if a_label and b_label and a_label == b_label and a_label != "uncertain":
        return a_label, "llm_consensus"
    return "", "missing"


def _confusion_cell(predicted_label: str, gold_negative: str) -> str:
    if gold_negative not in {"yes", "no"}:
        return "uncertain"
    predicted_negative = predicted_label == "negative"
    actual_negative = gold_negative == "yes"
    if predicted_negative and actual_negative:
        return "TP"
    if predicted_negative and not actual_negative:
        return "FP"
    if not predicted_negative and actual_negative:
        return "FN"
    return "TN"


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _metric_payload(counts: dict[str, float]) -> dict[str, float | None]:
    tp = counts.get("TP", 0.0)
    fp = counts.get("FP", 0.0)
    fn = counts.get("FN", 0.0)
    tn = counts.get("TN", 0.0)
    total = tp + fp + fn + tn
    return {
        "accuracy": _safe_divide(tp + tn, total),
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "false_negative_rate": _safe_divide(fn, fn + tp),
        "resolved_count": total,
    }


def _counts_by_cell(rows: list[dict[str, Any]], *, weighted: bool) -> dict[str, float]:
    counts = {cell: 0.0 for cell in CONFUSION_VALUES}
    for row in rows:
        weight = float(row.get("sample_weight") or 1.0) if weighted else 1.0
        counts[str(row["confusion_cell"])] = counts.get(str(row["confusion_cell"]), 0.0) + weight
    return counts


def _uncertain_metric_bounds(rows: list[dict[str, Any]], *, weighted: bool) -> dict[str, dict[str, float | None]]:
    resolved = _counts_by_cell([row for row in rows if row["confusion_cell"] != "uncertain"], weighted=weighted)
    uncertain_pred_pos = 0.0
    uncertain_pred_neg = 0.0
    for row in rows:
        if row["confusion_cell"] != "uncertain":
            continue
        weight = float(row.get("sample_weight") or 1.0) if weighted else 1.0
        if row["predicted_label"] == "negative":
            uncertain_pred_pos += weight
        else:
            uncertain_pred_neg += weight
    tp = resolved.get("TP", 0.0)
    fp = resolved.get("FP", 0.0)
    fn = resolved.get("FN", 0.0)
    tn = resolved.get("TN", 0.0)
    total = tp + fp + fn + tn + uncertain_pred_pos + uncertain_pred_neg
    return {
        "precision": {
            "lower": _safe_divide(tp, tp + fp + uncertain_pred_pos),
            "upper": _safe_divide(tp + uncertain_pred_pos, tp + uncertain_pred_pos + fp),
        },
        "recall": {
            "lower": _safe_divide(tp, tp + fn + uncertain_pred_neg),
            "upper": _safe_divide(tp + uncertain_pred_pos, tp + uncertain_pred_pos + fn),
        },
        "accuracy": {
            "lower": _safe_divide(tp + tn, total),
            "upper": _safe_divide(tp + tn + uncertain_pred_pos + uncertain_pred_neg, total),
        },
    }


def _metrics_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# FinBERT Negative/Adverse Sentence Review Metrics",
        "",
        f"Reviewed rows: {metrics['reviewed_row_count']} / {metrics['sample_row_count']}",
        f"Resolved rows: {metrics['unweighted']['resolved_count']}",
        f"Uncertain rows: {metrics['unweighted_counts_by_cell'].get('uncertain', 0)}",
        "",
        "## Unweighted Metrics",
        "",
    ]
    for key, value in metrics["unweighted"].items():
        if key == "resolved_count":
            continue
        lines.append(f"- {key}: {value:.4f}" if value is not None else f"- {key}: n/a")
    lines.extend(["", "## Weighted Metrics", ""])
    for key, value in metrics["weighted"].items():
        if key == "resolved_count":
            continue
        lines.append(f"- {key}: {value:.4f}" if value is not None else f"- {key}: n/a")
    lines.extend(
        [
            "",
            "## Thesis Caveat",
            "",
            (
                "These estimates are based on a stratified, population-weighted sentence sample from Item 1A "
                "and Item 7 FinBERT outputs. The audited positive class is substantive negative/adverse content; "
                "neutral and positive FinBERT predictions are both treated as not predicted adverse in the binary "
                "confusion matrix. Uncertain rows are excluded from primary metrics and reported with bounds."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _write_examples_by_cell(path: Path, rows: list[dict[str, Any]], per_cell: int = 10) -> None:
    lines = ["# Examples by Confusion Cell", ""]
    for cell in CONFUSION_VALUES:
        examples = [row for row in rows if row["confusion_cell"] == cell][:per_cell]
        lines.extend([f"## {cell}", ""])
        if not examples:
            lines.extend(["No rows.", ""])
            continue
        for row in examples:
            sentence = " ".join(str(row.get("sentence_text") or "").split())
            if len(sentence) > 260:
                sentence = sentence[:257] + "..."
            lines.append(
                f"- `{row['review_case_id']}` `{row['text_scope']}` pred={row['predicted_label']} "
                f"gold={row.get('gold_negative_final')}: {sentence}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _bucket_metric_rows(reviewed_df: pl.DataFrame) -> pl.DataFrame:
    if reviewed_df.is_empty():
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    for bucket in sorted(set(reviewed_df["probability_majority_bucket"].drop_nulls().to_list())):
        bucket_rows = reviewed_df.filter(pl.col("probability_majority_bucket") == bucket).to_dicts()
        counts = _counts_by_cell(bucket_rows, weighted=True)
        metrics = _metric_payload({key: value for key, value in counts.items() if key != "uncertain"})
        rows.append(
            {
                "probability_majority_bucket": bucket,
                "row_count": len(bucket_rows),
                "weighted_TP": counts.get("TP", 0.0),
                "weighted_FP": counts.get("FP", 0.0),
                "weighted_FN": counts.get("FN", 0.0),
                "weighted_TN": counts.get("TN", 0.0),
                "weighted_uncertain": counts.get("uncertain", 0.0),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
            }
        )
    return pl.DataFrame(rows)


def _csv_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_rows: list[dict[str, Any]] = []
    for row in rows:
        safe_rows.append(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    return safe_rows


def summarize_finbert_sentence_confusion_review(
    review_dir: Path,
    *,
    human_review_path: Path,
    output_dir: Path | None = None,
) -> FinbertSentenceConfusionSummaryArtifacts:
    resolved_review_dir = review_dir.resolve()
    sample_path = resolved_review_dir / "sample.parquet"
    if not sample_path.exists():
        raise FileNotFoundError(f"Review directory is missing sample.parquet: {sample_path}")
    labels = _load_review_records(human_review_path.resolve())
    labels_by_case_id = {
        str(record["review_case_id"]): record
        for record in labels
        if record.get("review_case_id") is not None
    }

    sample_df = pl.read_parquet(sample_path)
    reviewed_rows: list[dict[str, Any]] = []
    missing_review_case_ids: list[str] = []
    for row in sample_df.to_dicts():
        record = labels_by_case_id.get(str(row["review_case_id"]), {})
        gold_negative, label_source = _final_gold_negative(record)
        if not gold_negative:
            missing_review_case_ids.append(str(row["review_case_id"]))
            gold_negative = "uncertain"
        reviewed_rows.append(
            {
                **row,
                **{f"review_{key}": value for key, value in record.items() if key not in row},
                "gold_negative_final": gold_negative,
                "gold_negative_source": label_source,
                "confusion_cell": _confusion_cell(str(row["predicted_label"]), gold_negative),
            }
        )

    reviewed_df = pl.DataFrame(reviewed_rows)
    out_dir = (output_dir.resolve() if output_dir is not None else resolved_review_dir / "review_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    reviewed_cases_path = out_dir / "reviewed_cases.parquet"
    reviewed_cases_csv_path = out_dir / "reviewed_cases.csv"
    reviewed_df.write_parquet(reviewed_cases_path, compression="zstd")

    rows = reviewed_df.to_dicts()
    pl.DataFrame(_csv_safe_rows(rows)).write_csv(reviewed_cases_csv_path)
    unweighted_counts = _counts_by_cell(rows, weighted=False)
    weighted_counts = _counts_by_cell(rows, weighted=True)
    resolved_unweighted_counts = {key: value for key, value in unweighted_counts.items() if key != "uncertain"}
    resolved_weighted_counts = {key: value for key, value in weighted_counts.items() if key != "uncertain"}

    confusion_rows = [
        {
            "confusion_cell": cell,
            "unweighted_count": unweighted_counts.get(cell, 0.0),
            "weighted_count": weighted_counts.get(cell, 0.0),
        }
        for cell in CONFUSION_VALUES
    ]
    confusion_matrix_path = out_dir / "confusion_matrix.csv"
    pl.DataFrame(confusion_rows).write_csv(confusion_matrix_path)

    metrics = {
        "generated_at": _utc_now_iso(),
        "review_dir": str(resolved_review_dir),
        "human_review_path": str(human_review_path.resolve()),
        "sample_row_count": int(sample_df.height),
        "reviewed_row_count": int(sample_df.height - len(missing_review_case_ids)),
        "missing_review_case_count": len(missing_review_case_ids),
        "missing_review_case_ids": missing_review_case_ids[:50],
        "unweighted_counts_by_cell": unweighted_counts,
        "weighted_counts_by_cell": weighted_counts,
        "unweighted": _metric_payload(resolved_unweighted_counts),
        "weighted": _metric_payload(resolved_weighted_counts),
        "uncertain_bounds_unweighted": _uncertain_metric_bounds(rows, weighted=False),
        "uncertain_bounds_weighted": _uncertain_metric_bounds(rows, weighted=True),
    }
    metrics_json_path = out_dir / "metrics.json"
    _write_json(metrics_json_path, metrics)

    metrics_markdown_path = out_dir / "metrics.md"
    metrics_markdown_path.write_text(_metrics_markdown(metrics), encoding="utf-8")

    majority_bucket_metrics_path = out_dir / "majority_bucket_metrics.csv"
    bucket_metrics = _bucket_metric_rows(reviewed_df)
    if bucket_metrics.is_empty():
        majority_bucket_metrics_path.write_text("", encoding="utf-8")
    else:
        bucket_metrics.write_csv(majority_bucket_metrics_path)

    examples_by_cell_path = out_dir / "examples_by_cell.md"
    _write_examples_by_cell(examples_by_cell_path, rows)

    return FinbertSentenceConfusionSummaryArtifacts(
        output_dir=out_dir,
        reviewed_cases_path=reviewed_cases_path,
        reviewed_cases_csv_path=reviewed_cases_csv_path,
        confusion_matrix_path=confusion_matrix_path,
        metrics_json_path=metrics_json_path,
        metrics_markdown_path=metrics_markdown_path,
        majority_bucket_metrics_path=majority_bucket_metrics_path,
        examples_by_cell_path=examples_by_cell_path,
    )


__all__ = [
    "CONFUSION_VALUES",
    "FinbertSentenceConfusionReviewArtifacts",
    "FinbertSentenceConfusionReviewConfig",
    "FinbertSentenceConfusionSummaryArtifacts",
    "LABEL_VALUES",
    "MAJORITY_BUCKET_VALUES",
    "STRATUM_COLUMNS",
    "THESIS_REVIEW_TEXT_SCOPES",
    "add_probability_majority_bucket",
    "build_finbert_sentence_confusion_review_pack",
    "probability_majority_bucket_expr",
    "summarize_finbert_sentence_confusion_review",
]
